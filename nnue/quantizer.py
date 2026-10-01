from pathlib import Path

from .archs.chessnn import ChessNN
import numpy as np
from typing import TextIO


def write_array(f: TextIO, name: str, array: np.ndarray, max_val: int):
    array = np.asarray(array, dtype=np.float64)
    if len(array.shape) == 1:
        decl = f"const int16_t {name}[{len(array)}]"
    else:
        decl = f"const int16_t {name}[{len(array)}][{array.shape[1]}]"
    f.write(f"{decl} = {{")
    for i in range(len(array)):
        if len(array.shape) == 1:
            w = int(np.rint(array[i]))
            assert -max_val <= w <= max_val, f"Value {w} at index {i} exceeds int16 range after quantization"
            f.write(f"{w}, " if i < len(array) - 1 else f"{w}")
        else:
            f.write("{")
            for j in range(array.shape[1]):
                w = int(np.rint(array[i, j]))
                assert -max_val <= w <= max_val, f"Value {w} at index {i} exceeds int16 range after quantization"
                f.write(f"{w}, " if j < array.shape[1] - 1 else f"{w}")
            f.write("}, " if i < len(array) - 1 else "}")
    f.write("};\n")

    extern_decl = f"extern {decl};"
    return extern_decl

def quantize(model: ChessNN):
    int16_max = 32767
    state = model.state_dict()
    layer_max_values = {}
    factors = {}
    extern_decls = []
    
    stem = Path(model.model_path).stem
    Path("nnue/params").mkdir(parents=True, exist_ok=True)

    with open(f"nnue/params/{stem}.c", "w") as f:
        f.write("#include <stdint.h>\n\n")
        
        for name, param in state.items():
            array = param.cpu().numpy()
            layer = name.rsplit(".", 1)[0].replace(".", "_")
            max_value = float(np.max(np.abs(array))) if array.size else 0.0
            if np.isfinite(max_value):
                layer_max_values[layer] = max(layer_max_values.get(layer, 0.0), max_value)

        # Write the quantized parameters to C arrays
        for name, param in state.items():
            array = param.cpu().numpy()
            layer = name.rsplit(".", 1)[0].replace(".", "_")  # e.g. "fc1", "fc2", "fc3"
            name = stem + "_" + name.replace(".", "_")  # Replace dots with underscores for C variable names
            max_value = layer_max_values.get(layer, 0.0)
            
            # Transpose the weights of the first fully connected layer for better memory access patterns in C
            if len(array.shape) == 2 and len(array[0]) == 769:
                print(f"Transposing weights of layer {layer} for better memory access patterns in C")
                array = array.T
            
            if layer not in factors:
                if max_value <= 0 or not np.isfinite(max_value):
                    factor = 1
                else:
                    # Keep the scale as a Python int, but multiply through a float64 buffer
                    # later to avoid NumPy float32 overflow when max_value is extremely small.
                    factor = min(int16_max, max(1, int(int16_max / 16 / float(max_value))))
                assert factor > 0, f"Quantization factor {factor} for layer {layer} must be positive"
                factors[layer] = factor
            
            quantized = np.asarray(array, dtype=np.float64) * factors[layer]
            extern_decl = write_array(f, name, quantized, int16_max)  # Clamp to int16 range
            extern_decls.append(extern_decl)
    
        for layer, factor in factors.items():
            f.write(f"const int {stem}_{layer}_k = {factor};\n")
            
    print("Quantization complete. Parameters written to nnue/params.c")
    print("Add the following quantization factors and extern declarations:")

    for decl in extern_decls:
        print(decl)
    for layer, factor in factors.items():
        print(f"extern const int {stem}_{layer}_k;")
