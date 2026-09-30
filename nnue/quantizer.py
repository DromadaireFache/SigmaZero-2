from pathlib import Path

from .archs.chessnn import ChessNN
import numpy as np
from typing import TextIO


def write_array(f: TextIO, name: str, array: np.ndarray, max_val: int):
    if len(array.shape) == 1:
        decl = f"const int16_t {name}[{len(array)}]"
    else:
        decl = f"const int16_t {name}[{len(array)}][{array.shape[1]}]"
    f.write(f"{decl} = {{")
    for i in range(len(array)):
        if len(array.shape) == 1:
            w = int(array[i])
            assert -max_val <= w <= max_val, f"Value {w} at index {i} exceeds int16 range after quantization"
            f.write(f"{w}, " if i < len(array) - 1 else f"{w}")
        else:
            f.write("{")
            for j in range(array.shape[1]):
                w = int(array[i, j])
                assert -max_val <= w <= max_val, f"Value {w} at index {i} exceeds int16 range after quantization"
                f.write(f"{w}, " if j < array.shape[1] - 1 else f"{w}")
            f.write("}, " if i < len(array) - 1 else "}")
    f.write("};\n")

    extern_decl = f"extern {decl};"
    return extern_decl

def quantize(model: ChessNN):
    int16_max = 32767
    state = model.state_dict()
    factors = {}
    extern_decls = []
    
    stem = Path(model.model_path).stem
    Path("nnue/params").mkdir(parents=True, exist_ok=True)

    with open(f"nnue/params/{stem}.c", "w") as f:
        f.write("#include <stdint.h>\n\n")
        
        # Write the quantized parameters to C arrays
        for name, param in state.items():
            array = param.cpu().numpy()
            layer = name.split(".")[0]  # e.g. "fc1", "fc2", "fc3"
            name = stem + "_" + name.replace(".", "_")  # Replace dots with underscores for C variable names
            max_value = np.max(np.abs(array))
            
            # Transpose the weights of the first fully connected layer for better memory access patterns in C
            if len(array.shape) == 2 and len(array[0]) == 769:
                print(f"Transposing weights of layer {layer} for better memory access patterns in C")
                array = array.T
            
            if layer not in factors:
                factor = int(int16_max // 16 // max_value) if max_value > 0 else 1
                assert int16_max >= factor > 0, f"Quantization factor {factor} for layer {layer} is out of int16 range"
                factors[layer] = factor
            
            extern_decl = write_array(f, name, array * factors[layer], int16_max)  # Clamp to int16 range
            extern_decls.append(extern_decl)
    
        for layer, factor in factors.items():
            f.write(f"const int {stem}_{layer}_k = {factor};\n")
            
    print("Quantization complete. Parameters written to nnue/params.c")
    print("Add the following quantization factors and extern declarations:")

    for decl in extern_decls:
        print(decl)
    for layer, factor in factors.items():
        print(f"extern const int {stem}_{layer}_k;")
