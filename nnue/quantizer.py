from .archs.chessnn import ChessNN
import numpy as np
from typing import TextIO


def write_array(f: TextIO, name: str, array: np.ndarray, max_val: int, int_type: str):
    if len(array.shape) == 1:
        decl = f"const {int_type} {name}[{len(array)}]"
    else:
        decl = f"const {int_type} {name}[{len(array)}][{array.shape[1]}]"
    f.write(f"{decl} = {{")
    for i in range(len(array)):
        if len(array.shape) == 1:
            w = int(array[i])
            assert -max_val <= w <= max_val, f"Value {w} at index {i} exceeds {int_type} range after quantization"
            f.write(f"{w}, " if i < len(array) - 1 else f"{w}")
        else:
            f.write("{")
            for j in range(array.shape[1]):
                w = int(array[i, j])
                assert -max_val <= w <= max_val, f"Value {w} at index {i} exceeds {int_type} range after quantization"
                f.write(f"{w}, " if j < array.shape[1] - 1 else f"{w}")
            f.write("}, " if i < len(array) - 1 else "}")
    f.write("};\n")

    extern_decl = f"extern {decl};"
    return extern_decl

def quantize(model: ChessNN, int_type: str):
    scalar_type = np.dtype(int_type)
    int_max = np.iinfo(scalar_type).max
    c_int_type = int_type + "_t"
    state = model.state_dict()
    factors = {}
    extern_decls = []

    with open("nnue/params.c", "w") as f:
        f.write("#include <stdint.h>\n\n")
        
        # Write the quantized parameters to C arrays
        for name, param in state.items():
            array = param.cpu().numpy()
            layer = name.split(".")[0]  # e.g. "fc1", "fc2", "fc3"
            name = name.replace(".", "_")  # Replace dots with underscores for C variable names
            max_value = np.max(np.abs(array))
            
            # Transpose the weights of the first fully connected layer for better memory access patterns in C
            if len(array.shape) == 2 and len(array[0]) == 769:
                print(f"Transposing weights of layer {layer} for better memory access patterns in C")
                array = array.T
            
            if layer not in factors:
                factor = int(int_max // max_value) if max_value > 0 else 1
                assert factor > 0, f"Quantization factor {factor} for layer {layer} is invalid"
                factors[layer] = factor
            
            quantized = np.rint(array * factors[layer]).astype(scalar_type)
            extern_decl = write_array(f, name, quantized, int_max, c_int_type)
            extern_decls.append(extern_decl)

    print("Quantization complete. Parameters written to nnue/params.c")
    print("Add the following quantization factors and extern declarations:")

    for layer, factor in factors.items():
        print(f"const value_t {layer}_k = {factor};")
    for decl in extern_decls:
        print(decl)
