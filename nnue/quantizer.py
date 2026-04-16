from .archs.chessnn import ChessNN
import numpy as np
import typing


# Not sure if this will be useful since I plan to use torch.jit.script
def quantize(model: ChessNN):
    state = model.state_dict()
    QA, QB = 255, 64

    fc1_w = state["fc1.weight"].numpy() * QA
    fc2_w = state["fc2.weight"].numpy() * QB
    fc3_w = state["fc3.weight"].numpy() * QB
    fc1_b = state["fc1.bias"].numpy() * QA
    fc2_b = state["fc2.bias"].numpy() * QB
    fc3_b = state["fc3.bias"].numpy() * QB

    print(f"L0 weights range: [{fc1_w.min():.1f}, {fc1_w.max():.1f}] (int16 range: ±32767)")
    print(f"L1 weights range: [{fc2_w.min():.1f}, {fc2_w.max():.1f}] (int8 range: ±127)")
    print(f"L2 weights range: [{fc3_w.min():.1f}, {fc3_w.max():.1f}] (int8 range: ±127)")

    def write_array(f: typing.TextIO, name: str, array: np.ndarray, dtype: str, max_val: int):
        if len(array.shape) == 1:
            f.write(f"const {dtype} {name}[{len(array)}] = {{")
        else:
            f.write(f"const {dtype} {name}[{len(array)}][{array.shape[1]}] = {{\n")
        for i in range(len(array)):
            if len(array.shape) == 1:
                w = int(np.clip(array[i], -max_val, max_val))
                f.write(f"{w}, " if i < len(array) - 1 else f"{w}")
            else:
                f.write("    {")
                for j in range(array.shape[1]):
                    w = int(np.clip(array[i, j], -max_val, max_val))
                    f.write(f"{w}, " if j < array.shape[1] - 1 else f"{w}")
                f.write("},\n")
        f.write("};\n\n")

    with open("src/nnue/params.c", "w") as f:
        f.write("#include <stdint.h>\n\n")
        f.write(f"const int QA = {QA}, QB = {QB};\n\n")
        write_array(f, "fc1_weights", fc1_w, "int16_t", 32767)
        write_array(f, "fc2_weights", fc2_w, "int8_t", 127)
        write_array(f, "fc3_weights", fc3_w, "int8_t", 127)
        write_array(f, "fc1_biases", fc1_b, "int16_t", 32767)
        write_array(f, "fc2_biases", fc2_b, "int8_t", 127)
        write_array(f, "fc3_biases", fc3_b, "int8_t", 127)
