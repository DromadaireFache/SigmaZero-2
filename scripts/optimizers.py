import time


def log(message: str):
    with open("optimize_constants.log", "a") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    print(message)

class Consts:
    def __init__(self, filename="src/consts.h"):
        self.consts = {}
        self.load_consts(filename)

    def load_consts(self, filename):
        with open(filename, "r") as f:
            current_consts = f.read()

            for line in current_consts.splitlines():
                if line.startswith("#define "):
                    parts = line.split()
                    assert len(parts) == 3
                    const_name = parts[1]
                    const_value = int(parts[2])
                    self.consts[const_name] = const_value
                elif line.startswith("const int "):
                    assert "[]" in line and "{" in line and "};" in line
                    const_name = line.split()[2].removesuffix("[]")
                    array_values = line[line.find("{") + 1 : line.find("}")].split(",")
                    self.consts[const_name] = [int(v.strip()) for v in array_values]
                else:
                    continue

    def __getitem__(self, key: str):
        return self.consts[key]

    def get_scalar(self, key: str):
        value = self.consts[key]
        if isinstance(value, list):
            raise ValueError(f"Expected a scalar value for {key}, but got a list.")
        return value

    def get_vector(self, key: str):
        value = self.consts[key]
        if not isinstance(value, list):
            raise ValueError(f"Expected a vector value for {key}, but got a scalar.")
        return value

    def iter_scalars(self):
        for key, value in self.consts.items():
            if not isinstance(value, list):
                yield key, value

    def iter_vectors(self):
        for key, value in self.consts.items():
            if isinstance(value, list):
                yield key, value

    def to_string(self, show_vectors: bool = False) -> str:
        lines = ["{"]
        produce = lambda lines: "\n".join(lines) + "\n}"
        
        for key, value in self.iter_scalars():
            lines.append(f'    "{key}": {value},')
        if not show_vectors:
            return produce(lines)

        for key, value in self.iter_vectors():
            if len(value) != 64:
                lines.append(f'    "{key}": {value},')
                continue
            
            # Make an 8x8 grid of the vector values
            element_width = max(len(str(v)) for v in value)
            lines.append(f'    "{key}": [')
            for i in reversed(range(8)):
                line = " ".join(f"{v:>{element_width}}" for v in value[i * 8 : (i + 1) * 8])
                lines.append(f"        {line}")
            lines.append("    ],")

        return produce(lines)


best_consts = Consts()


if __name__ == "__main__":
    print(best_consts.to_string(show_vectors=True))
