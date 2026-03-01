import os
from pprint import pprint
import optimize_constants
import argparse
import subprocess
from random import random
import shutil


type Consts = dict[str, float | list[float]]


def multiply_by_rand(value: float) -> float:
    r = (1 if random() < 0.5 else -1)
    return value * r if value != 0 else r


def generate_delta(consts: Consts) -> Consts:
    delta = {}
    for key, value in consts.items():
        if isinstance(value, list):
            delta[key] = [multiply_by_rand(v) for v in value]
        elif isinstance(value, (int, float)):
            delta[key] = multiply_by_rand(value)
        else:
            raise ValueError(f"Unsupported type for constant '{key}': {type(value)}")
    return delta


def add_offset_to_consts(consts: Consts, offset: Consts, scaling_factor: float) -> Consts:
    new_consts = {}
    for key, value in consts.items():
        if isinstance(value, list):
            new_consts[key] = [v + scaling_factor * d for v, d in zip(value, offset[key])]
        elif isinstance(value, (int, float)):
            new_consts[key] = value + scaling_factor * offset[key]
        else:
            raise ValueError(f"Unsupported type for constant '{key}': {type(value)}")
    return new_consts


def evaluate(perturbed: Consts, baseline: Consts) -> float:
    """Returns the elo difference between the baseline and perturbed constants."""

    def make(consts: Consts, exe_name: str):
        with open("src/consts.c", "w") as f:
             f.write(optimize_constants.make_const_file(consts))
        subprocess.run(["make"], check=True)
        shutil.move(optimize_constants.executable("sigma-zero"), exe_name)

    baseline_exe = optimize_constants.executable("baseline_engine")
    perturbed_exe = optimize_constants.executable("perturbed_engine")
    make(baseline, baseline_exe)
    make(perturbed, perturbed_exe)

    elo = optimize_constants.tournament_elo(perturbed_exe, baseline_exe, n_games=20)
    return elo


def generate_gradient(consts: Consts, delta: Consts, y_plus: float, y_minus: float, ck: float) -> Consts:
    if ck == 0:
        raise ValueError("Perturbation magnitude ck must be non-zero to compute gradient.")
    gradient = {}
    for key in consts.keys():
        if isinstance(consts[key], list):
            gradient[key] = [(y_plus - y_minus) / (2 * ck * d) for d in delta[key]]
        elif isinstance(consts[key], (int, float)):
            gradient[key] = (y_plus - y_minus) / (2 * ck * delta[key])
        else:
            raise ValueError(f"Unsupported type for constant '{key}': {type(consts[key])}")
    return gradient


def spsa(number_of_iterations: int, A: float, a: float, c: float):
    print(f"Starting SPSA optimization with {number_of_iterations} iterations, A={A}, a={a}, c={c}")
    consts: Consts = optimize_constants.best_consts.copy()

    try:
        for k in range(1, number_of_iterations + 1):
            ak = a / (k + A) ** 0.602
            ck = c / k**0.101

            # Generate random perturbation vector Δ
            delta = generate_delta(consts)

            # Construct perturbed parameter vectors
            theta_plus = add_offset_to_consts(consts, delta, ck)
            theta_minus = add_offset_to_consts(consts, delta, -ck)

            # Evaluate via self-play
            y_plus = evaluate(theta_plus, consts)
            y_minus = evaluate(theta_minus, consts)

            # Gradient estimate and parameter update
            gradient = generate_gradient(consts, delta, y_plus, y_minus, ck)
            consts = add_offset_to_consts(consts, gradient, ak)
    
    except KeyboardInterrupt:
        print("SPSA optimization interrupted by user. Writing current best constants to src/consts.c.")
         
    finally:
        with open("src/consts.c", "w") as f:
            f.write(optimize_constants.make_const_file(consts))
        print("Final best constants written to src/consts.c")
        os.remove(optimize_constants.executable("baseline_engine"))
        os.remove(optimize_constants.executable("perturbed_engine"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPSA optimization for parameter fine tuning.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for SPSA.")
    parser.add_argument(
        "--A", type=float, default=10, help="SPSA constant A, which controls the stability of the algorithm."
    )
    parser.add_argument("--a", type=float, default=0.1, help="SPSA constant a, which controls the initial step size.")
    parser.add_argument(
        "--c", type=float, default=0.1, help="SPSA constant c, which controls the magnitude of the perturbation."
    )
    args = parser.parse_args()

    spsa(args.iterations, args.A, args.a, args.c)
