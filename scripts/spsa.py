from datetime import timedelta
import os
from pprint import pprint
import scripts.optimize_constants as optimize_constants
import argparse
import subprocess
from random import random
import math
import shutil
import copy
import matplotlib.pyplot as plt


type Consts = dict[str, float | list[float]]


def rand_sign() -> int:
    return 1 if random() < 0.5 else -1


def generate_delta(consts: Consts) -> Consts:
    """Generate SPSA Bernoulli perturbation vector (Delta_i in {-1, +1})."""
    delta: Consts = {}
    for key, value in consts.items():
        if isinstance(value, list):
            delta[key] = [rand_sign() for _ in value]
        elif isinstance(value, (int, float)):
            delta[key] = rand_sign()
        else:
            raise ValueError(f"Unsupported type for constant '{key}': {type(value)}")
    return delta


def perturbation_magnitude(consts: Consts, ck: float, scalar_floor: int = 2) -> Consts:
    mag: Consts = {}
    for key, value in consts.items():
        if isinstance(value, list):
            # PSQT: floor of 1 is fine, values are typically large
            mag[key] = [max(1, int(round(ck * max(1.0, abs(v))))) for v in value]
        elif isinstance(value, (int, float)):
            # Scalars: higher floor so gradient denominator isn't always 2
            mag[key] = max(scalar_floor, int(round(ck * max(1.0, abs(value)))))
    return mag


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


def zeros_like_consts(consts: Consts) -> Consts:
    """Create zero-initialized residual accumulators matching const layout."""
    out: Consts = {}
    for key, value in consts.items():
        if isinstance(value, list):
            out[key] = [0.0 for _ in value]
        else:
            out[key] = 0.0
    return out


def apply_integer_accumulated_step(
    consts: Consts, gradient: Consts, ak: float, residuals: Consts
) -> tuple[Consts, Consts, int]:
    """
    Apply SPSA step while respecting integer constants in consts.h generation.

    Fractional updates are accumulated in residuals and converted to integer steps
    once |residual + ak * gradient| reaches >= 1.
    Returns (updated_consts, updated_residuals, nonzero_step_count).
    """
    updated_consts: Consts = {}
    updated_residuals: Consts = {}
    nonzero_steps = 0

    for key, value in consts.items():
        grad = gradient[key]
        res = residuals[key]

        if isinstance(value, list):
            new_values: list[float] = []
            new_residuals: list[float] = []
            for v, g, r in zip(value, grad, res):  # type: ignore[arg-type]
                raw_step = r + ak * g
                int_step = math.floor(raw_step) if raw_step >= 0 else math.ceil(raw_step)
                if int_step != 0:
                    nonzero_steps += 1
                new_values.append(v + int_step)
                new_residuals.append(raw_step - int_step)
            updated_consts[key] = new_values
            updated_residuals[key] = new_residuals
        else:
            raw_step = res + ak * grad
            int_step = math.floor(raw_step) if raw_step >= 0 else math.ceil(raw_step)
            if int_step != 0:
                nonzero_steps += 1
            new_value = max(1.0, value + int_step)
            # Compute residual based on the step actually applied, not int_step
            actual_step = new_value - value
            updated_consts[key] = new_value
            updated_residuals[key] = raw_step - actual_step

    return updated_consts, updated_residuals, nonzero_steps


def apply_perturbation(consts: Consts, delta: Consts, mag: Consts, sign: int) -> Consts:
    """Apply theta +/- mag * delta."""
    out: Consts = {}
    for key, value in consts.items():
        if isinstance(value, list):
            out[key] = [v + sign * mag[key][i] * delta[key][i] for i, v in enumerate(value)]  # type: ignore[index]
        else:
            out[key] = value + sign * mag[key] * delta[key]  # type: ignore[index]
    return out


def build_engine(consts: Consts, exe_name: str) -> None:
    with open("src/consts.h", "w") as f:
        f.write(optimize_constants.make_const_file(consts))
    subprocess.run(["make"], check=True)
    shutil.move(optimize_constants.executable("engine"), exe_name)


def evaluate_pair(plus_exe: str, minus_exe: str, n_games: int) -> tuple[float, float, float]:
    """
    Compare theta+ and theta- directly using the same opening set.
    Returns (score_diff_raw, score_diff_norm, elo) from theta+ perspective.
    score_diff_raw = wins - losses in [-n_games, n_games]
    score_diff_norm = (wins - losses) / n_games in [-1, 1]
    """
    results = optimize_constants.tournament_result(plus_exe, minus_exe, required_score=-(10**9), n_games=n_games)
    score_diff_raw = float(results["wins"] - results["losses"])
    score_diff_norm = score_diff_raw / max(1, n_games)
    return score_diff_raw, score_diff_norm, float(results["elo"])


def evaluate_vs_baseline(candidate_exe: str, baseline_exe: str, n_games: int) -> float:
    """Validation Elo for checkpoint selection."""
    return optimize_constants.tournament_elo(candidate_exe, baseline_exe, n_games=n_games)


def generate_gradient(delta: Consts, mag: Consts, score_diff_raw: float) -> Consts:
    """
    SPSA gradient estimate using y(theta+) - y(theta-) and per-parameter perturbation size.
    g_i = score_diff_raw / (2 * mag_i * delta_i)
    Since delta_i in {-1,+1}, this is numerically stable and scale-aware.
    """
    gradient: Consts = {}
    for key, d in delta.items():
        if isinstance(d, list):
            gradient[key] = [score_diff_raw / (2.0 * mag[key][i] * d_i) for i, d_i in enumerate(d)]  # type: ignore[index]
        else:
            gradient[key] = score_diff_raw / (2.0 * mag[key] * d)  # type: ignore[index]
    pprint({k: round(v, 4) for k, v in gradient.items() if isinstance(v, (int, float))})
    return gradient


def clamp_consts(consts: Consts) -> Consts:
    """Match consts.h behavior for scalar defines: clamp to >= 1."""
    out: Consts = {}
    for key, value in consts.items():
        if isinstance(value, list):
            out[key] = value
        else:
            out[key] = max(1.0, value)
    return out


def spsa(
    number_of_iterations: int,
    A: float,
    a: float,
    c: float,
    n_games: int,
    checkpoint_games: int,
    checkpoint_every: int,
):
    print(
        f"Starting SPSA optimization with {number_of_iterations} iterations, "
        f"A={A}, a={a}, c={c}, games/iter={n_games}"
    )
    consts: Consts = copy.deepcopy(optimize_constants.best_consts)
    best_consts: Consts = copy.deepcopy(consts)
    residuals: Consts = zeros_like_consts(consts)
    checkpoint_ran = False
    best_elo = 0
    checkpoint_elos: list[float] = []
    print(f"Trainable params: {len(consts)} keys")

    baseline_exe = optimize_constants.executable("baseline_engine")
    plus_exe = optimize_constants.executable("perturbed_plus_engine")
    minus_exe = optimize_constants.executable("perturbed_minus_engine")
    current_exe = optimize_constants.executable("current_engine")

    build_engine(consts, baseline_exe)

    try:
        for k in range(1, number_of_iterations + 1):
            ak = a / (k + A) ** 0.602
            ck = c / k**0.101

            # Generate random perturbation vector Delta and magnitude.
            delta = generate_delta(consts)
            mag = perturbation_magnitude(consts, ck)

            # Construct perturbed parameter vectors.
            theta_plus = apply_perturbation(consts, delta, mag, +1)
            theta_minus = apply_perturbation(consts, delta, mag, -1)

            build_engine(theta_plus, plus_exe)
            build_engine(theta_minus, minus_exe)

            # Evaluate theta+ against theta- directly to reduce noise and cost.
            score_diff_raw, score_diff_norm, elo_pm = evaluate_pair(plus_exe, minus_exe, n_games=n_games)

            gradient = generate_gradient(delta, mag, score_diff_raw)
            consts, residuals, nonzero_steps = apply_integer_accumulated_step(consts, gradient, ak, residuals)
            consts = clamp_consts(consts)

            print(
                f"Iter {k:03d}/{number_of_iterations}: ak={ak:.4f}, ck={ck:.4f}, "
                f"plus-vs-minus score={score_diff_norm:+.3f}, elo={elo_pm:+.1f}, int-steps={nonzero_steps}"
            )

            if checkpoint_every > 0 and (k % checkpoint_every == 0 or k == number_of_iterations):
                build_engine(consts, current_exe)
                elo = evaluate_vs_baseline(current_exe, baseline_exe, n_games=checkpoint_games)
                print(f"  Checkpoint vs baseline: {elo:+.1f} Elo ({checkpoint_games} games)")
                checkpoint_ran = True
                if elo > best_elo:
                    best_elo = elo
                    best_consts = copy.deepcopy(consts)
                    print(f"  New best checkpoint constants: {best_elo:+.1f} Elo")
                checkpoint_elos.append(elo)
    except KeyboardInterrupt:
        print("SPSA optimization interrupted by user. Writing current best constants to src/consts.h.")

    finally:
        consts = best_consts if checkpoint_ran else consts
        with open("src/consts.h", "w") as f:
            f.write(optimize_constants.make_const_file(consts))
        print("Final best constants written to src/consts.h")
        for exe in [baseline_exe, plus_exe, minus_exe, current_exe]:
            if os.path.exists(exe):
                os.remove(exe)

        # Make a graph of checkpoint Elo over time if we have multiple checkpoints
        if len(checkpoint_elos) > 1:
            plt.plot(
                range(checkpoint_every, checkpoint_every * len(checkpoint_elos) + 1, checkpoint_every),
                checkpoint_elos,
                marker="o",
            )
            plt.title(f"SPSA Checkpoint Elo Over Iterations ({n_games} games/iter)")
            plt.xlabel("Iteration")
            plt.ylabel("Elo vs Baseline")
            plt.grid()
            plt.savefig("spsa_checkpoint_elo.png")
            print("Checkpoint Elo graph saved as spsa_checkpoint_elo.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPSA optimization for parameter fine tuning.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for SPSA.")
    parser.add_argument(
        "--A", type=float, default=20, help="SPSA constant A, which controls the stability of the algorithm."
    )
    parser.add_argument("--a", type=float, default=4, help="SPSA constant a, which controls the initial step size.")
    parser.add_argument(
        "--c", type=float, default=0.2, help="SPSA constant c, which controls the magnitude of the perturbation."
    )
    parser.add_argument("--games", type=int, default=50, help="Games per SPSA iteration (theta+ vs theta-).")
    parser.add_argument(
        "--checkpoint-games",
        type=int,
        default=30,
        help="Games for validation against baseline at each checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Run checkpoint validation every N iterations.",
    )
    args = parser.parse_args()

    time_estimate = timedelta(
        seconds=(
            args.iterations * args.games
            + (args.iterations // args.checkpoint_every * args.checkpoint_games if args.checkpoint_every > 0 else 0)
        )
        * 12  # Rough estimate: 12 seconds per game pair on CPU
    )
    print(f"Estimated total runtime: {time_estimate}")
    spsa(
        args.iterations,
        args.A,
        args.a,
        args.c,
        args.games,
        args.checkpoint_games,
        args.checkpoint_every,
    )
