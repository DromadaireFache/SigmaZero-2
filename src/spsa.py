import os
import optimize_constants
import argparse
import subprocess
from random import random
import shutil
import copy


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


def perturbation_magnitude(consts: Consts, ck: float) -> Consts:
    """Per-parameter perturbation magnitudes: max(1, round(ck * |theta_i|))."""
    mag: Consts = {}
    for key, value in consts.items():
        if isinstance(value, list):
            mag[key] = [max(1, int(round(ck * max(1.0, abs(v))))) for v in value]
        elif isinstance(value, (int, float)):
            mag[key] = max(1, int(round(ck * max(1.0, abs(value)))))
        else:
            raise ValueError(f"Unsupported type for constant '{key}': {type(value)}")
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


def apply_perturbation(consts: Consts, delta: Consts, mag: Consts, sign: int) -> Consts:
    """Apply theta +/- mag * delta."""
    out: Consts = {}
    for key, value in consts.items():
        if isinstance(value, list):
            out[key] = [
                v + sign * mag[key][i] * delta[key][i]  # type: ignore[index]
                for i, v in enumerate(value)
            ]
        else:
            out[key] = value + sign * mag[key] * delta[key]  # type: ignore[index]
    return out


def build_engine(consts: Consts, exe_name: str) -> None:
    with open("src/consts.c", "w") as f:
        f.write(optimize_constants.make_const_file(consts))
    subprocess.run(["make"], check=True)
    shutil.move(optimize_constants.executable("sigma-zero"), exe_name)


def evaluate_pair(plus_exe: str, minus_exe: str, n_games: int) -> tuple[float, float]:
    """
    Compare theta+ and theta- directly using the same opening set.
    Returns (score_diff, elo) from theta+ perspective.
    score_diff = (wins - losses) / n_games in [-1, 1].
    """
    results = optimize_constants.tournament_result(plus_exe, minus_exe, required_score=-10**9, n_games=n_games)
    score_diff = (results["wins"] - results["losses"]) / max(1, n_games)
    return score_diff, float(results["elo"])


def evaluate_vs_baseline(candidate_exe: str, baseline_exe: str, n_games: int) -> float:
    """Validation Elo for checkpoint selection."""
    return optimize_constants.tournament_elo(candidate_exe, baseline_exe, n_games=n_games)


def generate_gradient(delta: Consts, mag: Consts, score_diff: float) -> Consts:
    """
    SPSA gradient estimate using y(theta+) - y(theta-) and per-parameter perturbation size.
    g_i = score_diff / (2 * mag_i * delta_i)
    Since delta_i in {-1,+1}, this is numerically stable and scale-aware.
    """
    gradient: Consts = {}
    for key, d in delta.items():
        if isinstance(d, list):
            gradient[key] = [score_diff / (2.0 * mag[key][i] * d_i) for i, d_i in enumerate(d)]  # type: ignore[index]
        else:
            gradient[key] = score_diff / (2.0 * mag[key] * d)  # type: ignore[index]
    return gradient


def clamp_consts(consts: Consts) -> Consts:
    """Match consts.c behavior for scalar defines: clamp to >= 1."""
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
    best_elo = float("-inf")

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
            score_diff, elo_pm = evaluate_pair(plus_exe, minus_exe, n_games=n_games)

            gradient = generate_gradient(delta, mag, score_diff)
            consts = clamp_consts(add_offset_to_consts(consts, gradient, ak))

            print(
                f"Iter {k:03d}/{number_of_iterations}: ak={ak:.4f}, ck={ck:.4f}, "
                f"plus-vs-minus score={score_diff:+.3f}, elo={elo_pm:+.1f}"
            )

            if checkpoint_every > 0 and (k % checkpoint_every == 0 or k == number_of_iterations):
                build_engine(consts, current_exe)
                elo = evaluate_vs_baseline(current_exe, baseline_exe, n_games=checkpoint_games)
                print(f"  Checkpoint vs baseline: {elo:+.1f} Elo ({checkpoint_games} games)")
                if elo > best_elo:
                    best_elo = elo
                    best_consts = copy.deepcopy(consts)
                    print(f"  New best checkpoint constants: {best_elo:+.1f} Elo")
    
    except KeyboardInterrupt:
        print("SPSA optimization interrupted by user. Writing current best constants to src/consts.c.")
         
    finally:
        if best_elo != float("-inf"):
            consts = best_consts
        with open("src/consts.c", "w") as f:
            f.write(optimize_constants.make_const_file(consts))
        print("Final best constants written to src/consts.c")
        for exe in [baseline_exe, plus_exe, minus_exe, current_exe]:
            if os.path.exists(exe):
                os.remove(exe)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPSA optimization for parameter fine tuning.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for SPSA.")
    parser.add_argument(
        "--A", type=float, default=20, help="SPSA constant A, which controls the stability of the algorithm."
    )
    parser.add_argument("--a", type=float, default=2.0, help="SPSA constant a, which controls the initial step size.")
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

    spsa(
        args.iterations,
        args.A,
        args.a,
        args.c,
        args.games,
        args.checkpoint_games,
        args.checkpoint_every,
    )
