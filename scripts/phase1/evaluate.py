"""Fase 1 — Evaluación en 300 escenarios de test con tres niveles de congestión.

Niveles (traffic_density, accident_prob): baja (0.05, 0.1),
media (0.1, 0.25) y alta (0.15, 0.4).

Uso:
    python scripts/phase1/evaluate.py --algo sac_lag --level alta
    python scripts/phase1/evaluate.py --algo ppo --level all
"""

import argparse
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents import ALGO_CLASS, ALGOS, HORIZON
from src.envs import TEST_LEVELS, make_phase1_env
from src.evaluation import evaluate_policy_seeds

START_SEED_TEST = 1000
NUM_ESCENARIOS_TEST = 300


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", choices=ALGOS, required=True)
    p.add_argument("--level", choices=list(TEST_LEVELS) + ["all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--device", default="auto")
    p.add_argument("--results-dir", default=str(ROOT / "results" / "phase1"))
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.results_dir) / args.algo
    levels = list(TEST_LEVELS) if args.level == "all" else [args.level]

    for level in levels:
        cfg = TEST_LEVELS[level]
        print(f"\n### Nivel de congestión: {level} "
              f"(traffic_density={cfg['traffic_density']}, accident_prob={cfg['accident_prob']})")

        test_env_factory = partial(
            make_phase1_env,
            start_seed=START_SEED_TEST,
            num_scenarios=NUM_ESCENARIOS_TEST,
            horizon=HORIZON,
            cost_in_reward=args.algo in ("ppo", "sac"),
            **cfg,
        )

        evaluate_policy_seeds(
            model_class=ALGO_CLASS[args.algo],
            model_prefix=str(model_dir / f"{args.algo}_seed"),
            vecnorm_prefix=str(model_dir / f"{args.algo}_vecnorm_seed"),
            env_factory=test_env_factory,
            seeds=args.seeds,
            n_episodes=args.episodes,
            device=args.device,
            label=args.algo.upper(),
            output_json=str(model_dir / f"{args.algo}_test_results_{level}.json"),
        )


if __name__ == "__main__":
    main()
