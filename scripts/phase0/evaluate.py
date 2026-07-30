"""Fase 0 — Evaluación en 100 escenarios de test sin tráfico (no vistos en train).

Uso:
    python scripts/phase0/evaluate.py --algo ppo --config optimized
"""

import argparse
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents import ALGO_CLASS, HORIZON
from src.envs import make_phase0_default_env, make_phase0_optimized_env
from src.evaluation import evaluate_policy_seeds

START_SEED_TEST = 500
NUM_ESCENARIOS_TEST = 100


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", choices=["ppo", "sac"], required=True)
    p.add_argument("--config", choices=["default", "optimized"], default="optimized")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--results-dir", default=str(ROOT / "results" / "phase0"))
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.results_dir) / args.config / args.algo

    env_factory = (make_phase0_default_env if args.config == "default"
                   else make_phase0_optimized_env)
    test_env_factory = partial(
        env_factory,
        start_seed=START_SEED_TEST,
        num_scenarios=NUM_ESCENARIOS_TEST,
        horizon=HORIZON,
    )

    evaluate_policy_seeds(
        model_class=ALGO_CLASS[args.algo],
        model_prefix=str(model_dir / f"{args.algo}_seed"),
        vecnorm_prefix=str(model_dir / f"{args.algo}_vecnorm_seed"),
        env_factory=test_env_factory,
        seeds=args.seeds,
        n_episodes=args.episodes,
        device=args.device,
        label=f"{args.algo.upper()}-{args.config}",
        output_json=str(model_dir / f"{args.algo}_test_results.json"),
    )


if __name__ == "__main__":
    main()
