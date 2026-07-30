"""Fase 2 (paso 5) — Evaluación de las políticas entrenadas con coste de riesgo.

Igual que la evaluación de la Fase 1 (300 escenarios, 3 niveles de congestión),
pero el coste reportado es ``cost_hard`` (solo colisiones reales), de modo que
la columna "Cost" cuenta el número de colisiones por episodio.

Uso:
    python scripts/phase2/evaluate.py --algo sac_lag --risk-gamma 0.9 --level all
"""

import argparse
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents import ALGO_CLASS, ALGOS, HORIZON
from src.envs import TEST_LEVELS, make_phase2_env
from src.evaluation import evaluate_policy_seeds

START_SEED_TEST = 1000
NUM_ESCENARIOS_TEST = 300


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", choices=ALGOS, required=True)
    p.add_argument("--risk-gamma", type=float, choices=[0.8, 0.9, 0.95], default=0.9)
    p.add_argument("--risk-models-dir",
                   default=str(ROOT / "results" / "phase2" / "risk_models"))
    p.add_argument("--risk-device", default="cpu")
    p.add_argument("--level", choices=list(TEST_LEVELS) + ["all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--device", default="auto")
    p.add_argument("--results-dir", default=str(ROOT / "results" / "phase2" / "rl"))
    return p.parse_args()


def main():
    args = parse_args()
    gamma_tag = f"gamma_{args.risk_gamma}"
    risk_dir = Path(args.risk_models_dir) / gamma_tag
    model_dir = Path(args.results_dir) / gamma_tag / args.algo
    levels = list(TEST_LEVELS) if args.level == "all" else [args.level]

    for level in levels:
        cfg = TEST_LEVELS[level]
        print(f"\n### Nivel de congestión: {level} "
              f"(traffic_density={cfg['traffic_density']}, accident_prob={cfg['accident_prob']})")

        test_env_factory = partial(
            make_phase2_env,
            risk_model_path=str(risk_dir / "best_model.pt"),
            risk_scaler_path=str(risk_dir / "scaler_tabular.pkl"),
            risk_device=args.risk_device,
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
            cost_key="cost_hard",  # contar solo colisiones reales
            label=args.algo.upper(),
            output_json=str(model_dir / f"{args.algo}_test_results_{level}.json"),
        )


if __name__ == "__main__":
    main()
