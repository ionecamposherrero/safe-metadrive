"""Fase 1 — Entrenamiento en escenario CON tráfico y obstáculos.

Algoritmos: PPO, SAC (coste restado en la recompensa) y sus variantes
Lagrangianas PPO-Lag y SAC-Lag (coste como restricción explícita).

Entorno: SafeMetaDrive, 1000 escenarios, traffic_density=0.15, accident_prob=0.4.
Se guarda un checkpoint cada 1M de pasos (usado en la Fase 2 para recolectar
trayectorias con políticas de distinta calidad).

Uso:
    python scripts/phase1/train.py --algo sac_lag
"""

import argparse
import json
import random
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from src.agents import (ALGO_NORM_REWARD, ALGOS, HORIZON, LOG_INTERVAL,
                        N_ENVS, make_model)
from src.callbacks import MetricsCallback
from src.envs import make_phase1_env

NUM_ESCENARIOS_TRAIN = 1000
START_SEED_TRAIN = 0
TRAFFIC_DENSITY_TRAIN = 0.15
ACCIDENT_PROB_TRAIN = 0.4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", choices=ALGOS, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=N_ENVS)
    p.add_argument("--device", default=None, help="cpu | cuda | auto (por defecto, según algoritmo)")
    p.add_argument("--output-dir", default=str(ROOT / "results" / "phase1"))
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir) / args.algo
    out_dir.mkdir(parents=True, exist_ok=True)

    # PPO/SAC estándar reciben el coste restado en la recompensa;
    # los Lagrangianos lo reciben como señal separada (info["cost"]).
    cost_in_reward = args.algo in ("ppo", "sac")

    all_histories = {}

    for seed in args.seeds:
        print(f"\n{'=' * 70}")
        print(f"  Fase 1 — {args.algo.upper()} — semilla {seed}")
        print(f"{'=' * 70}\n")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        set_random_seed(seed)

        train_env = SubprocVecEnv([
            partial(
                make_phase1_env,
                start_seed=START_SEED_TRAIN,
                num_scenarios=NUM_ESCENARIOS_TRAIN,
                horizon=HORIZON,
                traffic_density=TRAFFIC_DENSITY_TRAIN,
                accident_prob=ACCIDENT_PROB_TRAIN,
                cost_in_reward=cost_in_reward,
            )
            for _ in range(args.n_envs)
        ])

        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=ALGO_NORM_REWARD[args.algo],
            clip_obs=10.0,
        )

        model = make_model(args.algo, train_env, seed, device=args.device)
        model.set_logger(configure(str(out_dir / f"{args.algo}_seed{seed}.log"), ["log"]))

        metrics_callback = MetricsCallback(log_interval=LOG_INTERVAL, verbose=1)
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1_000_000 // args.n_envs, 1),  # save_freq es por env
            save_path=str(out_dir / f"checkpoints_seed{seed}"),
            name_prefix=f"{args.algo}_seed{seed}",
            save_vecnormalize=True,
        )

        model.learn(total_timesteps=args.timesteps,
                    callback=CallbackList([metrics_callback, checkpoint_callback]))

        model.save(str(out_dir / f"{args.algo}_seed{seed}"))
        train_env.save(str(out_dir / f"{args.algo}_vecnorm_seed{seed}.pkl"))
        all_histories[seed] = {k: [float(x) for x in v]
                               for k, v in metrics_callback.history.items()}

        train_env.close()
        print(f"\nEntrenamiento de la semilla {seed} finalizado.\n")

    with open(out_dir / f"{args.algo}_histories.json", "w") as f:
        json.dump({str(k): v for k, v in all_histories.items()}, f, indent=2)
    print(f"Historiales guardados en {out_dir / f'{args.algo}_histories.json'}")


if __name__ == "__main__":
    main()
