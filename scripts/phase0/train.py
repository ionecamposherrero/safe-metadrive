"""Fase 0 — Entrenamiento en escenario SIN tráfico.

Compara la configuración por defecto (recompensa y hiperparámetros de
MetaDrive/SB3) con la configuración optimizada propuesta.

Uso:
    python scripts/phase0/train.py --algo ppo --config optimized
    python scripts/phase0/train.py --algo sac --config default --seeds 0 1 2
"""

import argparse
import json
import os
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

from src.agents import (ALGO_NORM_REWARD, HORIZON, LOG_INTERVAL, N_ENVS,
                        make_default_model, make_model)
from src.callbacks import MetricsCallback
from src.envs import make_phase0_default_env, make_phase0_optimized_env

NUM_ESCENARIOS_TRAIN = 500
START_SEED_TRAIN = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", choices=["ppo", "sac"], required=True)
    p.add_argument("--config", choices=["default", "optimized"], default="optimized")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=N_ENVS)
    p.add_argument("--output-dir", default=str(ROOT / "results" / "phase0"))
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir) / args.config / args.algo
    out_dir.mkdir(parents=True, exist_ok=True)

    env_factory = (make_phase0_default_env if args.config == "default"
                   else make_phase0_optimized_env)

    all_histories = {}

    for seed in args.seeds:
        print(f"\n{'=' * 70}")
        print(f"  Fase 0 — {args.algo.upper()} ({args.config}) — semilla {seed}")
        print(f"{'=' * 70}\n")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        set_random_seed(seed)

        train_env = SubprocVecEnv([
            partial(
                env_factory,
                start_seed=START_SEED_TRAIN,
                num_scenarios=NUM_ESCENARIOS_TRAIN,
                horizon=HORIZON,
            )
            for _ in range(args.n_envs)
        ])

        callbacks = [MetricsCallback(log_interval=LOG_INTERVAL, verbose=1)]

        if args.config == "default":
            model = make_default_model(args.algo, train_env, seed)
        else:
            # Normalización de observaciones (y recompensa solo en PPO)
            train_env = VecNormalize(
                train_env,
                norm_obs=True,
                norm_reward=ALGO_NORM_REWARD[args.algo],
                clip_obs=10.0,
            )
            model = make_model(args.algo, train_env, seed)
            callbacks.append(CheckpointCallback(
                save_freq=max(1_000_000 // args.n_envs, 1),  # save_freq es por env
                save_path=str(out_dir / f"checkpoints_seed{seed}"),
                name_prefix=f"{args.algo}_seed{seed}",
                save_vecnormalize=True,
            ))

        log_filename = str(out_dir / f"{args.algo}_seed{seed}.log")
        model.set_logger(configure(log_filename, ["log"]))

        model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks))

        model.save(str(out_dir / f"{args.algo}_seed{seed}"))
        if isinstance(train_env, VecNormalize):
            train_env.save(str(out_dir / f"{args.algo}_vecnorm_seed{seed}.pkl"))
        all_histories[seed] = {k: [float(x) for x in v]
                               for k, v in callbacks[0].history.items()}

        train_env.close()
        print(f"\nEntrenamiento de la semilla {seed} finalizado.\n")

    with open(out_dir / f"{args.algo}_histories.json", "w") as f:
        json.dump({str(k): v for k, v in all_histories.items()}, f, indent=2)
    print(f"Historiales guardados en {out_dir / f'{args.algo}_histories.json'}")


if __name__ == "__main__":
    main()
