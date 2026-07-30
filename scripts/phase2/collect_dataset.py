"""Fase 2 (paso 1) — Recolección de trayectorias con las políticas de la Fase 1.

Para una política dada (PPO, SAC, PPO-Lag o SAC-Lag) carga los 9 modelos
(3 semillas × 3 checkpoints de 1M/2M/3M pasos) y recolecta 30 episodios por
modelo y nivel de congestión (3 niveles) → 810 episodios por política.
Cada episodio usa una semilla de escenario única (offset por política) y
termina en la primera colisión, guardando las observaciones LidarState crudas
y el indicador de colisión por frame.

Uso:
    python scripts/phase2/collect_dataset.py --policy ppo
    python scripts/phase2/collect_dataset.py --policy sac_lag --models-dir results/phase1/sac_lag
"""

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.agents import ALGO_CLASS, ALGOS, HORIZON
from src.envs import make_dataset_env

NUM_EPISODES = 30          # episodios por combinación (modelo × nivel)
SEEDS = [0, 1, 2]
STEPS = [1_000_000, 2_000_000, 3_000_000]

# 9 modelos × 3 niveles × 30 ep = 810 seeds por política;
# 4 políticas × 810 = 3240 seeds únicos → NUM_SCENARIOS del entorno
NUM_SCENARIOS = 3240
POLICY_OFFSET = {"ppo": 0, "sac": 810, "ppo_lag": 1620, "sac_lag": 2430}

LEVELS = [
    {"id": "nivel1", "traffic_density": 0.05, "accident_prob": 0.10},
    {"id": "nivel2", "traffic_density": 0.10, "accident_prob": 0.25},
    {"id": "nivel3", "traffic_density": 0.15, "accident_prob": 0.40},
]

# Nombre con el que se identifica cada política dentro del dataset
POLICY_NAME = {"ppo": "PPO", "sac": "SAC", "ppo_lag": "PPOLag", "sac_lag": "SACLag"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=ALGOS, required=True)
    p.add_argument("--models-dir", default=None,
                   help="Carpeta con los checkpoints de la Fase 1 "
                        "(por defecto results/phase1/<policy>)")
    p.add_argument("--output-dir", default=str(ROOT / "results" / "phase2" / "dataset"))
    p.add_argument("--episodes", type=int, default=NUM_EPISODES)
    return p.parse_args()


def load_model_and_vecnorm(model_class, model_path, vecnorm_path):
    model = model_class.load(model_path)
    vec_norm = None
    if os.path.exists(vecnorm_path):
        dummy = DummyVecEnv([lambda: make_dataset_env(0.05, 0.1)])
        vec_norm = VecNormalize.load(vecnorm_path, dummy)
        vec_norm.training = False
        vec_norm.norm_reward = False
        dummy.close()
    else:
        print(f"  [AVISO] VecNormalize no encontrado: {vecnorm_path}")
    return model, vec_norm


def normalize_obs(obs, vec_norm):
    return vec_norm.normalize_obs(obs) if vec_norm is not None else obs


def collect_episode(env, model, vec_norm, ep_idx, n_episodes, episode_seed):
    observations, collisions = [], []
    obs, _ = env.reset(seed=episode_seed)
    step = 0
    while True:
        action, _ = model.predict(normalize_obs(obs, vec_norm), deterministic=True)
        obs_next, _, terminated, truncated, info = env.step(action)
        observations.append(obs.copy())
        collisions.append(1 if info.get("crash", False) else 0)
        if terminated or truncated:
            causes = [k for k in ("crash_vehicle", "crash_object",
                                  "out_of_road", "arrive_dest") if info.get(k, False)]
            cause_str = ", ".join(causes) if causes else "horizonte/truncado"
            print(f"    Ep {ep_idx + 1:>2}/{n_episodes} seed={episode_seed}: "
                  f"{step + 1:>4} pasos — {cause_str}")
            break
        obs = obs_next
        step += 1
    return {
        "observations": np.array(observations, dtype=np.float32),
        "collisions": np.array(collisions, dtype=np.int8),
    }


def main():
    args = parse_args()
    policy = args.policy
    policy_name = POLICY_NAME[policy]
    models_dir = Path(args.models_dir) if args.models_dir else ROOT / "results" / "phase1" / policy
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / f"dataset_base_{policy_name}.pkl"

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 9 modelos: 3 semillas × 3 checkpoints (guardados por phase1/train.py)
    models = []
    for seed in SEEDS:
        for step in STEPS:
            ckpt_dir = models_dir / f"checkpoints_seed{seed}"
            models.append({
                "name": f"{policy}_seed{seed}_{step}_steps",
                "model_path": str(ckpt_dir / f"{policy}_seed{seed}_{step}_steps"),
                "vecnorm_path": str(ckpt_dir / f"{policy}_seed{seed}_vecnormalize_{step}_steps.pkl"),
                "seed": seed,
                "steps": step,
            })

    offset = POLICY_OFFSET[policy]
    print(f"Política:        {policy_name}")
    print(f"Modelos:         {len(models)} (seeds {SEEDS} × steps {STEPS})")
    print(f"Seeds asignados: [{offset}..{offset + len(models) * 3 * args.episodes - 1}]")

    episodes = []
    combo_id = 0
    total_combos = len(models) * len(LEVELS)

    for model_info in models:
        print(f"\n{'=' * 65}")
        print(f"  Cargando modelo: {model_info['name']}")
        print(f"{'=' * 65}")
        model, vec_norm = load_model_and_vecnorm(
            ALGO_CLASS[policy], model_info["model_path"], model_info["vecnorm_path"])

        for level in LEVELS:
            # Seeds exclusivas: offset por política + combo_id * episodios
            base_seed = offset + combo_id * args.episodes
            episode_seeds = list(range(base_seed, base_seed + args.episodes))
            combo_id += 1

            print(f"\n  [{combo_id}/{total_combos}] {model_info['name']} — {level['id']}"
                  f"  seeds [{episode_seeds[0]}..{episode_seeds[-1]}]"
                  f"  (density={level['traffic_density']}, accident={level['accident_prob']})")

            env = make_dataset_env(
                level["traffic_density"], level["accident_prob"],
                start_seed=0, num_scenarios=NUM_SCENARIOS, horizon=HORIZON,
            )
            try:
                for ep_idx, ep_seed in enumerate(episode_seeds):
                    ep_data = collect_episode(env, model, vec_norm, ep_idx,
                                              args.episodes, episode_seed=ep_seed)
                    episodes.append({
                        "policy": policy_name,
                        "model": model_info["name"],
                        "seed": model_info["seed"],
                        "steps": model_info["steps"],
                        "level": level["id"],
                        "traffic_density": level["traffic_density"],
                        "accident_prob": level["accident_prob"],
                        "episode_idx": ep_idx,
                        "episode_seed": ep_seed,
                        "observations": ep_data["observations"],
                        "collisions": ep_data["collisions"],
                    })
            finally:
                env.close()

    total_frames = sum(len(e["observations"]) for e in episodes)
    total_crashes = sum(1 for e in episodes if e["collisions"].max() == 1)
    print(f"\n{'=' * 65}")
    print(f"  Recolección finalizada — {policy_name}")
    print(f"  Episodios:           {len(episodes):>10,}")
    print(f"  Frames totales:      {total_frames:>10,}")
    print(f"  Episodios con crash: {total_crashes:>10,}")
    print(f"{'=' * 65}")

    with open(output_file, "wb") as f:
        pickle.dump(episodes, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(output_file) / 1024 ** 2
    print(f"\nDataset guardado: {output_file} ({size_mb:.1f} MB)")

    # Verificación: todas las episode_seeds son únicas
    all_seeds = [e["episode_seed"] for e in episodes]
    assert len(all_seeds) == len(set(all_seeds)), "ERROR: seeds duplicadas"
    print(f"Verificación seeds: {len(set(all_seeds))} únicas ✓")


if __name__ == "__main__":
    main()
