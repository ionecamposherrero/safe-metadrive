"""Evaluación de políticas entrenadas sobre escenarios de test.
"""

import json
import os

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def run_episodes(model, env, n_episodes, normalized, cost_key="cost"):
    """Ejecuta n_episodes deterministas y devuelve métricas por episodio.

    cost_key: clave de info usada como coste por paso ("cost" en Fases 0/1,
    "cost_hard" en Fase 2 para contar solo colisiones reales).
    """
    ep_rewards, ep_costs, ep_successes, ep_lengths = [], [], [], []
    ep_crash_v, ep_crash_o = [], []

    if normalized:
        obs = env.reset()
    else:
        obs, _ = env.reset()

    ep_reward, ep_cost, ep_length = 0.0, 0.0, 0

    while len(ep_rewards) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        if normalized:
            obs, rewards, dones, infos = env.step(action)
            reward, done, info = rewards[0], dones[0], infos[0]
        else:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        ep_reward += reward
        ep_cost += info.get(cost_key, 0.0)
        ep_length += 1

        if done:
            ep_rewards.append(ep_reward)
            ep_costs.append(ep_cost)
            ep_successes.append(float(info.get("arrive_dest", False)))
            ep_lengths.append(ep_length)
            ep_crash_v.append(float(info.get("crash_vehicle", False)))
            ep_crash_o.append(float(info.get("crash_object", False)))

            ep_reward, ep_cost, ep_length = 0.0, 0.0, 0
            if normalized:
                obs = env.reset()
            else:
                obs, _ = env.reset()

    return {
        "cumulative_reward": ep_rewards,
        "cumulative_cost": ep_costs,
        "success_rate": ep_successes,
        "ep_length": ep_lengths,
        "crash_v_rate": ep_crash_v,
        "crash_o_rate": ep_crash_o,
    }


def evaluate_policy_seeds(model_class, model_prefix, vecnorm_prefix, env_factory,
                          seeds, n_episodes, device="auto", cost_key="cost",
                          label="", output_json=None):
    """Evalúa las políticas de varias semillas y agrega media ± std.

    - model_class: clase para cargar el modelo (PPO, SAC, PPOLagrangian, ...).
    - model_prefix / vecnorm_prefix: rutas sin el número de semilla
      (p. ej. "results/phase1/sac/sac_seed").
    - env_factory: callable sin argumentos que crea el entorno de test.
    """
    per_seed = {}

    for seed in seeds:
        vecnorm_path = f"{vecnorm_prefix}{seed}.pkl"
        normalized = os.path.exists(vecnorm_path)
        print(f"Seed {seed} {'con' if normalized else 'SIN'} VecNormalize")

        model = model_class.load(f"{model_prefix}{seed}", device=device)
        env = env_factory()

        if normalized:
            env = DummyVecEnv([lambda env=env: env])
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False
            env.norm_reward = False

        try:
            per_seed[seed] = run_episodes(model, env, n_episodes, normalized, cost_key)
        finally:
            env.close()
        del model

    # ── Agregación entre semillas ─────────────────────────────────────────
    r = [np.mean(per_seed[s]["cumulative_reward"]) for s in seeds]
    c = [np.mean(per_seed[s]["cumulative_cost"]) for s in seeds]
    su = [np.mean(per_seed[s]["success_rate"]) * 100 for s in seeds]
    cv = [np.mean(per_seed[s]["crash_v_rate"]) * 100 for s in seeds]
    co = [np.mean(per_seed[s]["crash_o_rate"]) * 100 for s in seeds]

    summary = {
        "reward": {"mean": float(np.mean(r)), "std": float(np.std(r))},
        "cost": {"mean": float(np.mean(c)), "std": float(np.std(c))},
        "success_rate": {"mean": float(np.mean(su)), "std": float(np.std(su))},
        "crash_vehicle_rate": {"mean": float(np.mean(cv)), "std": float(np.std(cv))},
        "crash_object_rate": {"mean": float(np.mean(co)), "std": float(np.std(co))},
    }

    # ── Tabla ─────────────────────────────────────────────────────────────
    w_meth, w_col = 12, 18
    print(f"\n{'=' * 107}")
    titulo = f"Test Performance — {label} (media ± std entre semillas)"
    print(f"{titulo:^107}")
    print(f"{'=' * 107}")
    print(f"{'Method':<{w_meth}} {'Reward':^{w_col}} {'Cost':^{w_col}} "
          f"{'Success %':^{w_col}} {'Crash V. %':^{w_col}} {'Crash O. %':^{w_col}}")
    print(f"{'-' * 107}")
    print(
        f"{label:<{w_meth}} "
        f"{summary['reward']['mean']:.1f} ± {summary['reward']['std']:.1f}".ljust(w_meth + w_col) +
        f"{summary['cost']['mean']:.2f} ± {summary['cost']['std']:.2f}".center(w_col) +
        f"{summary['success_rate']['mean']:.1f} ± {summary['success_rate']['std']:.1f}".center(w_col) +
        f"{summary['crash_vehicle_rate']['mean']:.1f} ± {summary['crash_vehicle_rate']['std']:.1f}".center(w_col) +
        f"{summary['crash_object_rate']['mean']:.1f} ± {summary['crash_object_rate']['std']:.1f}".center(w_col)
    )
    print(f"{'=' * 107}")

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w") as f:
            json.dump({
                "summary": summary,
                "per_seed_results": {str(s): v for s, v in per_seed.items()},
            }, f, indent=2)
        print(f"Resultados guardados en {output_json}")

    return summary, per_seed
