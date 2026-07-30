"""Callback de métricas de entrenamiento común a todas las fases.

Registra recompensa, longitud de episodio, tasa de éxito, coste y tasas de
colisión. Si el modelo es Lagrangiano (tiene ``log_lambda``) registra también λ,
y si el entorno produce ``cost_hard``/``risk_pred`` (Fase 2) los desglosa.
"""

from collections import defaultdict

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


class MetricsCallback(BaseCallback):

    def __init__(self, log_interval=8000, verbose=1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.history = defaultdict(list)
        self._ep_rewards = defaultdict(float)
        self._ep_lengths = defaultdict(int)
        self._ep_costs = defaultdict(float)
        self._ep_cost_hard = defaultdict(float)
        self._rewards_done = []
        self._lengths_done = []
        self._successes = []
        self._costs_done = []
        self._cost_hard_done = []
        self._crash_vehicles = []
        self._crash_objects = []

    def _on_step(self):
        if isinstance(self.training_env, VecNormalize):
            real_rewards = self.training_env.get_original_reward()
        else:
            real_rewards = self.locals["rewards"]

        for i, (reward, done, info) in enumerate(zip(
            real_rewards,
            self.locals["dones"],
            self.locals["infos"],
        )):
            self._ep_rewards[i] += reward
            self._ep_lengths[i] += 1
            self._ep_costs[i] += info.get("cost", 0)
            self._ep_cost_hard[i] += info.get("cost_hard", 0.0)

            if done:
                self._rewards_done.append(self._ep_rewards[i])
                self._lengths_done.append(self._ep_lengths[i])
                self._successes.append(float(info.get("arrive_dest", False)))
                self._costs_done.append(self._ep_costs[i])
                self._cost_hard_done.append(self._ep_cost_hard[i])
                self._crash_vehicles.append(float(info.get("crash_vehicle", False)))
                self._crash_objects.append(float(info.get("crash_object", False)))
                self._ep_rewards[i] = 0
                self._ep_lengths[i] = 0
                self._ep_costs[i] = 0
                self._ep_cost_hard[i] = 0

        if self.num_timesteps % self.log_interval == 0 and self._rewards_done:
            mean_reward = np.mean(self._rewards_done)
            mean_length = np.mean(self._lengths_done)
            success_rate = np.mean(self._successes) * 100
            mean_cost = np.mean(self._costs_done)
            mean_cost_hard = np.mean(self._cost_hard_done)
            crash_vehicle_rate = np.mean(self._crash_vehicles) * 100
            crash_object_rate = np.mean(self._crash_objects) * 100

            # λ actual (solo en modelos Lagrangianos)
            lam_val = float("nan")
            if hasattr(self.model, "log_lambda"):
                lam_val = float(torch.exp(self.model.log_lambda.detach()).item())

            self.history["timestep"].append(self.num_timesteps)
            self.history["mean_reward"].append(mean_reward)
            self.history["mean_ep_length"].append(mean_length)
            self.history["success_rate"].append(success_rate)
            self.history["mean_ep_cost"].append(mean_cost)
            self.history["mean_cost_hard"].append(mean_cost_hard)
            self.history["crash_vehicle_rate"].append(crash_vehicle_rate)
            self.history["crash_object_rate"].append(crash_object_rate)
            self.history["lambda"].append(lam_val)

            if self.verbose:
                extra = f" | λ: {lam_val:7.4f}" if hasattr(self.model, "log_lambda") else ""
                print(
                    f"[{self.num_timesteps:>8d} steps] "
                    f"reward: {mean_reward:6.2f} | "
                    f"ep_len: {mean_length:6.1f} | "
                    f"success: {success_rate:5.1f}% | "
                    f"cost: {mean_cost:6.2f} | "
                    f"crash v: {crash_vehicle_rate:5.1f}% | "
                    f"crash o: {crash_object_rate:5.1f}%"
                    f"{extra}"
                )
            self._rewards_done.clear()
            self._lengths_done.clear()
            self._successes.clear()
            self._costs_done.clear()
            self._cost_hard_done.clear()
            self._crash_vehicles.clear()
            self._crash_objects.clear()

        return True
