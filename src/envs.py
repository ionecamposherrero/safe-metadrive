"""Entornos MetaDrive con función de recompensa optimizada.

Contiene:
  - ``OptimizedMetaDriveEnv``  — Fase 0 (configuración optimizada, sin tráfico).
  - ``SafeOptimizedEnv``       — Fases 1 y 2: entorno con tráfico/obstáculos y coste
    por colisión. Si ``cost_in_reward=True`` el coste se resta de la recompensa
    (PPO y SAC estándar); con ``False`` la señal de coste queda separada para los
    algoritmos Lagrangianos.
  - ``RiskCostEnv``            — Fase 2: igual que ``SafeOptimizedEnv`` pero el coste
    denso lo proporciona el modelo de predicción de riesgo (C'_t = 1 si colisión,
    r̂_t en caso contrario).
  - Factorías ``make_*`` usadas por los scripts de entrenamiento/evaluación.
"""

import numpy as np

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.utils import clip


# ══════════════════════════════════════════════════════════════════════════
# Recompensa optimizada (común a todas las fases con configuración propia)
# ══════════════════════════════════════════════════════════════════════════

# Hiperparámetros de la velocidad deseada V_des
A_LAT_MAX = 3.5    # m/s² — aceleración lateral máxima admisible
K_ANGLE   = 0.7    # peso de la penalización por ángulo de giro ∈ [0, 1]
RADIUS_MAX = 60.0  # m — radio máximo del entorno (BlockParameterSpace)
V_MIN_KMH = 5.0    # km/h — velocidad mínima deseada (evita divisiones ~0)


def compute_vdes(vehicle):
    """Velocidad deseada según la geometría local de la vía.

    V_des = clip(sqrt(a_lat · R) · 3.6 · (1 − k·θ_norm), V_min, V_max),
    evaluada en los dos checkpoints de navegación; devuelve el más restrictivo.
    """
    navi = vehicle.navigation
    vmax = vehicle.max_speed_km_h          # km/h
    n_lanes = navi.get_current_lane_num()
    lane_w = navi.get_current_lane_width()  # m

    def _vdes_single(bend_norm, angle_norm):
        if bend_norm < 1e-3:               # tramo recto
            return vmax
        r_real = bend_norm * (RADIUS_MAX + n_lanes * lane_w)        # metros
        v_radius = np.sqrt(A_LAT_MAX * r_real) * 3.6                # m/s → km/h
        v_curve = v_radius * (1.0 - K_ANGLE * angle_norm)
        return float(np.clip(v_curve, V_MIN_KMH, vmax))

    # navi_info tiene 10 dims: [cp0 x5 | cp1 x5]
    navi_info = navi._navi_info
    vdes_cp0 = _vdes_single(navi_info[2], navi_info[4])   # checkpoint actual
    vdes_cp1 = _vdes_single(navi_info[7], navi_info[9])   # checkpoint siguiente
    return min(vdes_cp0, vdes_cp1)


def optimized_dense_reward(env, vehicle):
    """Términos densos de la recompensa optimizada: r_drive + r_speed + r_checkpoint.

    Devuelve (reward, step_info) sin aplicar coste ni recompensa terminal.
    """
    step_info = dict()

    # ── Carril de referencia y sentido de la marcha ──────────────────────
    if vehicle.lane in vehicle.navigation.current_ref_lanes:
        current_lane = vehicle.lane
        positive_road = 1
    else:
        current_lane = vehicle.navigation.current_ref_lanes[0]
        current_road = vehicle.navigation.current_road
        positive_road = 1 if not current_road.is_negative_road() else -1

    long_last, _ = current_lane.local_coordinates(vehicle.last_position)
    long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

    # ── Factor lateral progresivo con la velocidad ───────────────────────
    if env.config["use_lateral_reward"]:
        v_min, v_max = 5.0, 20.0  # km/h: sin penalización por debajo, completa por encima
        current_speed = vehicle.speed_km_h
        interp_weight = np.clip((current_speed - v_min) / (v_max - v_min), 0.0, 1.0)
        target_lateral_factor = clip(
            1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(),
            0.0, 1.0,
        )
        lateral_factor = (1.0 - interp_weight) + (interp_weight * target_lateral_factor)
    else:
        lateral_factor = 1.0

    # ── Término 1: avance longitudinal en el carril ──────────────────────
    r_drive = env.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road

    # ── Término 2: velocidad respecto a V_des (asimétrica) ───────────────
    vdes = compute_vdes(vehicle)
    v = vehicle.speed_km_h
    if v <= vdes:
        speed_factor = v / vdes
    else:
        excess = (v - vdes) / vdes
        speed_factor = 1.0 - excess ** 2
    r_speed = env.config["speed_reward"] * speed_factor * positive_road

    # ── Término 3: alineación con el checkpoint próximo ──────────────────
    lanes_heading = vehicle.navigation.navi_arrow_dir[0]
    ego_heading = vehicle.heading_theta
    heading_diff = np.cos(lanes_heading - ego_heading)
    r_checkpoint = 0.5 * heading_diff * positive_road if vehicle.speed_km_h > 5.0 else 0.0

    reward = r_drive + r_speed + r_checkpoint

    step_info["r_drive"] = r_drive
    step_info["r_speed"] = r_speed
    step_info["r_checkpoint"] = r_checkpoint
    step_info["vdes_kmh"] = vdes
    step_info["speed_kmh"] = v
    step_info["speed_factor"] = speed_factor
    step_info["route_completion"] = vehicle.navigation.route_completion
    return reward, step_info


def _apply_terminal_reward(env, vehicle, reward):
    """Recompensa dispersa de terminación: anula el resto de términos."""
    if env._is_arrive_destination(vehicle):
        return +env.config["success_reward"]
    if env._is_out_of_road(vehicle):
        return -env.config["out_of_road_penalty"]
    return reward


# ══════════════════════════════════════════════════════════════════════════
# Fase 0 — entorno sin tráfico con recompensa optimizada
# ══════════════════════════════════════════════════════════════════════════
class OptimizedMetaDriveEnv(MetaDriveEnv):
    """MetaDriveEnv con la función de recompensa optimizada (sin coste)."""

    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        reward, step_info = optimized_dense_reward(self, vehicle)
        reward = _apply_terminal_reward(self, vehicle, reward)
        step_info["step_reward"] = reward
        return reward, step_info


# ══════════════════════════════════════════════════════════════════════════
# Fase 1 — entorno con tráfico, coste por colisión
# ══════════════════════════════════════════════════════════════════════════
class SafeOptimizedEnv(SafeMetaDriveEnv):
    """SafeMetaDriveEnv con recompensa optimizada y coste = 1 por colisión.

    Config extra:
      - ``cost_in_reward`` (bool): si True, el coste se resta de la recompensa
        (PPO/SAC estándar). Los Lagrangianos lo dejan a False y reciben el coste
        como señal separada vía ``info["cost"]``.
    """

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(dict(cost_in_reward=False), allow_add_new_key=True)
        return config

    def cost_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        cost = 0
        if vehicle.crash_vehicle:
            cost = self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            cost = self.config["crash_object_cost"]
        step_info["cost"] = cost
        return cost, step_info

    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        reward, step_info = optimized_dense_reward(self, vehicle)

        if self.config["cost_in_reward"]:
            cost, _ = self.cost_function(vehicle_id)
            reward -= cost

        reward = _apply_terminal_reward(self, vehicle, reward)
        step_info["step_reward"] = reward
        return reward, step_info


# ══════════════════════════════════════════════════════════════════════════
# Fase 2 — coste denso basado en el modelo de predicción de riesgo
# ══════════════════════════════════════════════════════════════════════════
class RiskCostEnv(SafeOptimizedEnv):
    """Entorno de la Fase 2: C'_t = 1 si hay colisión, r̂_t (riesgo predicho) si no.

    Config extra:
      - ``risk_model_path``: checkpoint .pt del modelo de riesgo.
      - ``risk_scaler_path``: StandardScaler de las variables tabulares.
      - ``risk_device``: "cuda" o "cpu" para la inferencia del modelo.
    """

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(
            dict(risk_model_path="", risk_scaler_path="", risk_device="cpu"),
            allow_add_new_key=True,
        )
        return config

    def __init__(self, config):
        super().__init__(config)
        # Observación LidarState para alimentar el modelo de riesgo en cost_function
        from metadrive.obs.state_obs import LidarStateObservation
        self.lidar_obs = LidarStateObservation(self.config)
        self._risk_predictor = None

    def _predict_risk(self, vehicle):
        if self._risk_predictor is None:
            from src.risk_model import RiskPredictor
            self._risk_predictor = RiskPredictor(
                model_path=self.config["risk_model_path"],
                scaler_path=self.config["risk_scaler_path"],
                device=self.config["risk_device"],
            )
        obs_raw = self.lidar_obs.observe(vehicle)
        return self._risk_predictor.predict(obs_raw)

    def cost_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        cost_hard, cost_risk = 0.0, 0.0

        risk_pred = self._predict_risk(vehicle)

        if vehicle.crash_vehicle or vehicle.crash_object:
            cost_hard = 1.0
            cost = cost_hard
        else:
            cost_risk = risk_pred
            cost = cost_risk

        step_info["cost_hard"] = cost_hard
        step_info["cost_risk"] = cost_risk
        step_info["risk_pred"] = risk_pred
        step_info["cost"] = cost
        return cost, step_info


# ══════════════════════════════════════════════════════════════════════════
# Factorías de entornos
# ══════════════════════════════════════════════════════════════════════════
MAP_CONFIG = {
    BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
    BaseMap.GENERATE_CONFIG: 3,   # 3 bloques por mapa
    BaseMap.LANE_WIDTH: 3.5,
    BaseMap.LANE_NUM: 2,
}

# Pesos de la recompensa optimizada y recompensas terminales
REWARD_CONFIG = dict(
    driving_reward=2.5,
    speed_reward=1.5,
    success_reward=100.0,
    out_of_road_penalty=25.0,
    use_lateral_reward=True,
)

# Niveles de congestión usados en test (tráfico, obstáculos)
TEST_LEVELS = {
    "baja":  dict(traffic_density=0.05, accident_prob=0.10),
    "media": dict(traffic_density=0.10, accident_prob=0.25),
    "alta":  dict(traffic_density=0.15, accident_prob=0.40),
}


def make_phase0_default_env(start_seed=0, num_scenarios=500, horizon=1000):
    """Fase 0, configuración base: MetaDriveEnv con recompensa por defecto."""
    return MetaDriveEnv(dict(
        map_config=MAP_CONFIG,
        horizon=horizon,
        num_scenarios=num_scenarios,
        start_seed=start_seed,
        traffic_density=0,
        accident_prob=0,
        use_lateral_reward=True,
        log_level=50,
    ))


def make_phase0_optimized_env(start_seed=0, num_scenarios=500, horizon=1000):
    """Fase 0, configuración optimizada: recompensa rediseñada, sin tráfico."""
    return OptimizedMetaDriveEnv(dict(
        map_config=MAP_CONFIG,
        horizon=horizon,
        num_scenarios=num_scenarios,
        start_seed=start_seed,
        traffic_density=0,
        accident_prob=0,
        log_level=50,
        **REWARD_CONFIG,
    ))


def make_phase1_env(start_seed=0, num_scenarios=1000, horizon=1000,
                    traffic_density=0.15, accident_prob=0.4,
                    cost_in_reward=False):
    """Fases 1: SafeMetaDrive con tráfico/obstáculos y coste por colisión."""
    return SafeOptimizedEnv(dict(
        map_config=MAP_CONFIG,
        horizon=horizon,
        num_scenarios=num_scenarios,
        start_seed=start_seed,
        traffic_density=traffic_density,
        accident_prob=accident_prob,
        crash_vehicle_cost=1.0,
        crash_object_cost=1.0,
        traffic_mode="basic",
        cost_in_reward=cost_in_reward,
        log_level=50,
        **REWARD_CONFIG,
    ))


def make_phase2_env(risk_model_path, risk_scaler_path, risk_device="cpu",
                    start_seed=0, num_scenarios=1000, horizon=1000,
                    traffic_density=0.15, accident_prob=0.4,
                    cost_in_reward=False):
    """Fase 2: coste denso del modelo de predicción de riesgo."""
    return RiskCostEnv(dict(
        map_config=MAP_CONFIG,
        horizon=horizon,
        num_scenarios=num_scenarios,
        start_seed=start_seed,
        traffic_density=traffic_density,
        accident_prob=accident_prob,
        crash_vehicle_cost=1.0,
        crash_object_cost=1.0,
        traffic_mode="basic",
        cost_in_reward=cost_in_reward,
        risk_model_path=str(risk_model_path),
        risk_scaler_path=str(risk_scaler_path),
        risk_device=risk_device,
        log_level=50,
        **REWARD_CONFIG,
    ))


def make_dataset_env(traffic_density, accident_prob, start_seed=0,
                     num_scenarios=3240, horizon=1000):
    """Entorno de recolección de trayectorias (Fase 2): SafeMetaDrive estándar.

    Los episodios terminan en la primera colisión (crash_*_done=True) para que
    el riesgo etiquetado no supere 1.
    """
    return SafeMetaDriveEnv(dict(
        map_config=MAP_CONFIG,
        horizon=horizon,
        num_scenarios=num_scenarios,
        start_seed=start_seed,
        traffic_density=traffic_density,
        accident_prob=accident_prob,
        log_level=50,
        traffic_mode="basic",
        random_traffic=False,
        crash_vehicle_done=True,
        crash_object_done=True,
    ))
