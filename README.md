# Aprendizaje por Refuerzo Profundo Seguro para Conducción Autónoma en MetaDrive

Este trabajo aborda el problema de la **seguridad en conducción
autónoma** mediante algoritmos de Aprendizaje por Refuerzo Profundo (PPO, SAC) y
Aprendizaje por Refuerzo Profundo Seguro (PPO-Lag, SAC-Lag) sobre el simulador
[MetaDrive](https://github.com/metadriverse/metadrive). Adicionalmente, se
propone un **enfoque híbrido** que combina RL con un modelo supervisado de
predicción de riesgo, cuya salida se incorpora como señal densa de coste para
que el agente anticipe las situaciones de peligro y reduzca las colisiones.

## Fases del proyecto

| Fase | Descripción |
|------|-------------|
| **0** | Escenario **sin tráfico**. Comparación de la configuración por defecto de MetaDrive/Stable-Baselines3 frente a la función de recompensa rediseñada y los hiperparámetros optimizados (PPO y SAC). |
| **1** | Escenario **con tráfico y obstáculos** (SafeMetaDrive). Comparación de PPO y SAC (coste por colisión restado en la recompensa) frente a sus variantes Lagrangianas PPO-Lag y SAC-Lag (coste como restricción explícita). |
| **2** | **Penalización basada en predicción de riesgo.** Se recolectan trayectorias con las 36 políticas de la Fase 1, se etiquetan con un riesgo por descuento temporal `r_t = γ^(T−t)` (γ ∈ {0.8, 0.9, 0.95}), se entrena un modelo supervisado de predicción de riesgo y su predicción se usa como coste denso durante el RL. |

**Resultado principal:** SAC-Lag con el modelo de riesgo (γ = 0.9) reduce las
colisiones entre un 46 % y un 67 % respecto a la Fase 1 con una caída de la
tasa de éxito inferior a 4.2 puntos porcentuales en todos los niveles.

## Estructura del repositorio

```
├── informe.md                  # Memoria del TFG
├── requirements.txt
├── src/                        # Código común
│   ├── envs.py                 # Entornos MetaDrive + recompensa optimizada + coste de riesgo
│   ├── agents.py               # Hiperparámetros y construcción de PPO/SAC/PPO-Lag/SAC-Lag
│   ├── ppo_lagrangian.py       # PPO-Lag (implementación propia sobre SB3)
│   ├── sac_lagrangian.py       # SAC-Lag (implementación propia sobre SB3)
│   ├── risk_model.py           # Modelo de predicción de riesgo (dos cabezas, zero-inflated)
│   ├── callbacks.py            # Callback de métricas de entrenamiento
│   └── evaluation.py           # Bucle de evaluación común
├── scripts/
│   ├── phase0/
│   │   ├── train.py            # Entrenamiento sin tráfico (default | optimized)
│   │   └── evaluate.py         # Test en 100 escenarios no vistos
│   ├── phase1/
│   │   ├── train.py            # Entrenamiento con tráfico (ppo | sac | ppo_lag | sac_lag)
│   │   └── evaluate.py         # Test en 300 escenarios × 3 niveles de congestión
│   └── phase2/
│       ├── collect_dataset.py  # 1. Recolección de trayectorias con políticas de la Fase 1
│       ├── build_dataset.py    # 2. Etiquetado por descuento temporal y fusión
│       ├── train_risk_model.py # 3. Entrenamiento del modelo de riesgo (por γ)
│       ├── train.py            # 4. RL con el riesgo predicho como coste denso
│       └── evaluate.py         # 5. Test (el coste reportado cuenta solo colisiones)
├── FASE 0/, FASE 1/, FASE 2/   # Notebooks originales y resultados de los experimentos
└── results/                    # Salidas de los scripts (modelos, logs, JSON) — no versionado
```

## Instalación

Requiere Python ≥ 3.9. Se recomienda un entorno virtual:

```bash
pip install -r requirements.txt
```

> En Windows, MetaDrive puede requerir además `pip install metadrive-simulator[all]`
> para los assets de renderizado. El entrenamiento no necesita render.

## Uso

### Fase 0 — sin tráfico

```bash
# Configuración por defecto (baseline) y configuración optimizada
python scripts/phase0/train.py --algo ppo --config default
python scripts/phase0/train.py --algo sac --config optimized

python scripts/phase0/evaluate.py --algo sac --config optimized
```

### Fase 1 — con tráfico y obstáculos

```bash
python scripts/phase1/train.py --algo sac_lag          # también: ppo, sac, ppo_lag
python scripts/phase1/evaluate.py --algo sac_lag --level all
```

Cada entrenamiento usa 3 semillas (configurable con `--seeds`), 3M de pasos y
guarda un checkpoint cada 1M de pasos (necesarios para la Fase 2).

### Fase 2 — modelo de predicción de riesgo

```bash
# 1. Recolectar trayectorias con las políticas de la Fase 1 (una vez por algoritmo)
python scripts/phase2/collect_dataset.py --policy ppo
python scripts/phase2/collect_dataset.py --policy sac
python scripts/phase2/collect_dataset.py --policy ppo_lag
python scripts/phase2/collect_dataset.py --policy sac_lag

# 2. Etiquetar y fusionar → dataset_final_multilabel.npz
python scripts/phase2/build_dataset.py

# 3. Entrenar los modelos de riesgo (uno por γ)
python scripts/phase2/train_risk_model.py --gammas 0.8 0.9 0.95

# 4. Entrenar el agente con el coste de riesgo
python scripts/phase2/train.py --algo sac_lag --risk-gamma 0.9

# 5. Evaluar
python scripts/phase2/evaluate.py --algo sac_lag --risk-gamma 0.9 --level all
```

## Configuración experimental

- **Observación:** `LidarStateObservation` (259 dims = 9 estado ego + 10 navegación + 240 LiDAR).
- **Acción:** continua, `[dirección, aceleración/freno] ∈ [−1, 1]²`.
- **Mapas:** 3 bloques generados proceduralmente (algoritmo BIG), 2 carriles de 3.5 m, horizonte de 1000 pasos.
- **Entrenamiento:** 3M de pasos, 8 entornos en paralelo, 3 semillas; 500 escenarios (Fase 0) o 1000 (Fases 1–2) con `traffic_density=0.15` y `accident_prob=0.4`.
- **Test:** escenarios no vistos (100 en Fase 0; 300 en Fases 1–2 con niveles de congestión baja/media/alta).
- **Redes:** MLP de 2×256 para actor y críticos; el modelo de riesgo usa un encoder LiDAR convolucional circular + encoder tabular residual con dos cabezas (clasificación + magnitud).

Los hiperparámetros de cada algoritmo están centralizados en
[`src/agents.py`](src/agents.py).
