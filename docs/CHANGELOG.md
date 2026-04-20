# Diario de cambios — Mario RL

Registro cronológico de todas las decisiones, problemas y ajustes realizados durante el entrenamiento del agente PPO para Super Mario Bros 1-2.

---

## Sesión 1 — Setup y migración (19 abril 2026)

### Punto de partida
- Código legacy: 5 archivos .py sueltos, sin estructura, con varios bugs
- Objetivo: agente PPO que complete el nivel 1-2 de Super Mario Bros
- Hardware: HP Omen 16, RTX 3060 Laptop (6 GB VRAM), CUDA 12.1

### Migración a WSL
- **Problema**: `nes-py` requiere compilación con gcc. En Windows necesita Visual Studio Build Tools (no instalados).
- **Decisión**: mover todo a Ubuntu 24.04 en WSL. Editar en Windows, ejecutar en WSL.
- Python 3.12.3, torch 2.5.1+cu121, stable-baselines3 2.1.0

### Refactor completo del código
Reestructuración en módulos:
```
app.py → TUI con Rich
configs/ → schema tipado + YAMLs
env/ → wrappers + reward shaping
models/ → save/load + factory PPO
training/ → Trainer thread + SharedState + callbacks
ui/ → dashboard FastAPI + WebSockets
tests/ → 27 unit tests
```

### 9 bugs corregidos de la versión legacy
1. `resume_path` tenía una rama muerta que nunca se ejecutaba
2. `resize_shape` era `int` pero `ResizeObservation` requiere tupla
3. Detección de muerte usaba `life < 2` (fijo) en vez de `life < prev_life` (flanco descendente)
4. Cálculo de media usaba slicing `[-100:]` → O(n). Reemplazado por `RingBuffer` → O(1)
5. Comentario decía `SIMPLE_MOVEMENT` pero el código usaba `COMPLEX_MOVEMENT`
6. README decía 2M steps pero el config tenía 5M
7. gym 0.26 API mismatch: wrappers custom devolvían 4-tupla, `gym.wrappers` devuelve 5-tupla
8. `GrayScaleObservation` producía `(84,84,1)` → nuevo `SqueezeChannelObservation` para quitar el canal extra
9. `JoypadSpace.reset()` no acepta kwargs → try/except en `AdvancedRewardWrapper.reset()`

### Reward shaping diseñado (17 componentes)
**Base (8):** forward, coin, score, vertical_explore, flag, time_penalty, stuck, backward

**Nuevos (9):**
- Muerte por tiempo vs enemigo (penalización diferenciada)
- Detección de patrones cíclicos de acciones
- Mapa de muertes persistente (death_map.json)
- Penalización por retroceso excesivo
- Detección de micro-movimientos sin progreso neto
- Stuck adaptativo (umbral varía según progreso en el nivel)
- Detección de muro (presionar derecha sin moverse)
- Bonus por récord de distancia histórico
- Bonus por récord de tiempo

### Resultado de la sesión
- 27/27 tests pasando
- CUDA detecta RTX 3060
- Training debug validado: Mario llegó a x=678 en 7,500 steps
- Dashboard importable y serveable

---

## Sesión 2 — Primer entrenamiento real (19 abril 2026)

### Config inicial
```yaml
ppo:
  learning_rate: 0.00025
  n_steps: 512
  batch_size: 64
  n_epochs: 10
  ent_coef: 0.02
reward:
  forward_reward_coef: 1.0
  stuck_penalty_base: -0.5
  wall_stuck_penalty: -0.3
  micro_movement_penalty: -1.0
  excessive_left_penalty: -0.1
training:
  total_timesteps: 5000000
  n_envs: 1  # un solo environment
```

### Observaciones a 100k steps
| Métrica | Valor |
|---|---|
| steps/s | ~65 |
| distancia avg100 | 186 |
| max x histórico | 1310 |
| reward avg100 | -307.79 |
| clear rate | 0% |

- **Problema 1**: velocidad muy baja (65 steps/s). A este ritmo, 5M steps tomaría ~21 horas.
- **Causa**: un solo environment. El cuello de botella es el emulador NES (CPU), no la GPU.

### Observaciones a 134k steps
| Métrica | Valor |
|---|---|
| reward avg100 | -438.65 |
| último reward | -7,720 |
| distancia avg100 | 207 |

- **Problema 2**: rewards extremadamente negativos (-7,720 en un episodio).
- **Causa**: Mario se atascaba contra un muro por ~2000 steps. Acumulaba: stuck (-0.5/step) + wall_stuck (-0.3/step) + micro_movement (-1.0/ventana) = miles de penalización antes de que el timer del juego lo matara.

---

## Cambio 1 — SubprocVecEnv (4 environments paralelos)

### Problema
65 steps/s con 1 env. El emulador NES es CPU-bound y la GPU esperaba.

### Solución
Implementar `SubprocVecEnv` de stable-baselines3: 4 emuladores NES corriendo en procesos paralelos.

### Archivos modificados
- `configs/schema.py` — agregado `n_envs: int = 4` a TrainingConfig
- `env/factory.py` — nuevo `ConfigurableEnvWrapper` + `make_vec_env()`
- `training/trainer.py` — usa vec env cuando n_envs > 1, propagación de reward config
- `training/callbacks.py` — acumuladores por environment
- `training/metrics_callback.py` — telemetría de env 0, episodios de todos
- `ui/controller.py` — propagación de reward config a subprocesos
- `env/death_map.py` — método `merge_counts()` para fusionar datos de subprocesos

### Desafíos resueltos
1. **Hot-reload de RewardConfig**: con SubprocVecEnv cada env está en un proceso separado. Solución: `ConfigurableEnvWrapper` expone `update_reward_config()` accesible via `env_method()`.
2. **Death map compartido**: cada subproceso tiene su propio tracker. Al finalizar, el proceso principal recoge y fusiona conteos via `env_method("get_death_counts")`.
3. **Callbacks multi-env**: adapté todos los callbacks para iterar sobre N environments en vez de asumir 1.

### Benchmark de velocidad
| n_envs | steps/s | Speedup |
|---|---|---|
| 1 | 128 | — |
| 2 | 241 | 1.9x |
| **4** | **332** | **2.6x** |

### Config actualizada
```yaml
training:
  n_envs: 4
```

---

## Cambio 2 — Métricas de entrenamiento en TUI

### Problema
No se veía qué porcentaje del entrenamiento iba ni a qué velocidad.

### Solución
Agregué dos métricas nuevas al panel de la terminal:
- **entrenamiento %** — timesteps actuales / total_timesteps
- **steps/s** — timesteps / tiempo transcurrido

### Archivos modificados
- `training/shared_state.py` — agregado `training_start_time: float`
- `app.py` — nuevas filas en `_metrics_panel()`

---

## Cambio 3 — Dashboard web observer

### Problema
La TUI de terminal muestra métricas, pero el usuario quería ver el juego en vivo, trayectorias, y métricas gráficas desde el navegador.

### Solución
Servidor FastAPI read-only que se engancha al mismo `SharedState` que la terminal.

### Archivos creados
- `ui/observer.py` — servidor con WebSockets (métricas, frames, trayectorias)
- `ui/static/observer.html` — dashboard dark theme con:
  - Frame del juego en vivo (~10 fps)
  - Barra de progreso del nivel + del entrenamiento
  - Grid de métricas en tiempo real
  - Mapa de trayectorias en canvas (50 episodios superpuestos)
  - Info de config

### Integración
Se auto-arranca en `app.py` al iniciar. URL: `http://127.0.0.1:8000`

Es 100% read-only — solo observa, no controla el training.

---

## Cambio 4 — Truncamiento por falta de progreso

### Problema a 487k steps
| Métrica | Valor |
|---|---|
| distancia avg100 | 180 |
| max x histórico | 290 |
| reward avg100 | -158.8 |
| behavior score | 3079 |

Mario se quedaba anclado en un lugar, moviéndose sin avanzar, acumulando penalizaciones durante miles de steps hasta que el timer del juego lo mataba. Episodios de 2000+ steps que generaban rewards de -7000.

### Solución
Nuevo mecanismo: `no_progress_limit: 200`. Si Mario no supera su max x en 200 steps consecutivos, el episodio se trunca automáticamente.

### Efecto
- Episodios atascados mueren en ~200 steps en vez de ~2000 → **10x más episodios útiles por hora**
- Rewards menos extremos (-200 en vez de -7000) → **señal más limpia para PPO**
- Más ciclos de exploración → convergencia más rápida

### Archivos modificados
- `configs/schema.py` — agregado `no_progress_limit: int = 200` a RewardConfig
- `env/reward_shaping.py` — contador `_no_progress_counter`, truncamiento cuando excede límite

---

## Cambio 5 — Suavizar penalizaciones + subir incentivo

### Problema
Las penalizaciones eran demasiado agresivas para las etapas tempranas del entrenamiento. El agente recibía tanta señal negativa que no podía distinguir "ir a la derecha es bueno".

### Cambios en default.yaml
| Parámetro | Antes | Después | Por qué |
|---|---|---|---|
| `forward_reward_coef` | 1.0 | **2.0** | Señal más fuerte de "avanzar es bueno" |
| `stuck_penalty_base` | -0.5 | **-0.2** | Menos castigo por quedarse quieto |
| `wall_stuck_penalty` | -0.3 | **-0.1** | Menos castigo por chocar muro |
| `micro_movement_penalty` | -1.0 | **-0.3** | Era el más agresivo, ahogaba el aprendizaje |
| `excessive_left_penalty` | -0.1 | **-0.05** | Tolerar más el retroceso exploratorio |

### Filosofía
En etapas tempranas, el agente necesita explorar. Penalizaciones fuertes lo paralizan: aprende que "hacer cualquier cosa = castigo", así que no hace nada. Mejor dar una señal positiva clara ("ir a la derecha = bien") y castigos suaves.

---

## Cambio 6 — Fix crash LazyFrames

### Problema
Al implementar el truncamiento por falta de progreso, el training crasheaba:
```
AttributeError: 'LazyFrames' object has no attribute 'reshape'
```

### Causa
Cuando SubprocVecEnv trunca un episodio, guarda la observación terminal en `info["terminal_observation"]`. `FrameStack` devuelve `LazyFrames` (objeto lazy), no un numpy array. SB3 intentaba `.reshape()` sobre él → crash.

### Solución
`ConfigurableEnvWrapper` (el wrapper más externo) ahora convierte toda observación a `np.asarray()` en `step()` y `reset()`. Esto garantiza que las observaciones siempre sean numpy arrays compatibles con SB3.

---

## Config actual (post todos los cambios)

```yaml
env:
  env_id: SuperMarioBros-1-2-v0
  frame_stack: 4
  resize_shape: 84
  skip_frames: 4

ppo:
  policy: CnnPolicy
  learning_rate: 0.00025
  n_steps: 512
  batch_size: 64
  n_epochs: 10
  ent_coef: 0.02
  use_linear_lr_schedule: true

reward:
  forward_reward_coef: 2.0          # subido de 1.0
  stuck_penalty_base: -0.2          # suavizado de -0.5
  wall_stuck_penalty: -0.1          # suavizado de -0.3
  micro_movement_penalty: -0.3      # suavizado de -1.0
  excessive_left_penalty: -0.05     # suavizado de -0.1
  no_progress_limit: 200            # NUEVO: trunca episodios estancados

training:
  total_timesteps: 5000000
  n_envs: 4                         # NUEVO: 4 environments paralelos
```

---

## Cambio 7 — Subir exploración y desactivar death map

### Problema a ~400k steps
El agente mostraba un patrón cíclico: subía distancia avg100 hasta ~500, luego caía al mínimo (~180), y repetía. Nunca salía de ese ciclo. `max x hist.` estancado en 1,934 por mucho tiempo.

### Diagnóstico
- **`ent_coef: 0.02` insuficiente**: el agente encontraba una estrategia, un update la destruía, y no tenía suficiente exploración para recuperarse rápido.
- **death_map activo**: penalizaba morir en zonas donde ya había muerto. Esto paralizaba la exploración — el agente evitaba zonas difíciles en vez de aprender a superarlas.
- **Penalizaciones de muerte altas**: -15 por enemigo hacía que el agente priorizara "no morir" sobre "avanzar".

### Cambios en default.yaml
| Parámetro | Antes | Después | Razón |
|---|---|---|---|
| `ent_coef` | 0.02 | **0.04** | Más exploración, el cambio más impactante |
| `flag_bonus` | 15 | **30** | Más incentivo por completar el nivel |
| `death_by_enemy_penalty` | -15 | **-10** | Menos miedo a morir = explora más |
| `death_by_time_penalty` | -8 | **-5** | Idem |
| `enable_death_map` | true | **false** | Dejaba de explorar zonas difíciles |

### Decisión: reiniciar desde cero
La política de 400k steps estaba viciada (patrón cíclico sin progreso). Dado que 400k es solo ~8% del entrenamiento total y se recupera en ~56 minutos, reiniciar desde cero con la nueva config es más eficiente que intentar arreglar una red con malos hábitos.

---

## Run 3 — Resultados con ent_coef 0.04 (19 abril 2026)

### Mejor arranque hasta ahora
A 206k steps, el run 3 fue el mejor de todos:

| Métrica | Run 1 (ent 0.02) | Run 2 (ent 0.02) | **Run 3 (ent 0.04)** |
|---|---|---|---|
| distancia avg100 | 186 | 392 | **638** |
| reward avg100 | -307 | 807 | **1,278** |
| max x hist. | 300 | 1,542 | **1,654** |
| coins avg100 | 0.1 | 0.4 | **1.3** |

### Problema: oscilaciones destructivas
El run mostró un patrón recurrente:
- Subía distancia avg100 a ~600
- Bajaba a ~200
- Subía a ~500
- Bajaba a ~200

A 331k steps: distancia avg100 en 198, max hist estancado en 1,730. Los gráficos de tendencia confirmaron que no salía del ciclo.

### Diagnóstico
**`n_epochs: 10` era demasiado alto.** Con 10 epochs, PPO pasa 10 veces sobre cada batch de 2048 transiciones. Si un batch tiene datos malos (muchos episodios atascados), esas 10 pasadas "sobreescriben" la política buena que tenía. Es la causa principal del catastrophic forgetting severo.

Otros factores:
- `no_progress_limit: 200` todavía desperdiciaba muchos steps en episodios atascados
- `enable_cyclic_detection: true` agregaba ruido de penalización innecesario (redundante con no_progress_limit)

---

## Cambio 8 — Reducir n_epochs y optimizar episodios

### Cambios en default.yaml
| Parámetro | Antes | Después | Razón |
|---|---|---|---|
| `n_epochs` | 10 | **4** | Menos overfitting por batch, updates más suaves, menos catastrophic forgetting |
| `no_progress_limit` | 200 | **120** | Corta episodios atascados en 120 steps en vez de 200 |
| `enable_cyclic_detection` | true | **false** | Redundante con no_progress_limit, solo agregaba ruido negativo |

### Razonamiento
- `n_epochs: 4` es el valor que usa fast_debug.yaml y funciona bien. La literatura de PPO (Schulman et al.) sugiere 3-10 epochs; para environments inestables como Mario, menos es más estable.
- `no_progress_limit: 120` = ~480 frames de juego (~8 segundos). Si Mario no avanza en 8 segundos, está genuinamente atascado.
- Sin detección cíclica, el agente recibe menos señal negativa espuria. Los episodios atascados ya mueren por el no_progress_limit.

### Decisión: reiniciar desde cero
Mismo razonamiento que el cambio 7: la política de 331k steps estaba en un ciclo destructivo, y 331k es solo ~6.6% del total.

---

## Cambio 9 — Modo demostración para clase

### Implementación
Nuevo modo `--demo` para presentar el modelo entrenado en clase:

```bash
python app.py --demo --model-path models_saved/mario_ppo
```

Abre `http://127.0.0.1:8000/demo` con:
- Gameplay grande (512x480, pixelated)
- Barra de progreso del nivel en tiempo real
- Stats en vivo: score, monedas, tiempo, distancia
- Scoreboard con historial de episodios (CLEAR/MUERTO, distancia, %, score, monedas)
- Resumen: total clears, mejor distancia, clear rate

### Archivos creados/modificados
- `ui/static/demo.html` — página de demostración limpia, sin métricas de entrenamiento
- `ui/observer.py` — ruta `/demo`
- `training/trainer.py` — método `run_demo()` con streaming de frames
- `training/shared_state.py` — campos demo (score, coins, time, results)
- `app.py` — flag `--demo`

Se puede elegir cualquier checkpoint: `--model-path checkpoints/mario_ppo_500000`

---

## Cambio 10 — Config basada en investigación (19-20 abril 2026)

### Investigación realizada
Análisis de implementaciones exitosas de PPO para Mario:
- **RL Zoo** (SB3 oficial, tuneado con Optuna para Atari)
- **vietnh1009/Super-mario-bros-PPO-pytorch** (31/32 niveles completados)
- **"The 37 Implementation Details of PPO"** (ICLR blog track)
- **OpenAI Baselines original** (PPO2 Atari)
- **alirezakazemipour/Mario-PPO**, **yumouwei/super-mario-bros-RL** (SB3)

### Hallazgo principal
El **batch_size=64** con **n_steps=512** y **n_epochs=10** producía **192 gradient steps** por update. La config estándar investigada usa **16 gradient steps**. Esa diferencia de 12x era la causa principal del catastrophic forgetting — cada update sobreescribía agresivamente la política.

### Cambios completos
| Parámetro | Antes | Después | Fuente |
|---|---|---|---|
| `action_space` | complex (12) | **simple (7)** | vietnh1009 (31/32 niveles) |
| `n_steps` | 512 | **128** | RL Zoo Atari, OpenAI Baselines |
| `batch_size` | 64 | **256** | RL Zoo Atari |
| `n_epochs` | 6 | **4** | RL Zoo, OpenAI, "37 Details" paper |
| `clip_range` | 0.2 fijo | **0.1 con decay lineal** | "37 Details" paper |
| `ent_coef` | 0.04 | **0.01** | Estándar Atari |
| `n_envs` | 4 | **8** | Más datos diversos por update |
| `total_timesteps` | 5M | **10M** | Margen para convergencia |
| `checkpoint_freq` | 50k | **100k** | Menos I/O |
| `death_by_enemy_penalty` | -10 | **-15** | Estándar en repos exitosos |
| Reward components | 9 activos | **solo 3** | Simplificar señal |
| `reset_memory` | false | **true** | Limpiar death_map/records de runs viejos |

### Componentes de reward simplificados
Desactivados: cyclic_detection, death_map, excessive_left, micro_movement, wall_stuck.
Activos: forward, stuck, records (+ base: coins, score, flag, death, time).

Razón: 17 componentes generaban señal ruidosa y conflictiva. Los repos exitosos usan 3-5 componentes.

### Gradient steps por update: antes vs después
```
ANTES: (512 × 4 envs) / 64 batch × 6 epochs = 192 gradient steps
AHORA: (128 × 8 envs) / 256 batch × 4 epochs = 16 gradient steps
```

### Implementación técnica
- `configs/schema.py` — nuevo campo `action_space` en EnvConfig, `use_linear_clip_schedule` en PPOConfig
- `models/ppo_factory.py` — decay lineal de clip_range via `get_linear_fn`
- `env/factory.py` — soporte para SIMPLE_MOVEMENT/COMPLEX_MOVEMENT configurable

### Verificación de checkpoints existentes
| Archivo | Acciones | Timesteps | Compatible con config nueva |
|---|---|---|---|
| `checkpoints/mario_ppo_10000.zip` | 12 (COMPLEX) | 10k | No |
| `checkpoints/mario_ppo_50000.zip` | 12 (COMPLEX) | 50k | No |
| `checkpoints/mario_ppo_100000.zip` | 12 (COMPLEX) | 100k | No |
| `checkpoints/mario_ppo_150000.zip` | 12 (COMPLEX) | 150k | No |
| `checkpoints/mario_ppo_200000.zip` | 12 (COMPLEX) | 200k | No |
| `checkpoints/mario_ppo_400000.zip` | 12 (COMPLEX) | 400k | No |
| `models_saved/mario_ppo.zip` | 7 (SIMPLE) | 736 | Sí pero inútil (apenas arrancó) |

Los checkpoints viejos (COMPLEX_MOVEMENT) no son compatibles con la config nueva (SIMPLE_MOVEMENT) — diferente tamaño de red neuronal. Sirven para demo con `action_space: complex`.

---

## Run 4 — Config investigada con ent_coef=0.01 (20 abril 2026)

### Resultado
A 176k steps: distancia avg100=271, max hist=1,193. Picos de distancia avg100 llegaban a ~500-600 pero no subían más. `ent_coef=0.01` no daba suficiente exploración para salir de la zona 0-400.

### Decisión
Subir `ent_coef` a 0.02 (punto medio entre 0.01 que no exploraba y 0.04 que oscilaba mucho).

---

## Run 5 — ent_coef=0.02 (20 abril 2026)

### Observaciones del CSV (logs/run_20260420_005543.csv)

**Primeros 50 episodios**: buenos, variedad de resultados, max_x desde 160 hasta 936. Distancia avg100 llegó a 430.

**Episodio ~50 en adelante**: colapso a política degenerada. El 90% de los episodios terminan con max_x=198 exacto, reward=385.5 exacto. El agente llega a x=198, se atascacontra un obstáculo, y el `no_progress_limit=120` lo mata.

**Patrón cíclico observado**:
- dist avg100 baja a 198 (todos los episodios iguales)
- Ocasionalmente un episodio alcanza 600-900 (por entropy)
- dist avg100 sube brevemente a 300-340
- Vuelve a colapsar a 198

A 51k steps: dist avg100=198, max hist=950. **Misma trampa de siempre.**

### Diagnóstico
El obstáculo en x~198 (probablemente primer goomba o pipe del nivel 1-2) crea un mínimo local fuerte. El agente recibe reward=385.5 por llegar ahí y morir — suficiente para que PPO refuerce esa política.

Problema adicional: `n_steps=128` era más corto que un episodio completo (~145 steps con truncamiento). El rollout no contenía episodios completos, lo que dificultaba el aprendizaje de las consecuencias de atascarse.

---

## Cambio 11 — Subir n_steps a 256

### Cambios
| Parámetro | Antes | Después | Razón |
|---|---|---|---|
| `ent_coef` | 0.01 | **0.02** | Más exploración para escapar mínimo local en x=198 |
| `n_steps` | 128 | **256** | Rollouts contengan episodios completos (~145 steps) |

### Gradient steps por update
`(256 × 8 envs) / 256 batch × 4 epochs = 32 gradient steps` (vs 16 antes, vs 192 originalmente).

### Bug fix: checkpoint timing
El callback de checkpoints usaba `self.n_calls` (una vez por step del VecEnv) en vez de `self.num_timesteps` (total timesteps). Con 8 envs, el checkpoint de "100k" se generaba a 800k timesteps reales. Corregido para usar `self.num_timesteps`.

### Limpieza
Checkpoints viejos COMPLEX_MOVEMENT eliminados (incompatibles con SIMPLE_MOVEMENT).

---

## Run 5 — ent_coef=0.02 + config investigada (20 abril 2026)

### Análisis del CSV (logs/run_20260420_005543.csv)
- 3532 episodios, 96k steps
- **82% de episodios terminaron en x=198 exacto** — política degenerada
- Patrón: colapsa a x=198, ocasionalmente escapa a 600-900 por entropy, vuelve a colapsar
- Distancia promedio global: 228
- Max x histórico: 972

### Diagnóstico
Hay un obstáculo específico en x~198 (posiblemente primer goomba o pipe del nivel 1-2). El agente aprende que llegar a x=198 da ~385 reward "seguro" y deja de explorar más allá. Con ent_coef=0.02 no tiene suficiente exploración para superar el obstáculo consistentemente.

---

## Cambio 12 — Reward escalado por distancia

### Problema
Llegar a x=198 (reward ~385) vs x=400 (reward ~785) — la diferencia no es suficiente para que PPO priorice las trayectorias largas sobre las cortas "seguras".

### Solución: forward_distance_scale
Nuevo parámetro que multiplica el forward_reward por `(1 + progress_ratio * scale)`. Cada pixel vale más cuanto más lejos está Mario:

```
x=198:  reward ~420   (scale 1.06x)
x=500:  reward ~1,150 (scale 1.15x) → 2.7x más que 198
x=1000: reward ~2,600 (scale 1.30x) → 6x más que 198
```

También:
- `record_distance_bonus`: 5 → **30** (fuerte incentivo por romper récords)
- Bonus territorio nuevo: 0.25 → **0.5**

### Archivos modificados
- `configs/schema.py` — nuevo campo `forward_distance_scale` en RewardConfig
- `env/reward_shaping.py` — forward reward multiplicado por distance_scale

---

## Run 6 — ent_coef=0.02 + distance scaling (20 abril 2026)

### Resultado
Colapsó incluso más rápido que el run 5. En 400 episodios:
- ep 1-200: dist_avg=131
- ep 201-400: dist_avg=**41** (peor que todos los runs anteriores)
- 90% de episodios con x < 200

### Análisis
La config "investigada" (batch=256, n_steps=128/256, clip=0.1, ent=0.01/0.02) no funciona para este environment específico. Producía consistentemente colapso temprano en todos los intentos (runs 4, 5, 6).

---

## Cambio 13 — Volver a lo que funcionó (20 abril 2026)

### Reflexión
Nuestro **mejor run fue el Run 3**: ent_coef=0.04, n_steps=512, batch=64, clip=0.2, COMPLEX_MOVEMENT, 4 envs. Llegó a dist_avg=638, reward=1278 a 206k steps. Las oscilaciones eran fuertes pero el agente progresaba.

Los cambios "investigados" (batch=256, clip=0.1, ent=0.01) causaban colapso temprano porque:
1. **batch=256 + n_epochs=4 = solo 32 gradient steps** → no aprendía lo suficiente de cada rollout
2. **clip=0.1** → demasiado restrictivo, no dejaba que la política cambiara lo suficiente
3. **ent=0.01** → insuficiente exploración para este nivel

### Config de compromiso: lo mejor del Run 3 + mejoras confirmadas

| Parámetro | Run 3 (mejor) | Config investigada | **Compromiso** |
|---|---|---|---|
| n_steps | 512 | 128 | **512** (rollouts completos) |
| batch_size | 64 | 256 | **64** (128 grad steps, suficiente aprendizaje) |
| n_epochs | 10 | 4 | **4** (menos catastrophic forgetting) |
| clip_range | 0.2 | 0.1 | **0.2** (más libertad de cambio) |
| ent_coef | 0.04 | 0.01 | **0.04** (el que mejor exploró) |
| action_space | complex | simple | **simple** (menos acciones) |
| n_envs | 4 | 8 | **8** (más velocidad) |
| distance_scale | no existía | 1.0 | **1.0** (incentiva ir lejos) |
| record_bonus | 5 | 5 | **30** (incentiva romper récords) |
| death_penalty | -10 | -15 | **-10** (menos miedo a morir) |

**128 gradient steps por update** (vs 32 de la investigada, vs 192 del original con n_epochs=10).

### Decisión
Dejar correr **mínimo 1 hora** sin tocar. Los runs cortos (50-200k steps) no son suficientes para juzgar — el agente necesita tiempo para superar las oscilaciones iniciales.

---

## Config actual (post cambio 13)

```yaml
env:
  action_space: simple

ppo:
  n_steps: 512
  batch_size: 64
  n_epochs: 4
  clip_range: 0.2
  ent_coef: 0.04
  use_linear_clip_schedule: false

reward:
  forward_reward_coef: 2.0
  forward_distance_scale: 1.0
  death_by_enemy_penalty: -10.0
  death_by_time_penalty: -5.0
  flag_bonus: 30.0
  record_distance_bonus: 30.0
  no_progress_limit: 120
  # Desactivados: cyclic, death_map, excessive_left, micro, wall

training:
  n_envs: 8
  total_timesteps: 10000000
```

**128 gradient steps por update.**

---

## Resumen de todos los runs

| Run | Config clave | Mejor dist avg100 | Max x hist | Problema |
|---|---|---|---|---|
| 1 | ent=0.02, epochs=10, 1 env | 207 | 1,339 | Lento (65 sps), rewards -7000 |
| 2 | ent=0.02, epochs=10, 4 envs | 392 | 1,542 | Oscilaciones, estancado a 400k |
| **3** | **ent=0.04, epochs=10, 4 envs** | **638** | **1,934** | Oscilaciones pero progresaba |
| 4 | ent=0.01, epochs=4, 8 envs, SIMPLE | 271 | 1,193 | Picos no subían |
| 5 | ent=0.02, epochs=4, batch=256 | 332 | 950 | 82% episodios en x=198 |
| 6 | ent=0.02 + distance scale | 131 | 856 | Colapso rápido a x=41 |
| 7 | **compromiso (actual)** | ? | ? | En curso... |

---

## Próximos pasos pendientes

- **Dejar correr 1 hora mínimo sin tocar**
- **500k steps**: dist avg100 > 500, max hist subiendo
- **1M steps**: dist avg100 > 800
- **2M steps**: primer clear
- **10M steps**: resultado final
