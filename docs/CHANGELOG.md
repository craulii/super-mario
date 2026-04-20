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

## Config actual (post cambio 8)

```yaml
ppo:
  ent_coef: 0.04
  n_epochs: 4              # bajado de 10

reward:
  forward_reward_coef: 2.0
  flag_bonus: 30.0
  death_by_enemy_penalty: -10.0
  death_by_time_penalty: -5.0
  stuck_penalty_base: -0.2
  wall_stuck_penalty: -0.1
  micro_movement_penalty: -0.3
  excessive_left_penalty: -0.05
  enable_cyclic_detection: false   # desactivado
  enable_death_map: false
  no_progress_limit: 120           # bajado de 200

training:
  n_envs: 4
  total_timesteps: 5000000
```

---

## Próximos pasos pendientes

Ver `docs/GUIA_AJUSTES.md` para qué hacer en cada checkpoint:
- **500k steps**: distancia avg100 > 500, gráficos subiendo → no tocar
- **1M steps**: distancia avg100 > 800, max hist > 2500 → no tocar
- **2M steps**: primer clear (clear rate > 0%) → subir flag_bonus si se acerca pero no completa
- **3-4M steps**: clear rate > 50% → optimizar para puntaje (coins, time bonus)
- **5M steps**: evaluar modelo final con 20 episodios
