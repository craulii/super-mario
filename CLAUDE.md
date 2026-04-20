# Mario RL — Contexto para Claude

Proyecto de RL para competencia universitaria. Agente PPO que aprende a completar Super Mario Bros 1-2.

## Ubicación canónica del código

**Ejecutar SIEMPRE desde Ubuntu 24.04 (WSL):** `~/projects/super-mario/`

Esta carpeta de Windows (`C:\Users\Jose Ramon\OneDrive\Desktop\super-mario`) es solo el mirror para editar desde Windows. Los archivos **se ejecutan en WSL** porque `nes-py` requiere compilación con `gcc`, que en Windows exige Visual Studio Build Tools (no instalados en la máquina del usuario).

## Cómo sincronizar cambios Windows → WSL

```powershell
wsl -d Ubuntu -- bash -c "rsync -a --exclude='venv' --exclude='venv_py313_backup' --exclude='__pycache__' --exclude='.pytest_cache' '/mnt/c/Users/Jose Ramon/OneDrive/Desktop/super-mario/' ~/projects/super-mario/"
```

## Cómo ejecutar en WSL

```bash
wsl -d Ubuntu -- bash -c "cd ~/projects/super-mario && source venv/bin/activate && <comando>"
```

Ejemplos:
- Tests: `... && pytest tests/`
- Training debug: `... && python app.py --config configs/fast_debug.yaml --train`
- Training real: `... && python app.py --train`
- Dashboard: `... && uvicorn ui.server:app --host 127.0.0.1 --port 8000`

## Stack instalado en Ubuntu 24.04 WSL

| Paquete | Versión |
|---|---|
| Python | 3.12.3 |
| torch | 2.5.1+cu121 (**CUDA ON → RTX 3060 Laptop**) |
| stable-baselines3 | 2.1.0 |
| gym | 0.26.2 |
| gymnasium | 0.29.1 |
| shimmy | 1.3.0 |
| nes-py | 8.2.1 (compilado con gcc local) |
| gym-super-mario-bros | 7.4.0 |
| numpy | 1.26.x (fijado <2 por compatibilidad) |
| fastapi, uvicorn, websockets | deps dashboard |
| rich | TUI terminal |
| pytest | testing |

## Reglas al modificar el código

1. **Editar archivos en Windows** (es el wd de Claude Code).
2. **Sincronizar a WSL** con rsync (comando arriba) antes de correr tests/training.
3. **Ejecutar en WSL** siempre.
4. No tocar los archivos de `legacy/` — son solo referencia.

## Estructura del proyecto

```
super-mario/
├── app.py                  # terminal app (rich)
├── configs/                # YAMLs + schema tipado (default.yaml, fast_debug.yaml)
├── env/                    # wrappers + reward shaping avanzado (17 componentes)
├── models/                 # save/load con metadata + factory PPO
├── training/               # Trainer (thread) + SharedState + callbacks
├── ui/                     # dashboard FastAPI + WebSockets + Alpine.js
├── utils/                  # ring_buffer, seed, progress_bar, frame_encode
├── tests/                  # 27 unit tests
├── docs/                   # MANUAL.md, SETUP.md, COLAB.md
├── notebooks/              # train_colab.ipynb (fallback cloud)
├── legacy/                 # código anterior (5 .py originales)
├── checkpoints/            # runtime
├── models_saved/           # runtime
├── logs/                   # TensorBoard runtime
└── videos/                 # runtime
```

## Bugs corregidos en la migración

1. `train.py::resume_path` logic dead branch.
2. `config.py::resize_shape` `int` → tupla para `ResizeObservation`.
3. `wrappers.py::life<2` → flanco descendente `life<prev_life`.
4. `callback.py::[-100:]` slicing O(n) → `RingBuffer` O(1).
5. Comentario `SIMPLE_MOVEMENT` vs código `COMPLEX_MOVEMENT`.
6. README desincronizado (decía 2M steps, config tenía 5M).
7. gym 0.26 API mismatch entre wrappers custom (4-tupla) y `gym.wrappers` (5-tupla + reset tupla 2).
8. Canal extra `(84,84,1)` del GrayScale → `SqueezeChannelObservation` antes de FrameStack.
9. `JoypadSpace.reset()` no acepta kwargs → try/except TypeError en `AdvancedRewardWrapper.reset`.

## Reward shaping — 17 componentes

Base (8): forward, coin, score, vertical_explore, flag, time_penalty, stuck, backward.
Nuevos (9): muerte_por_tiempo vs enemigo, patrones cíclicos, mapa de muertes persistente, retroceso excesivo, micro-movimientos, stuck adaptativo, muro, récord distancia, récord tiempo.

Todos en `env/reward_shaping.py` via `RewardConfig` mutable en caliente (referenciado desde el wrapper).

## Concurrencia

`SharedState` con `threading.Lock` + dos `threading.Event` (pause, resume) + `stop_event`.

El `AdvancedRewardWrapper` guarda **referencia** al `RewardConfig` — mutar in-place desde el dashboard propaga cambios al próximo `env.step()` sin reiniciar training.

## Hardware target

HP Omen 16 con **RTX 3060 Laptop GPU** (6 GB VRAM). CUDA 12.1. Tiempo esperado 5M steps: 3.5-5.5 h.

## Documentación

- `docs/MANUAL.md` — uso completo, hiperparámetros, troubleshooting.
- `docs/SETUP.md` — setup paso a paso.
- `docs/COLAB.md` — fallback en Google Colab.
- `SESSION.md` — historial de la sesión más reciente.
