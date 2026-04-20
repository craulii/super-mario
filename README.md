# Mario RL — Super Mario Bros 1-2

Agente PPO que aprende a completar el nivel 1-2 con reward shaping avanzado (17 componentes), terminal interactiva y dashboard web local con edición de parámetros en caliente.

**Requiere Python 3.11** (nes-py no tiene wheels para 3.12/3.13).

## Documentación

- [`docs/SETUP.md`](docs/SETUP.md) — **setup completo paso a paso** (venv, CUDA, instalación, verificación).
- [`docs/MANUAL.md`](docs/MANUAL.md) — manual de uso, hiperparámetros, troubleshooting.

## Inicio rápido

```powershell
cd "C:\Users\Jose Ramon\OneDrive\Desktop\super-mario"
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip wheel setuptools

# PyTorch con CUDA (para tu RTX 3060 Laptop)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Resto
pip install "numpy==1.26.4" gym==0.26.2 stable-baselines3==2.1.0 "opencv-python==4.10.0.84"
pip install rich "fastapi>=0.110.0" "uvicorn[standard]" websockets pyyaml "pydantic>=2.6.0" pillow pytest pytest-asyncio httpx tensorboard

# Mario (requiere Visual C++ Build Tools — ver SETUP.md paso 0)
pip install nes-py==8.2.1 gym-super-mario-bros==7.4.0
```

Verificar:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest tests/ -v                # debe dar: 27 passed
```

## Uso

### Terminal

```powershell
venv\Scripts\activate
python app.py                                       # menú interactivo
python app.py --train                               # 5M steps con configs/default.yaml
python app.py --config configs/fast_debug.yaml --train   # debug 50k steps
python app.py --evaluate --model-path models_saved/mario_ppo --episodes 5
```

### Dashboard web

```powershell
venv\Scripts\activate
uvicorn ui.server:app --host 127.0.0.1 --port 8000
```

Abrir `http://127.0.0.1:8000/` — control de training, stream del juego en vivo, edición de reward shaping en caliente, mapa de trayectorias acumuladas.

### TensorBoard

```powershell
venv\Scripts\activate
tensorboard --logdir logs
```

## Estructura

- `app.py` — terminal app unificada (rich)
- `configs/` — YAMLs: `default.yaml`, `fast_debug.yaml`, schema tipado
- `env/` — wrappers + reward shaping (9 componentes nuevos)
- `models/` — save/load con metadata JSON + factory PPO
- `training/` — Trainer con threading + SharedState + 3 callbacks
- `ui/` — dashboard FastAPI + 3 WebSockets + Alpine.js + Chart.js
- `utils/` — ring buffer, seed, progress bar, frame encode, logging
- `tests/` — 27 tests unitarios
- `docs/` — `MANUAL.md`, `SETUP.md`
- `legacy/` — versión anterior del código (referencia)

## Tiempos de entrenamiento

En tu HP Omen 16 con RTX 3060 Laptop:

- Debug (50k steps): ~1-2 min
- Clear ocasional (~2M steps): ~1-1.5 h
- Clear consistente (~4M steps): ~2.5-4 h
- Total (5M steps): ~3.5-5.5 h
