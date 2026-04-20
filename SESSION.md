# Sesión del 2026-04-19 — resumen

## Lo que se logró

Proyecto Mario RL refactor completo **funcionando end-to-end en WSL Ubuntu 24.04** con CUDA.

### Validado

- ✅ 27/27 tests unitarios pasan.
- ✅ CUDA detecta RTX 3060 Laptop dentro de WSL.
- ✅ `nes-py` compila sin problema con `gcc` (ya venía instalado en Ubuntu 24.04).
- ✅ Entrenamiento real funciona: probado con `fast_debug.yaml`, Mario llegó a x=678 en los primeros 7,500 steps con reward_avg100=220.
- ✅ Terminal app con barra de progreso horizontal live.
- ✅ Dashboard FastAPI importable y servible.

## Stack final

| Componente | Versión / Detalle |
|---|---|
| Distro | WSL Ubuntu 24.04 (`wsl -d Ubuntu`) |
| Python | 3.12.3 |
| torch | 2.5.1+cu121 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| stable-baselines3 | 2.1.0 |
| gym | 0.26.2 + shimmy 1.3.0 + gymnasium 0.29.1 |
| nes-py | 8.2.1 (compilado local con gcc) |
| gym-super-mario-bros | 7.4.0 |
| numpy | <2 (fijado por compatibilidad con gym) |

## Cambios hechos en esta sesión

### Decisiones arquitectónicas

1. **Migración Python 3.13 Windows → Python 3.12 Ubuntu 24.04 WSL.**
   - Razón: `nes-py` requiere compilación C++. En Windows necesitaría Visual Studio Build Tools (que fallaron repetidamente al instalar). En Linux, `gcc` default compila sin fricción.
2. **Distro elegida: "Ubuntu" (24.04)**, no "Ubuntu-22.04" — el usuario reporta errores en 22.04.
3. **Proyecto canónico vive en `~/projects/super-mario/` dentro de WSL.** Carpeta Windows es solo espejo editable.

### Bugs fixeados

Además de los 6 bugs originales del código legacy:

- **gym 0.26 API mismatch**: `env.step()` cambió de 4-tupla a 5-tupla, `env.reset()` de obs a `(obs, info)`. Adapté:
  - `env/reward_shaping.py`: maneja ambas APIs, devuelve 5-tupla.
  - `env/wrappers_obs.py::SkipFrame`: idem.
  - `tests/conftest.py::DummyMarioEnv`: API nueva.
  - `tests/test_reward_shaping.py`: desempaquetados actualizados.
  - `training/trainer.py::evaluate`: maneja ambos formatos.
- **Canal gris extra**: GrayScaleObservation+ResizeObservation producía `(H, W, 1)` en vez de `(H, W)`, lo que hacía que FrameStack diera `(4, 84, 84, 1)` incompatible con SB3 CnnPolicy. Fix: nuevo `SqueezeChannelObservation` wrapper en `env/wrappers_obs.py`.
- **JoypadSpace no acepta kwargs**: `reset(seed=...)` de SB3 caía en `TypeError`. Fix: try/except en `AdvancedRewardWrapper.reset`.
- **apply_api_compatibility**: cambiado a `True` en `env/factory.py` para usar API moderna.

### Deps extra agregadas durante la sesión

- `shimmy==1.3.0` (puente gym ↔ gymnasium requerido por SB3 2.1.0).
- `gymnasium==0.29.1` (compatible con SB3).

## Documentos generados

- `CLAUDE.md` — contexto persistente del proyecto (para futuras sesiones).
- `SESSION.md` — este archivo.
- `docs/MANUAL.md` — manual de uso completo.
- `docs/SETUP.md` — setup paso a paso (ahora orientado a WSL Ubuntu).
- `docs/COLAB.md` — fallback cloud.
- `notebooks/train_colab.ipynb` — notebook Colab completo.
- 27 tests en `tests/`.

## Comandos útiles para próxima sesión

### Sincronizar cambios desde Windows a WSL

```powershell
wsl -d Ubuntu -- bash -c "rsync -a --exclude='venv' --exclude='__pycache__' --exclude='.pytest_cache' '/mnt/c/Users/Jose Ramon/OneDrive/Desktop/super-mario/' ~/projects/super-mario/"
```

### Correr tests

```powershell
wsl -d Ubuntu -- bash -c "cd ~/projects/super-mario && source venv/bin/activate && pytest tests/"
```

### Entrenar (debug rápido)

```powershell
wsl -d Ubuntu -- bash -c "cd ~/projects/super-mario && source venv/bin/activate && python app.py --config configs/fast_debug.yaml --train"
```

### Entrenar (real, 5M steps)

```powershell
wsl -d Ubuntu -- bash -c "cd ~/projects/super-mario && source venv/bin/activate && python app.py --train"
```

### Dashboard web

```powershell
wsl -d Ubuntu -- bash -c "cd ~/projects/super-mario && source venv/bin/activate && uvicorn ui.server:app --host 127.0.0.1 --port 8000"
```

Luego abrir `http://127.0.0.1:8000/` en el navegador de Windows (WSL expone puertos a Windows automáticamente).

## Pendientes / siguientes pasos

1. **Training completo** — correr `python app.py --train` por 3.5-5.5 h con `configs/default.yaml` para 5M steps.
2. **Monitorear con TensorBoard** — `tensorboard --logdir logs`.
3. **Validar dashboard web en browser** — comprobar que el frame stream llega, las trayectorias se dibujan, la edición de reward en caliente funciona.
4. **Opcional: `wsl --set-default Ubuntu`** para que al abrir WSL caiga directo en Ubuntu 24.04.
5. **Opcional: instalar extensión "WSL" de VS Code** para que `code .` desde WSL funcione.

## Temas que quedaron abiertos

- El usuario intentó `code .` desde WSL y le dio "Exec format error". Causa: falta la extensión "WSL" de VS Code en Windows. Solución: abrir VS Code en Windows → Extensions → buscar "WSL" → instalar. Después `code .` en WSL abrirá VS Code con el backend remoto.
- La distro Ubuntu-22.04 se limpió del proyecto (`~/projects/super-mario` ahí fue borrado). Queda el venv y la instalación — si el usuario quiere puede `wsl --unregister Ubuntu-22.04` pero no es urgente.
- La carpeta Windows `venv_py313_backup/` puede eliminarse para ahorrar espacio.
