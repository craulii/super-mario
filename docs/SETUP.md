# Setup — paso a paso (WSL Ubuntu 24.04)

Guía lineal. El proyecto corre sobre **WSL Ubuntu 24.04** con **Python 3.12** y **CUDA 12.1**.

> **¿Por qué WSL y no Windows nativo?** `nes-py` (el emulador del NES) requiere compilación C++. En Linux `gcc` viene con `build-essential`; en Windows hay que instalar Visual Studio Build Tools (~4 GB + conflictos frecuentes con el redist). WSL evita todo ese problema.

---

## Estado actual del setup

Ya está hecho. Solo te queda usarlo:

- ✅ WSL Ubuntu 24.04 con gcc/cmake instalados.
- ✅ venv en `~/projects/super-mario/venv` con Python 3.12.
- ✅ torch 2.5.1+cu121 — detecta tu RTX 3060 Laptop.
- ✅ stable-baselines3, gym, gymnasium, shimmy, nes-py, gym-super-mario-bros.
- ✅ fastapi, uvicorn, rich, pytest, tensorboard.
- ✅ 27/27 tests pasan.
- ✅ Entrenamiento `fast_debug` validado.

---

## Uso diario

### 0. Abrir WSL en la distro correcta

Abrí Windows Terminal y elegí **Ubuntu** (24.04) del menú desplegable. O desde PowerShell:

```powershell
wsl -d Ubuntu
```

Una vez dentro:

```bash
cd ~/projects/super-mario
source venv/bin/activate
```

### 1. Correr tests

```bash
pytest tests/
```

Esperado: `27 passed`.

### 2. Entrenamiento debug (~2 min)

```bash
python app.py --config configs/fast_debug.yaml --train
```

50k steps, validación rápida del pipeline. Barra horizontal muestra el progreso en el nivel. Ctrl+C corta limpio.

### 3. Entrenamiento real (3.5-5.5 h con tu GPU)

```bash
python app.py --train
```

5M steps. Deja la terminal abierta. Modelos se guardan en `models_saved/`.

### 4. Dashboard web

Terminal A:
```bash
uvicorn ui.server:app --host 127.0.0.1 --port 8000
```

En tu **navegador de Windows**: `http://127.0.0.1:8000/`  
(WSL expone puertos a Windows automáticamente.)

### 5. TensorBoard (opcional)

Terminal B:
```bash
tensorboard --logdir logs
```

Navegador: `http://localhost:6006`.

### 6. Evaluar un modelo entrenado

```bash
python app.py --evaluate --model-path models_saved/mario_ppo --episodes 5
```

---

## Si necesitás reinstalar todo desde cero (referencia)

```bash
# Dentro de Ubuntu 24.04 WSL:
sudo apt update && sudo apt install -y build-essential python3.12-venv python3.12-dev python3-pip cmake

cd ~/projects
rsync -a --exclude='venv' --exclude='__pycache__' '/mnt/c/Users/Jose Ramon/OneDrive/Desktop/super-mario/' super-mario/
cd super-mario

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools

# PyTorch con CUDA (~2.5 GB)
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1

# Runtime + dashboard + tests
pip install "numpy<2" gym==0.26.2 gymnasium==0.29.1 stable-baselines3==2.1.0 shimmy==1.3.0 \
            "opencv-python==4.10.0.84" rich "fastapi>=0.110.0" "uvicorn[standard]" websockets \
            pyyaml "pydantic>=2.6.0" pillow pytest pytest-asyncio httpx tensorboard

# Mario (compila con gcc automáticamente)
pip install nes-py==8.2.1 gym-super-mario-bros==7.4.0

# Verificar
pytest tests/
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Sincronizar cambios desde Windows a WSL

Si editás archivos desde Windows (VS Code, notepad++, etc.), sincronizá a WSL antes de correr:

**Desde PowerShell (Windows):**
```powershell
wsl -d Ubuntu -- bash -c "rsync -a --exclude='venv' --exclude='__pycache__' --exclude='.pytest_cache' '/mnt/c/Users/Jose Ramon/OneDrive/Desktop/super-mario/' ~/projects/super-mario/"
```

**Desde WSL:**
```bash
rsync -a --exclude='venv' --exclude='__pycache__' /mnt/c/Users/Jose\ Ramon/OneDrive/Desktop/super-mario/ ~/projects/super-mario/
```

---

## VS Code desde WSL (opcional pero cómodo)

Si `code .` desde WSL te da *"Exec format error"*:

1. En **Windows**, abrí VS Code.
2. Extensions (Ctrl+Shift+X) → buscá **"WSL"** (de Microsoft) → Install.
3. Cerrá VS Code, abrilo de nuevo.
4. En tu WSL: `cd ~/projects/super-mario && code .`.

VS Code se abre con indicador `[WSL: Ubuntu]` abajo a la izquierda y edita los archivos Linux directamente, con terminal integrada que ya es bash.

---

## Diagnósticos rápidos

| Chequeo | Comando | Esperado |
|---|---|---|
| Distro | `lsb_release -a` | Ubuntu 24.04 |
| Python | `python --version` (dentro del venv) | Python 3.12.3 |
| CUDA | `python -c "import torch; print(torch.cuda.is_available())"` | `True` |
| GPU | `python -c "import torch; print(torch.cuda.get_device_name(0))"` | `NVIDIA GeForce RTX 3060 Laptop GPU` |
| Mario env | `python -c "import gym_super_mario_bros; print('OK')"` | `OK` |
| Tests | `pytest tests/` | `27 passed` |

---

## Tiempos de entrenamiento (RTX 3060 Laptop)

- Debug (50k steps): **~1-2 min**
- Primera señal de mejora (~500k steps): ~15-25 min
- Clear ocasional (~2M steps): ~1-1.5 h
- Clear consistente (~4M steps): ~2.5-4 h
- Total (5M steps): **~3.5-5.5 h**

---

## Problemas comunes

| Problema | Solución |
|---|---|
| `code .` → Exec format error | Instalar extensión **WSL** en VS Code. |
| `No module named nes_py` | Activaste el venv? `source venv/bin/activate`. |
| CUDA out of memory | Bajá `ppo.batch_size` a 32 en el YAML. |
| `apt-get update` 404 | `sudo apt-get update` (con sudo) primero. |
| Distro equivocada al abrir | `wsl --set-default Ubuntu`. |
| Abrir el proyecto desde Windows | Explorador: `\\wsl$\Ubuntu\home\crauli\projects\super-mario\`. |
