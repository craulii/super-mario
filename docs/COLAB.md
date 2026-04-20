# Entrenar en Google Colab — guía rápida

**Funciona en 5 min, sin tocar tu Windows, con GPU gratis.**

## Pasos

### 1. Comprimir tu proyecto

Desde PowerShell:

```powershell
cd "C:\Users\Jose Ramon\OneDrive\Desktop"
Compress-Archive -Path "super-mario\configs","super-mario\env","super-mario\models","super-mario\training","super-mario\ui","super-mario\utils","super-mario\tests","super-mario\app.py","super-mario\requirements.txt" -DestinationPath super-mario.zip -Force
```

Esto crea `super-mario.zip` sin incluir `venv/` ni `legacy/`.

### 2. Abrir Colab

1. Ir a https://colab.research.google.com/
2. **Archivo → Subir notebook** → elegir `notebooks/train_colab.ipynb` de tu proyecto.
3. **Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)** → Guardar.

### 3. Subir el zip

En el panel izquierdo, icono de carpeta, click en "Subir al almacenamiento de sesión", seleccionar `super-mario.zip`.

### 4. Ejecutar celdas

Ctrl+Enter en cada celda, de arriba hacia abajo. La primera verifica GPU; la segunda instala deps (~2 min); la cuarta entrena debug (~2 min); la quinta entrena 5M steps (~1-2 h en T4).

### 5. Descargar modelo entrenado

La última celda comprime `models_saved/` + `logs/` y dispara la descarga automáticamente.

### 6. Usar el modelo localmente

Cuando fixees Build Tools + nes-py en local, vas a poder:

```powershell
python app.py --evaluate --model-path models_saved/mario_ppo --episodes 5
```

Mientras tanto, el modelo trainreado vive en el `.zip` descargado — te sirve para presentar resultados de la competencia.

## Límites de Colab gratis

- Sesión se cierra tras ~90 min de inactividad → usá el notebook sin cerrar la pestaña.
- GPU T4 disponible casi siempre; ocasionalmente cae a CPU (sigue entrenando, más lento).
- Límite diario de ~12 h de GPU. Suficiente para varios runs.
