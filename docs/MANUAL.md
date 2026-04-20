# Manual de uso — Mario RL

Agente PPO que aprende a completar Super Mario Bros **nivel 1-2**, con terminal interactiva, dashboard web y reward shaping avanzado.

---

## 1. Requisitos

- Windows, Linux o macOS.
- **Python 3.10 o 3.11 recomendado.** `nes-py` (motor del ROM) no tiene wheels para Python 3.13, requiere compilador C si se fuerza.
- Opcional: GPU NVIDIA con CUDA (ver sección 9).

## 2. Instalación

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# Linux:    source venv/bin/activate
pip install -r requirements.txt
```

Si tenés GPU NVIDIA (recomendado), instalá PyTorch con CUDA **antes** del requirements:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Problemas comunes

| Error | Solución |
|---|---|
| `nes-py` no compila | Usá Python 3.10/3.11. En Windows, instalá "Visual Studio Build Tools (C++)" si igual querés 3.13. |
| `No module named gym_super_mario_bros` | `pip install gym-super-mario-bros==7.4.0` |
| `CUDA out of memory` | Bajá `batch_size` en el YAML o pasá a `device: cpu`. |

---

## 3. Estructura del proyecto

```
super-mario/
├── app.py                 # entrada: terminal interactiva
├── configs/               # YAMLs de configuración
├── env/                   # wrappers + reward shaping
├── models/                # save/load + factory PPO
├── training/              # Trainer (thread) + callbacks + SharedState
├── ui/                    # dashboard FastAPI + estáticos
├── utils/                 # buffers, seeds, helpers
├── tests/                 # tests unitarios
├── docs/                  # este manual
└── legacy/                # versión anterior del código (referencia)
```

---

## 4. Terminal app

Menú interactivo:

```bash
python app.py
```

Opciones:

1. **Entrenar desde cero** — usa `configs/default.yaml`.
2. **Reanudar entrenamiento** — pide la ruta del `.zip` del modelo previo.
3. **Evaluar** — carga un modelo y corre N episodios.
4. **Guardar** — solo permitido en pausa o idle.
5. **Cargar** — solo cuando trainer está detenido.
6. **Ver estado** — snapshot actual.
7. **Salir**.

Durante el entrenamiento/evaluación se muestra una barra horizontal de progreso del nivel:

```
[████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 30.2%  x=1015/3360
```

Atajos CLI:

```bash
python app.py --train                                    # arranca training
python app.py --config configs/fast_debug.yaml --train   # debug rápido (50k steps)
python app.py --evaluate --model-path models_saved/mario_ppo --episodes 10
python app.py --train --timesteps 1000000                # override de timesteps
```

**Ctrl+C** detiene limpio y guarda un checkpoint de emergencia en `checkpoints/interrupted.zip`.

---

## 5. Dashboard web

Levantar servidor:

```bash
uvicorn ui.server:app --host 127.0.0.1 --port 8000
```

Abrir `http://127.0.0.1:8000/` en el navegador.

### Secciones

- **Control** — iniciar / pausar / reanudar / detener entrenamiento. Guardar / cargar modelo.
- **Juego en vivo** — stream del NES a ~10 Hz + barra de progreso del nivel.
- **Reward config** — todos los coeficientes editables. Click *Aplicar en caliente* propaga los cambios al próximo `env.step()` sin reiniciar.
- **Mapa del nivel** — todas las trayectorias de la sesión superpuestas, con color por episodio y alpha degradada por antigüedad.
- **Métricas** — curvas de `reward_avg_100` y `clear_rate_100`.

### Qué se puede editar en caliente

Todo lo de la sección `reward:` del YAML. Los coeficientes y flags de penalización/recompensa se aplican de inmediato.

### Qué requiere reiniciar

- Cualquier campo de `ppo:` (learning_rate, n_steps, batch_size, etc.)
- `env_id`, `frame_stack`, `resize_shape`
- `total_timesteps`, `seed`, `device`

Para cambiarlos: detené el trainer, editá el YAML, y volvé a iniciar.

---

## 6. Entrenar — qué esperar

### Tiempos aproximados (5M timesteps)

Basado en PPO con 1 entorno + CnnPolicy.

| Hardware | Steps/s | 1M steps | 5M steps | Clear consistente |
|---|---|---|---|---|
| CPU (i5 mid) | 60-90 | 3-5 h | 15-25 h | difícil; posible con reward shaping fino |
| **GPU media (GTX 1660 / RTX 3060)** | 250-400 | 40-70 min | **3.5-5.5 h** | **2-4M steps** |
| GPU alta (RTX 4070/4080) | 500-800 | 20-35 min | 1.5-3 h | 1.5-3M steps |

**Nota:** PPO con 1 env está limitado por la emulación NES, no por la GPU. Escalar a múltiples envs aceleraría ~3-6× pero requiere rediseñar la sincronización del dashboard (no incluido en esta versión).

### Cómo saber que está mejorando

1. **`clear_rate_100`** — ratio de episodios completados en los últimos 100. Debe subir desde 0 hasta ≥ 0.8.
2. **`avg_distance_100`** — distancia máxima por episodio, promedio 100. Debe subir monótonamente.
3. **`reward_avg_100`** — reward total promedio.
4. **`progress_ratio`** — `x / 3360`. Si alcanza 1.0 de forma estable, el agente aprendió.
5. **`behavior_score`** — suma de penalizaciones ineficientes. **Debe bajar** a medida que avanza el entrenamiento.

En TensorBoard:

```bash
tensorboard --logdir logs
```

### Qué hiperparámetros tocar

| Síntoma | Acción |
|---|---|
| La loss oscila | Bajá `learning_rate` (ej: 2.5e-4 → 1e-4). |
| Se estanca temprano, explora poco | Subí `ent_coef` (0.02 → 0.05). |
| Llega cerca del final pero nunca al flag | Subí `flag_bonus` (15 → 30) o `time_remaining_bonus_coef`. |
| Se queda quieto mucho | Más `stuck_penalty_base` (más negativo). |
| Repite patrones inútiles | Activá `enable_cyclic_detection`. |

---

## 7. Reward shaping — componentes

Los 9 componentes nuevos (sobre la base de forward / coins / score / flag / death):

| Componente | Config | Efecto |
|---|---|---|
| Muerte por tiempo vs enemigo | `death_by_time_penalty` < `death_by_enemy_penalty` | Penaliza menos la muerte por exploración. |
| Stuck adaptativo | `stuck_threshold_base` | Tolera más stuck early-game, menos late-game. |
| Patrones cíclicos | `enable_cyclic_detection` + `cyclic_*` | Detecta y penaliza bucles de acciones sin progreso. |
| Mapa de muertes | `enable_death_map` + `death_bucket_*` | Penaliza morir en zonas donde ya murió antes. Persiste en `checkpoints/death_map.json`. |
| Retroceso excesivo | `enable_excessive_left` | Tolera retroceso breve, penaliza ≥30 frames hacia la izquierda. |
| Micro-movimientos | `enable_micro_movement` | Ventana de 60 frames: si no hay progreso neto, penaliza. |
| Muro | `enable_wall_stuck` | Detecta acción right sin cambio de X. |
| Récord histórico | `enable_records` | Bonus one-shot al superar mejor distancia / mejor tiempo. |

Memoria persistente (`death_map.json`, `best_distance.json`, `best_clear_time.json`) se puede resetear con `training.reset_memory: true` en el YAML.

---

## 8. Tests

```bash
pytest tests/ -v
```

Cubre: ring_buffer, action_history, death_map, shared_state (incl. concurrencia), reward_shaping (cada componente).

---

## 9. CUDA — ¿qué es y conviene?

**CUDA** es la plataforma de NVIDIA que permite que PyTorch corra los cálculos de la red neuronal en la GPU en lugar de en la CPU. Para un HP Omen 16 con GPU RTX, **sí conviene**: acelera el forward/backward de PPO entre ~4× y ~8× en este proyecto.

### Cómo instalarlo

1. Fijate qué RTX tenés (en el Administrador de dispositivos o con `nvidia-smi` en terminal).
2. Instalá drivers actualizados desde nvidia.com.
3. Instalá PyTorch con CUDA:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
4. Verificá:
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Debe imprimir `True RTX 3060` (o lo que tengas).

El proyecto detecta CUDA automáticamente (`device: cuda` si está disponible). Sin cambios en el YAML.

### ¿Todo va mejor?

En este proyecto, la GPU acelera el *entrenamiento de la red neuronal*, pero el emulador NES corre en CPU y es el cuello de botella principal con 1 solo env. Con CUDA podés esperar:

- ~250-400 steps/s en tu Omen vs ~60-90 steps/s en CPU pura.
- Entrenamiento de 5M steps en 3-5 h en lugar de 15-25 h.
- Para exprimir más, el siguiente paso sería usar `SubprocVecEnv` con 4-8 envs en paralelo (no incluido en esta versión).

---

## 10. Flujo recomendado para la competencia

1. **Validar pipeline**: `python app.py --config configs/fast_debug.yaml --train`. Debe completar 50k steps sin errores.
2. **Entrenar**: `python app.py --train` con `default.yaml`. Dejá 4-5 h.
3. **Monitorear**: `uvicorn ui.server:app` + `tensorboard --logdir logs`.
4. **Ajustar en caliente**: si se estanca, desde el dashboard subí `ent_coef` (requiere restart) o tocá los coeficientes de reward (en caliente).
5. **Guardar mejor modelo**: pausá → guardar con nombre descriptivo → reanudar.
6. **Evaluar final**: `python app.py --evaluate --model-path models_saved/mario_ppo --episodes 10`.

---

## 11. Referencias rápidas

- Código viejo (como referencia): `legacy/`.
- Config base: `configs/default.yaml`.
- Config de debug rápido: `configs/fast_debug.yaml`.
- Trayectorias persistentes: `checkpoints/death_map.json`, `best_distance.json`, `best_clear_time.json`.
- Modelos guardados: `models_saved/mario_ppo.zip` + `.meta.json`.
- Logs de TensorBoard: `logs/`.
