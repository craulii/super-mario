# Guía de ajustes — Qué hacer cuando el agente no mejora

Guía paso a paso para diagnosticar y arreglar el entrenamiento.
No necesitas saber de RL — solo sigue las instrucciones según lo que ves en pantalla.

---

## Cómo leer las métricas

Antes de tocar nada, entiende qué significa cada número:

| Métrica | Qué es | Buena señal | Mala señal |
|---|---|---|---|
| **distancia avg100** | Promedio de qué tan lejos llega Mario en los últimos 100 episodios | Sube con el tiempo | Estancado o baja |
| **reward avg100** | Recompensa promedio últimos 100 episodios | Sube (puede ser negativo al inicio) | Muy negativo y no sube |
| **clear rate100** | % de veces que completa el nivel en últimos 100 episodios | > 0% a partir de ~2M steps | 0% después de 2M steps |
| **max x hist.** | Lo más lejos que llegó alguna vez | Sigue creciendo | No se mueve hace rato |
| **behavior score** | Penalizaciones acumuladas (más bajo = más penalizaciones) | Baja gradualmente | Muy alto (muchas penalizaciones) |
| **steps/s** | Velocidad de entrenamiento | 200-400 con GPU + 4 envs | < 100 (algo anda mal) |
| **coins avg100** | Monedas promedio recogidas | Sube | No importa mucho |

---

## Checkpoint 500k steps — Primera revisión

### Si va BIEN (distancia avg100 > 400):
No toques nada. Déjalo correr hasta 2M steps y revisá de nuevo.

### Si va REGULAR (distancia avg100 entre 250-400):
Paciencia. Algunos runs tardan más. Revisá de nuevo a 1M steps.

### Si va MAL (distancia avg100 < 250):
El agente no está aprendiendo a avanzar. Hacé esto:

**Paso 1 — Suavizar penalizaciones (hot-reload, NO requiere reiniciar)**

Abrí el dashboard en `http://127.0.0.1:8000` o editá `configs/default.yaml`:

```yaml
reward:
  stuck_penalty_base: -0.2        # era -0.5 (menos castigo por quedarse quieto)
  wall_stuck_penalty: -0.1        # era -0.3
  micro_movement_penalty: -0.3    # era -1.0 (este es el más agresivo)
  excessive_left_penalty: -0.05   # era -0.1
  cyclic_action_penalty: -0.1     # era -0.2
```

Si usas el dashboard, cambiá los valores y apretá "Aplicar en caliente".
Si editás el YAML, tenés que reiniciar el training.

**Paso 2 — Esperar 200k steps más y revisar**

Si después de 200k steps más sigue sin mejorar, ir al paso de 1M.

---

## Checkpoint 1M steps — Segunda revisión

### Si va BIEN (distancia avg100 > 600, max hist > 2000):
Perfecto. Déjalo correr.

### Si va MAL (distancia avg100 < 400):

**Paso 1 — Subir exploración (REQUIERE reiniciar)**

Editá `configs/default.yaml`:

```yaml
ppo:
  ent_coef: 0.04    # era 0.02 — fuerza más exploración
```

Y reiniciá el training **desde el último checkpoint**:

```bash
python app.py
# Opción 2 (Reanudar)
# Ruta: checkpoints/mario_ppo_1000000
```

**Paso 2 — Si sigue mal después de 500k steps más:**

Subir la recompensa por avanzar:

```yaml
reward:
  forward_reward_coef: 2.0    # era 1.0 — más incentivo para ir a la derecha
  flag_bonus: 30.0            # era 15.0 — más premio por completar
```

---

## Checkpoint 2M steps — Tercera revisión

### Si va BIEN (clear rate > 0%, distancia avg100 > 1500):
El agente está aprendiendo a completar el nivel. Déjalo hasta 5M.

### Si hay clears esporádicos pero clear rate < 5%:
Está cerca. Ayudalo:

```yaml
reward:
  flag_bonus: 50.0                    # era 15 — mucho más premio por completar
  time_remaining_bonus_coef: 0.3      # era 0.1 — premio por completar rápido
  death_by_enemy_penalty: -20.0       # era -15 — más castigo por morir
```

Estos cambios se pueden aplicar en caliente desde el dashboard.

### Si distancia avg100 < 800 y 0 clears:
El run probablemente no va a converger. Reiniciar desde cero con config agresiva:

```bash
# Borrar checkpoints viejos (opcional)
rm checkpoints/mario_ppo_*.zip
rm checkpoints/death_map.json
rm checkpoints/best_distance.json
```

Editá `configs/default.yaml` con estos cambios:

```yaml
ppo:
  ent_coef: 0.05              # exploración alta
  learning_rate: 0.0003       # lr un poco más alto
  use_linear_lr_schedule: true

reward:
  forward_reward_coef: 2.0
  flag_bonus: 40.0
  stuck_penalty_base: -0.2
  micro_movement_penalty: -0.3
  wall_stuck_penalty: -0.1
  death_by_enemy_penalty: -10.0    # menos castigo por morir (que explore)
  enable_death_map: false          # desactivar — puede ser contraproducente
```

```bash
python app.py --train
```

---

## Checkpoint 3M-4M steps — Refinamiento

### Si clear rate > 50%:
El agente ya sabe completar el nivel. Para maximizar el puntaje:

```yaml
reward:
  coin_reward: 10.0                   # era 5 — más incentivo por monedas
  score_reward_coef: 0.05             # era 0.025 — más incentivo por puntos
  time_remaining_bonus_coef: 0.5      # era 0.1 — completar rápido = más puntos
```

### Si clear rate entre 10-50%:
Sigue mejorando pero lento. Déjalo hasta 5M. Opcionalmente:

```yaml
reward:
  flag_bonus: 60.0      # incentivo fuerte por completar
```

### Si clear rate < 10%:
Está costando. Considerá:
1. Bajar `death_by_enemy_penalty` a `-8.0` (menos miedo a morir)
2. Subir `ent_coef` a `0.03` (requiere reiniciar desde checkpoint)

---

## Checkpoint 5M steps — Final

### Si clear rate > 80%:
Excelente. Guardá el modelo final:

```bash
python app.py
# Opción 3 (Evaluar)
# Ruta: models_saved/mario_ppo
# Episodios: 20
```

Si el promedio de x es > 3000 y tiene clears consistentes, el modelo está listo.

### Si clear rate entre 30-80%:
Extender el entrenamiento:

```bash
python app.py --train --resume --model-path models_saved/mario_ppo --timesteps 8000000
```

O desde la CLI:

```bash
# Editá default.yaml
total_timesteps: 10000000

python app.py
# Opción 2 (Reanudar)
# Ruta: models_saved/mario_ppo
```

### Si clear rate < 30%:
Reiniciar con la config agresiva de la sección 2M (arriba).

---

## Tabla rápida de emergencia

| Síntoma | Solución | Hot-reload? |
|---|---|---|
| Mario se queda quieto | Bajar `stuck_penalty_base` a `-0.2` | Sí |
| Mario va y viene sin avanzar | Bajar `micro_movement_penalty` a `-0.3` | Sí |
| Mario muere mucho y no aprende | Bajar `death_by_enemy_penalty` a `-8` | Sí |
| Mario avanza pero nunca llega al final | Subir `flag_bonus` a `40+` | Sí |
| Mario no explora, siempre hace lo mismo | Subir `ent_coef` a `0.04` | **No — reiniciar** |
| Todo estancado hace 500k+ steps | Subir `learning_rate` a `0.0003` | **No — reiniciar** |
| Reward muy negativo (-1000+) | Suavizar TODAS las penalizaciones | Sí |
| Steps/s muy bajo (< 100) | Verificar que `device: cuda` y `n_envs: 4` | No — reiniciar |
| Se congela al arrancar | Normal — SubprocVecEnv tarda 5-10s en crear envs | Esperar |

---

## Qué significa "hot-reload"

- **Sí (hot-reload)**: podés cambiar el valor mientras el training corre, sin reiniciar. Desde el dashboard web o editando el YAML + reiniciando.
- **No — reiniciar**: tenés que parar el training (Ctrl+C), editar el YAML, y reanudar desde el último checkpoint.

Para reanudar siempre:

```bash
python app.py
# Opción 2
# Ruta: checkpoints/mario_ppo_XXXXXX    (el número más alto)
```

---

## Orden recomendado de ajustes

Si no sabés por dónde empezar, seguí este orden:

1. **Primero**: suavizar penalizaciones (hot-reload, fácil, sin riesgo)
2. **Segundo**: subir recompensas positivas (hot-reload)
3. **Tercero**: subir `ent_coef` (requiere reiniciar, pero es el cambio más impactante)
4. **Último recurso**: reiniciar desde cero con config agresiva

Nunca cambies más de 2-3 cosas a la vez. Esperá al menos 200k-500k steps después de cada cambio para ver si tuvo efecto.
