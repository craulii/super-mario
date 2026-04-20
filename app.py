"""Terminal app unificada para Mario RL.

Uso:
    python app.py                         # menú interactivo
    python app.py --train                 # entrena con configs/default.yaml
    python app.py --config configs/fast_debug.yaml --train
    python app.py --evaluate --model-path models_saved/mario_ppo --episodes 5
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.prompt import IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from configs.schema import Config, load_config
from env.level_metrics import X_MAX_LEVEL_1_2
from training.shared_state import SharedState
from training.trainer import Trainer
from ui.observer import start_in_background as start_dashboard
from utils.progress_bar import render_bar

console = Console()


def _header(cfg: Config, mode: str) -> Panel:
    body = (
        f"modo={mode}  env={cfg.env.env_id}  device={cfg.training.device}  "
        f"timesteps={cfg.training.total_timesteps:,}  lr={cfg.ppo.learning_rate}"
    )
    return Panel(body, title="Mario RL", border_style="cyan")


def _progress_panel(state: SharedState, cfg: Config) -> Panel:
    snap = state.snapshot()
    x = snap["current_x"]
    pct = (x / X_MAX_LEVEL_1_2) * 100 if X_MAX_LEVEL_1_2 else 0.0
    bar = ProgressBar(total=X_MAX_LEVEL_1_2, completed=x, width=None)
    label = Text.assemble(
        ("progreso del nivel  ", "bold"),
        (f"{pct:5.1f}%  ", "cyan bold"),
        (f"x={x}/{X_MAX_LEVEL_1_2}", "dim"),
    )
    body = Group(label, bar)
    color = "green" if snap["mode"] == "training" else "yellow" if snap["mode"] == "paused" else "cyan"
    return Panel(body, title=f"Mario RL — {snap['mode']}", border_style=color, padding=(1, 2))


def _metrics_panel(state: SharedState, cfg: Config) -> Panel:
    snap = state.snapshot()

    # Calcular % de entrenamiento y steps/s
    total = cfg.training.total_timesteps
    train_pct = (snap["timesteps"] / total * 100) if total > 0 else 0.0
    t0 = snap.get("training_start_time", 0.0)
    elapsed = time.time() - t0 if t0 > 0 else 0.0
    sps = snap["timesteps"] / elapsed if elapsed > 1 else 0.0

    tbl = Table.grid(padding=(0, 2), expand=True)
    tbl.add_column(justify="right", style="bold cyan", no_wrap=True)
    tbl.add_column(justify="left")
    tbl.add_column(justify="right", style="bold cyan", no_wrap=True)
    tbl.add_column(justify="left")
    rows = [
        ("timesteps", f"{snap['timesteps']:,}", "episodios", f"{snap['episodes']}"),
        ("entrenamiento", f"{train_pct:.1f}%", "steps/s", f"{sps:.0f}"),
        ("max x ep.", f"{snap['max_x_this_ep']}", "max x hist.", f"{snap['max_x_historical']:.0f}"),
        ("reward avg100", f"{snap['reward_avg_100']:.2f}", "último reward", f"{snap['last_episode_reward']:.2f}"),
        ("distancia avg100", f"{snap['distance_avg_100']:.0f}", "clear rate100", f"{snap['clear_rate_100']*100:.1f}%"),
        ("coins avg100", f"{snap['coins_avg_100']:.1f}", "behavior score", f"{snap['behavior_score']:.2f}"),
        ("último flag", "sí" if snap["last_clear"] else "no", "device", cfg.training.device),
    ]
    for a, b, c, d in rows:
        tbl.add_row(a, b, c, d)
    return Panel(tbl, title="Métricas", border_style="blue", padding=(0, 1))


def _info_panel(cfg: Config, state: SharedState) -> Panel:
    snap = state.snapshot()
    lines = [
        f"[bold]env:[/]         {cfg.env.env_id}",
        f"[bold]timesteps tot:[/] {cfg.training.total_timesteps:,}",
        f"[bold]lr:[/]          {cfg.ppo.learning_rate}   [bold]batch:[/] {cfg.ppo.batch_size}   [bold]n_steps:[/] {cfg.ppo.n_steps}",
        f"[bold]flag bonus:[/]  {cfg.reward.flag_bonus}   [bold]forward:[/] {cfg.reward.forward_reward_coef}",
    ]
    if snap.get("message"):
        lines.append("")
        lines.append(f"[yellow]{snap['message']}[/]")
    return Panel("\n".join(lines), title="Config", border_style="magenta", padding=(0, 1))


def _build_layout(state: SharedState, cfg: Config) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top", size=5),
        Layout(name="middle", size=10),
        Layout(name="bottom"),
    )
    layout["top"].update(_progress_panel(state, cfg))
    layout["middle"].update(_metrics_panel(state, cfg))
    layout["bottom"].update(_info_panel(cfg, state))
    return layout


def _run_with_live(state: SharedState, cfg: Config, label: str) -> None:
    """Bloquea mostrando un Live pantalla-completa hasta que mode vuelva a idle.

    Usa `screen=True` para ocupar el alternate buffer del terminal (tipo htop):
    la vista se actualiza en su lugar sin reimprimir ni empujar scroll.
    """
    with Live(
        _build_layout(state, cfg),
        refresh_per_second=4,
        screen=True,
        console=console,
    ) as live:
        while True:
            live.update(_build_layout(state, cfg))
            snap = state.snapshot()
            if snap["mode"] == "idle" and snap["timesteps"] > 0:
                time.sleep(0.3)
                live.update(_build_layout(state, cfg))
                break
            if snap["mode"] == "idle" and snap["timesteps"] == 0 and label == "eval":
                break
            time.sleep(0.25)


def _install_sigint(trainer: Trainer) -> None:
    def _handler(signum, frame):
        if trainer.is_running():
            console.print("\n[yellow]Ctrl+C detectado. Deteniendo y guardando...[/]")
            try:
                trainer.save(Path("checkpoints") / "interrupted")
            except Exception as exc:
                console.print(f"[red]no se pudo guardar checkpoint: {exc}[/]")
            trainer.stop(timeout=30)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)


# ---------------------------------------------------------------------------
# Acciones del menú
# ---------------------------------------------------------------------------
def _action_train(trainer: Trainer, cfg: Config, resume: bool) -> None:
    resume_path: Path | None = None
    if resume:
        default = cfg.paths.model_save_path
        entered = Prompt.ask(
            "Ruta del modelo a reanudar",
            default=str(default) if Path(str(default) + ".zip").exists() else "",
        )
        resume_path = Path(entered) if entered else None
    trainer.start(resume_path=resume_path)
    _run_with_live(trainer._state, cfg, label="train")


def _action_evaluate(trainer: Trainer, cfg: Config) -> None:
    default_path = cfg.paths.model_save_path
    model_path = Prompt.ask("Ruta del modelo", default=str(default_path))
    episodes = IntPrompt.ask("Episodios", default=5)
    render = Prompt.ask("Renderizar (y/n)", default="n") == "y"

    try:
        results = trainer.evaluate(model_path=model_path, n_episodes=episodes, render=render)
    except FileNotFoundError:
        console.print(f"[red]Modelo no encontrado: {model_path}[/]")
        return

    tbl = Table(title="Resultados de evaluación", show_lines=False)
    tbl.add_column("Ep", justify="right")
    tbl.add_column("Reward", justify="right")
    tbl.add_column("Max X", justify="right")
    tbl.add_column("Clear", justify="center")
    tbl.add_column("Coins", justify="right")
    tbl.add_column("Score", justify="right")
    tbl.add_column("T.left", justify="right")
    tbl.add_column("Tiempo", justify="right")
    for r in results:
        tbl.add_row(
            str(r["episode"]),
            f"{r['reward']:.1f}",
            str(r["max_x"]),
            "sí" if r["flag_get"] else "no",
            str(r["coins"]),
            str(r["score"]),
            str(r["time_left"]),
            f"{r['elapsed_s']:.1f}s",
        )
    console.print(tbl)


def _action_save(trainer: Trainer, cfg: Config) -> None:
    if trainer.is_running() and not trainer.is_paused():
        console.print("[yellow]Pausa el entrenamiento antes de guardar.[/]")
        return
    path = Prompt.ask("Ruta destino", default=str(cfg.paths.model_save_path))
    try:
        final = trainer.save(Path(path))
        console.print(f"[green]Guardado en {final}[/]")
    except Exception as exc:
        console.print(f"[red]error: {exc}[/]")


def _action_load(trainer: Trainer, cfg: Config) -> None:
    if trainer.is_running():
        console.print("[yellow]Detén el entrenamiento antes de cargar.[/]")
        return
    path = Prompt.ask("Ruta del modelo", default=str(cfg.paths.model_save_path))
    try:
        meta = trainer.load(Path(path))
        console.print(f"[green]Modelo cargado. Metadata:[/] {meta}")
    except Exception as exc:
        console.print(f"[red]error: {exc}[/]")


def _action_status(state: SharedState, cfg: Config) -> None:
    console.print(_live_table(state, cfg))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mario RL terminal app")
    p.add_argument("--config", default="configs/default.yaml", help="Ruta al YAML de config")
    p.add_argument("--train", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--demo", action="store_true", help="Modo demostración en http://127.0.0.1:8000/demo")
    p.add_argument("--model-path", default=None)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--timesteps", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    if args.timesteps is not None:
        cfg.training.total_timesteps = args.timesteps
    cfg.ensure_dirs()

    state = SharedState(reward_config=cfg.reward)
    trainer = Trainer(cfg, state)
    _install_sigint(trainer)

    # Dashboard web read-only — se engancha al mismo SharedState
    start_dashboard(state, cfg, port=8000)
    console.print("[dim]dashboard en http://127.0.0.1:8000[/]")

    # Modo demo
    if args.demo:
        model = args.model_path or str(cfg.paths.model_save_path)
        console.print(f"[bold cyan]Modo demostración[/]")
        console.print(f"[dim]modelo: {model}[/]")
        console.print(f"[bold green]Abrí http://127.0.0.1:8000/demo en el navegador[/]")
        console.print("[dim]Ctrl+C para salir[/]")
        try:
            trainer.run_demo(model_path=model, n_episodes=0)
        except FileNotFoundError:
            console.print(f"[red]Modelo no encontrado: {model}[/]")
        except KeyboardInterrupt:
            pass
        return

    # Modo directo por CLI
    if args.train:
        console.print(_header(cfg, "train"))
        resume_path = Path(args.model_path) if (args.resume and args.model_path) else None
        trainer.start(resume_path=resume_path)
        _run_with_live(state, cfg, label="train")
        return

    if args.evaluate:
        console.print(_header(cfg, "evaluate"))
        try:
            results = trainer.evaluate(
                model_path=args.model_path, n_episodes=args.episodes, render=False
            )
        except FileNotFoundError:
            console.print(f"[red]Modelo no encontrado.[/]")
            return
        for r in results:
            console.print(r)
        return

    # Menú interactivo
    while True:
        console.print(_header(cfg, state.mode))
        console.print(
            "\n"
            "  1) Entrenar desde cero\n"
            "  2) Reanudar entrenamiento\n"
            "  3) Evaluar modelo\n"
            "  4) Guardar modelo actual\n"
            "  5) Cargar modelo\n"
            "  6) Ver estado\n"
            "  7) Salir\n"
        )
        choice = IntPrompt.ask("Elegí una opción", choices=[str(i) for i in range(1, 8)])
        if choice == 1:
            _action_train(trainer, cfg, resume=False)
        elif choice == 2:
            _action_train(trainer, cfg, resume=True)
        elif choice == 3:
            _action_evaluate(trainer, cfg)
        elif choice == 4:
            _action_save(trainer, cfg)
        elif choice == 5:
            _action_load(trainer, cfg)
        elif choice == 6:
            _action_status(state, cfg)
        elif choice == 7:
            if trainer.is_running():
                trainer.stop(timeout=10)
            console.print("hasta luego")
            return


if __name__ == "__main__":
    main()
