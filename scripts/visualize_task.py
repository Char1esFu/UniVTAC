"""
Visualize a task's scripted motion in a GUI Isaac Sim window WITHOUT collecting data.

This reuses each task's own movement definition (`pre_move` + `_play_once`), so it is
exactly the same combined motion that data collection would run -- but nothing is saved
(no hdf5, no video, no pkl frames). Use it to eyeball whether the composed motions of a
task are correct.

Examples:
    # show one task, seeds 0..4, with a local GUI window
    python scripts/visualize_task.py lift_can

    # specific seeds, slower playback, and draw a box at each motion target
    python scripts/visualize_task.py insert_hole --seeds 0 1 2 --debug_vis

    # loop forever over a seed range until you close the window
    python scripts/visualize_task.py lift_bottle --start_seed 0 --max_seed 20 --loop
"""

import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.append('.')

parser = argparse.ArgumentParser(description="Visualize a task's scripted motion in a GUI (no data collection).")
parser.add_argument("task", type=str, help="Task file name under envs/ (e.g. lift_can, insert_hole).")
parser.add_argument("--seeds", type=int, nargs='+', default=None,
                    help="Explicit list of seeds to play. Overrides --start_seed/--max_seed.")
parser.add_argument("--start_seed", type=int, default=0, help="First seed (when --seeds is not given).")
parser.add_argument("--max_seed", type=int, default=4, help="Last seed, inclusive (when --seeds is not given).")
parser.add_argument("--sensor_type", type=str, default='gsmini', choices=['gsmini', 'gf225', 'xensews'])
parser.add_argument("--render_frequency", type=int, default=1,
                    help="Render every N sim steps. 1 = smoothest GUI (default), higher = faster but choppier.")
parser.add_argument("--debug_vis", action="store_true",
                    help="Draw a red box at each motion-planning target pose (helps verify targets).")
parser.add_argument("--loop", action="store_true",
                    help="Keep cycling through the seeds until the window is closed.")
parser.add_argument("--hold", type=float, default=1.0,
                    help="Seconds to keep rendering between episodes so you can inspect the final state.")
parser.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES value.")

args_cli = parser.parse_args()
if args_cli.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args_cli.gpu

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)

# --- force a local GUI window, no data, cameras enabled (tactile/camera managers need them) ---
args_cli.headless = False      # GUI window (livestream is intentionally NOT used)
args_cli.enable_cameras = True
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib
if TYPE_CHECKING:
    from envs._base_task import BaseTask, BaseTaskCfg


def build_task(task_file_name: str) -> 'BaseTask':
    task_module = importlib.import_module(f"envs.{task_file_name}")
    env_cfg: 'BaseTaskCfg' = task_module.TaskCfg()

    env_cfg.tactile_sensor_type = args_cli.sensor_type

    # Keep all scene scratch files out of ./data so nothing looks like collected data.
    env_cfg.save_dir = Path('/tmp') / 'univtac_vis' / task_file_name

    # Disable every save path; only render for the GUI.
    env_cfg.render_frequency = args_cli.render_frequency  # >0 -> _update_render() drives the GUI
    env_cfg.video_frequency = 0                           # no video writing
    env_cfg.save_frequency = 10 ** 9                      # never hit the pkl-save branch
    env_cfg.obs_data_type = {}                            # don't gather observations
    env_cfg.random_texture = False
    env_cfg.debug_vis = args_cli.debug_vis

    env_cfg.scene.num_envs = 1

    # mode='collect' so the scripted gripper/grasp logic (adaptive grasp) runs as designed.
    return task_module.Task(env_cfg, mode='collect')


def render_for(task: 'BaseTask', seconds: float):
    """Keep the GUI responsive / let physics settle for a moment."""
    if seconds <= 0:
        return
    end = time.perf_counter() + seconds
    while time.perf_counter() < end and simulation_app.is_running():
        task._update_render()


def play_seed(task: 'BaseTask', seed: int):
    print(f"\n=== Visualizing task '{args_cli.task}', seed {seed} ===")
    start = time.perf_counter()
    # reset() runs pre_move(); _play_once() runs the task's main scripted motion.
    # We call _play_once() directly (instead of play_once()) to skip metadata.json writing.
    task.reset(seed=seed)
    task._play_once()
    print(f"\nseed {seed}: plan_success={task.plan_success}, "
          f"check_success={task.check_success()}, took {time.perf_counter() - start:.1f}s")
    render_for(task, args_cli.hold)


def main():
    if args_cli.seeds is not None:
        seeds = args_cli.seeds
    else:
        seeds = list(range(args_cli.start_seed, args_cli.max_seed + 1))

    task = build_task(args_cli.task)

    try:
        while simulation_app.is_running():
            for seed in seeds:
                if not simulation_app.is_running():
                    break
                try:
                    play_seed(task, seed)
                except Exception:
                    print(f"[seed {seed}] failed:\n{traceback.format_exc()}")
            if not args_cli.loop:
                break

        # Keep the window open at the end so the user can inspect the final scene.
        if simulation_app.is_running():
            print("\nDone. Close the Isaac Sim window to exit.")
            while simulation_app.is_running():
                task._update_render()
    finally:
        task.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
