from isaaclab.sensors import TiledCameraCfg, TiledCamera
from isaaclab.utils import configclass

import torch
import torchvision.transforms.functional as F
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .._base_task import BaseTask
    from tacex_uipc import UipcInteractiveScene

@configclass
class CameraCfg(TiledCameraCfg):
    name: str = 'camera'


class _DebugImageWindow:
    """A lazily-created Isaac Sim GUI window that displays a camera RGB frame.

    Mirrors the GelSight sensors' built-in debug windows so camera previews can
    be shown alongside the tactile images when debug_vis is enabled. omni.ui /
    numpy are imported lazily so this module stays importable without Kit.
    """

    def __init__(self, title: str, width: int, height: int):
        import omni.ui as ui

        self._ui = ui
        self._window = ui.Window(title, width=width, height=height)
        self._provider = ui.ByteImageProvider()

    def update(self, rgb) -> None:
        import numpy as np

        if rgb.ndim == 4:  # (num_envs, H, W, C) -> first env
            rgb = rgb[0]
        frame = rgb.detach().cpu().numpy()
        if frame.dtype != np.uint8:
            max_value = float(np.nanmax(frame)) if frame.size else 1.0
            if max_value <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        alpha = np.full(frame.shape[:2] + (1,), 255, dtype=np.uint8)
        frame = np.concatenate((frame[..., :3], alpha), axis=-1)
        height, width, _ = frame.shape
        with self._window.frame:
            self._provider.set_bytes_data(frame.flatten().data, [width, height])
            self._ui.ImageWithProvider(self._provider)


class CameraManager:
    def __init__(self, cfg_list: list[CameraCfg], task:'BaseTask'):
        self.scene = task.scene
        self.cfg_list = cfg_list
        self.cameras = {}
        self._debug_vis = False
        self._debug_windows = {}

    def setup(self):
        self.cameras = {
            cam_cfg.name: self.add_camera(cam_cfg) for cam_cfg in self.cfg_list
        }

    def set_debug_vis(self, debug_vis: bool):
        # Windows are created lazily in update_debug_vis() once the cameras have
        # produced their first frame (and omni.ui is available under Kit).
        self._debug_vis = debug_vis

    def update_debug_vis(self):
        if not self._debug_vis:
            return
        for name, cam in self.cameras.items():
            output = cam.data.output
            if 'rgb' not in output:
                continue
            if name not in self._debug_windows:
                self._debug_windows[name] = _DebugImageWindow(
                    f"camera: {name}", cam.cfg.width, cam.cfg.height
                )
            self._debug_windows[name].update(output['rgb'])

    def add_camera(self, cam_cfg: CameraCfg):
        camera = TiledCamera(cam_cfg)
        camera._initialize_impl()
        camera._is_initialized = True
        self.scene.sensors[f'camera_{cam_cfg.name}'] = camera
        return camera
    
    def get_observations(self, data_types: list[str] = None):
        obs = {}
        if data_types is None:
            data_types = ['rgb', 'rgba']
        for name, cam in self.cameras.items():
            obs[name] = {}
            for data_type in data_types:
                if data_type == 'rgb':
                    obs[name]['rgb'] = cam.data.output['rgb'].squeeze(0)
                elif data_type == 'rgba':
                    obs[name]['rgba'] = cam.data.output['rgba'].squeeze(0)
                elif data_type == 'depth':
                    obs[name]['depth'] = cam.data.output['depth'].squeeze(0)
        return obs
