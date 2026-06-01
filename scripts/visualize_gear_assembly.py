"""Spawn the insert-HDMI table scene, the Factory gear-assembly assets and a
Franka + GelSight Mini gripper, then open Isaac Sim for visual inspection.

The Franka arm is initialized by sampling a random gripper target near
GRIPPER_INIT_CENTER_W and solving IK from FRANKA_IK_SEED_JOINT_POS. PhysX FK
propagates through every link automatically, including the gelsight_mini_case
left/right since they are FixedJoint-connected to the fingers inside the Franka
USD.

The gelpads have no RigidBodyAPI and no joint in the USD — they normally rely
on UIPC `UipcIsaacAttachments` to follow the cases. This script doesn't run
UIPC, so we manually sync `gelpad_{left,right}` world pose to the corresponding
case body each frame (the position-only part of what UIPC attachments do).

Usage:
    python scripts/visualize_gear_assembly.py
"""

import argparse
import os
import subprocess
import sys
from types import SimpleNamespace

from isaaclab.app import AppLauncher

sys.path.append(".")

parser = argparse.ArgumentParser(description="Visualize gear assembly assets on the table.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# WebRTC livestream (mode 2) — view in the isaacsim-webrtc-streaming-client.
# Livestream implies headless; the local GUI window is not opened.
args_cli.livestream = 2
args_cli.enable_cameras = True
args_cli.num_envs = 1

# AppLauncher emits `--/app/livestream/publicEndpointAddress=$PUBLIC_IP` itself
# (default 127.0.0.1), so override via the env var rather than sys.argv — any
# manual sys.argv entry would be appended *before* AppLauncher's, and carb
# takes the last value wins.
os.environ["PUBLIC_IP"] = subprocess.check_output(
    ["tailscale", "ip", "-4"], text=True
).strip().splitlines()[0]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import omni.ui as ui
import omni.usd
from pxr import UsdGeom

from isaacsim.core.prims import XFormPrim

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from tacex import GelSightSensor
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext
from isaaclab.sim.schemas.schemas_cfg import (
    ArticulationRootPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from tacex_assets.robots.franka.franka_gsmini_gripper_uipc_high_res import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_taxim import GELSIGHT_MINI_TAXIM_CFG

from envs._global import EMBODIMENTS_ROOT, OBJECTS_ROOT, SCENE_ASSETS_ROOT
from envs.robot.curobo_planner import CuroboPlanner, CuroboPlannerCfg
from envs.utils.transforms import Pose


# Contact-ready seed joint configuration copied from
# envs/robot/robot_cfg.py::create_franka_gsmini_gripper — same starting pose
# the production insert_* tasks use, so the gripper sits over the table
# workspace at startup.
FRANKA_IK_SEED_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -1.1,
    "panda_joint3": 0.0,
    "panda_joint4": -2.6,
    "panda_joint5": 0.0,
    "panda_joint6": 1.6,
    "panda_joint7": 0.8,
    "panda_finger.*": 0.04,
}

GRIPPER_INIT_CENTER_W = (0.25, 0.0, 0.55)
GRIPPER_INIT_RANDOM_RANGE = 0.05
GRIPPER_IK_BODY_NAME = "panda_link8"
FRANKA_FINGER_INIT_POS = 0.04
FRANKA_FINGER_GRASP_POS = 0.0
GRASP_CLOSE_STEPS = 50
GRASP_OPEN_STEPS = 50
GRASP_ARRIVAL_POS_TOL = 0.02
IK_MAX_ITERS = 100
IK_DLS_LAMBDA = 0.1
IK_POSITION_TOL = 1e-3
CUROBO_GRASP_TARGET_X_OFFSET = 0.02
CUROBO_GRASP_TARGET_Z_OFFSET = 0.035
MEDIUM_GEAR_PREP_X_OFFSET = 0.018
MEDIUM_GEAR_PREP_Z_OFFSET = 0.07
MEDIUM_GEAR_LOWER_Z_OFFSET = 0.03
MEDIUM_GEAR_LIFT_POS_TOL = 0.02
# Translate must finish much closer to its target than the lift/lower phases —
# the next motion (lowering with the gear) drops the gear onto its peg, and
# any horizontal residual at the start of that descent leaves the gear off-peg.
MEDIUM_GEAR_CONTACT_MOVE_TOL = 0.002
CUROBO_GRASP_TARGET_TOP_DOWN_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)

REALSENSE_CAMERA_PRIM_PATH = "/World/RealsenseCamera"
REALSENSE_CAMERA_POS_W = (1.5, 0.0, 1.5)
REALSENSE_CAMERA_TARGET_W = (0.3, 0.0, 0.0)
REALSENSE_CAMERA_WIDTH = 640
REALSENSE_CAMERA_HEIGHT = 480
TACTILE_DEBUG_VIS = True
REALSENSE_DEBUG_VIS = True




# (dx, dy, dz) offset of each gear's spawn position relative to the gear-base
# origin. dz is added on top of the drop height defined in `design_scene()`.
# Factory's reference peg locations are
#   small=(5.075e-2, 0, 0), medium=(2.025e-2, 0, 0), large=(-3.025e-2, 0, 0)
# (see IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/
#  factory_tasks_cfg.py:191-193) — set these here to drop each gear onto its
# matching peg slot.
GEAR_BASE_OFFSETS: dict[str, tuple[float, float, float]] = {
    "small":  (0.0, 0.0, 0.0),
    "medium": (0.0, 0.15, 0.0),
    "large":  (0.0, 0.0, 0.0),
}
GEAR_MASS = {"small": 0.019, "medium": 0.012, "large": 0.019}

# Yaw (radians) of the gear-base around world +z. The base sits flat on the
# plate, so pitch/roll are not exposed — only yaw is physically meaningful.
BASE_YAW = 1.57


def _gear_rigid_props(disable_gravity: bool = False) -> RigidBodyPropertiesCfg:
    return RigidBodyPropertiesCfg(
        disable_gravity=disable_gravity,
        max_depenetration_velocity=5.0,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=3666.0,
        enable_gyroscopic_forces=True,
        solver_position_iteration_count=192,
        solver_velocity_iteration_count=1,
        max_contact_impulse=1e32,
    )


def design_scene() -> dict[str, RigidObject | Articulation]:
    # dome light + HDRI background (matches BaseTaskCfg.light)
    dome_cfg = sim_utils.DomeLightCfg(
        color=(0.75, 0.75, 0.75),
        intensity=3000.0,
        texture_file=str(SCENE_ASSETS_ROOT / "base0.exr"),
    )
    dome_cfg.func("/World/light", dome_cfg)

    # ground plate (kinematic, matches BaseTaskCfg.plate)
    plate = RigidObject(
        RigidObjectCfg(
            prim_path="/World/plate",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(SCENE_ASSETS_ROOT / "plate.usda"),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    kinematic_enabled=True,
                ),
            ),
        )
    )

    # gear base: pinned to the world via a fixed joint on the articulation
    # root, so it stays put under contact loads while internal joints stay
    # valid for the contact-rich solver.
    base_pose = (0.4, 0.0, 0.005)
    half_yaw = 0.5 * BASE_YAW
    base_quat = (float(np.cos(half_yaw)), 0.0, 0.0, float(np.sin(half_yaw)))
    cos_yaw = float(np.cos(BASE_YAW))
    sin_yaw = float(np.sin(BASE_YAW))
    gear_base = Articulation(
        ArticulationCfg(
            prim_path="/World/gear_base",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=base_pose, rot=base_quat, joint_pos={}, joint_vel={}
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(OBJECTS_ROOT / "factory_gear_base.usd"),
                activate_contact_sensors=False,
                rigid_props=_gear_rigid_props(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.005, rest_offset=0.0
                ),
                articulation_props=ArticulationRootPropertiesCfg(fix_root_link=True),
            ),
            actuators={},
        )
    )

    # dynamic gears, dropped a few cm above their corresponding pegs
    drop_height = 0.0
    gears: dict[str, Articulation] = {}
    for size, (dx, dy, dz) in GEAR_BASE_OFFSETS.items():
        dxr = cos_yaw * dx - sin_yaw * dy
        dyr = sin_yaw * dx + cos_yaw * dy
        gears[size] = Articulation(
            ArticulationCfg(
                prim_path=f"/World/gear_{size}",
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(
                        base_pose[0] + dxr,
                        base_pose[1] + dyr,
                        drop_height + dz,
                    ),
                    rot=base_quat,
                    joint_pos={},
                    joint_vel={},
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(OBJECTS_ROOT / f"factory_gear_{size}.usd"),
                    activate_contact_sensors=False,
                    rigid_props=_gear_rigid_props(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=GEAR_MASS[size]),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.005, rest_offset=0.0
                    ),
                ),
                actuators={},
            )
        )

    # Franka arm + GelSight Mini gripper. Base sits at world (0, 0, 0) —
    # the table edge — matching the production insert_* task layout where the
    # plate is centered at x=0.5 and is reachable from this base position.
    robot_cfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path="/World/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=FRANKA_IK_SEED_JOINT_POS,
        ),
    )
    robot = Articulation(robot_cfg)

    return {"plate": plate, "gear_base": gear_base, "robot": robot, **gears}


def _make_transform(pos: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    transform = torch.eye(4, device=pos.device, dtype=pos.dtype).unsqueeze(0).repeat(pos.shape[0], 1, 1)
    transform[:, :3, :3] = math_utils.matrix_from_quat(quat_wxyz)
    transform[:, :3, 3] = pos
    return transform


def _pose_from_transform(transform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pos = transform[:, :3, 3]
    quat = math_utils.quat_from_matrix(transform[:, :3, :3])
    return pos, quat


def _make_gelsight_taxim_sensor(side: str) -> GelSightSensor:
    cfg = GELSIGHT_MINI_TAXIM_CFG.replace(
        prim_path=f"/World/Robot/gelsight_mini_case_{side}",
        debug_vis=TACTILE_DEBUG_VIS,
        update_period=0.0,
    )
    return GelSightSensor(cfg)


def _enable_tactile_debug_windows(sensor: GelSightSensor) -> None:
    if not TACTILE_DEBUG_VIS or sensor._prim_view is None:
        return
    for prim in sensor._prim_view.prims:
        for data_type in sensor.cfg.data_types:
            attr = prim.GetAttribute(f"debug_{data_type}")
            if attr:
                attr.Set(True)


class DebugImageWindow:
    def __init__(self, name: str, width: int, height: int):
        self._window = ui.Window(name, width=width, height=height)
        self._provider = ui.ByteImageProvider()

    def update_rgb(self, image: torch.Tensor) -> None:
        if image.ndim == 4:
            image = image[0]
        frame = image.detach().cpu().numpy()
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
            ui.ImageWithProvider(self._provider)


def _get_gripper_task_pos_w(robot: Articulation) -> tuple[str, torch.Tensor]:
    case_body_names = ("gelsight_mini_case_left", "gelsight_mini_case_right")
    if all(name in robot.body_names for name in case_body_names):
        left_idx = robot.body_names.index(case_body_names[0])
        right_idx = robot.body_names.index(case_body_names[1])
        return (
            "midpoint(gelsight_mini_case_left,gelsight_mini_case_right)",
            0.5 * (robot.data.body_link_pos_w[:, left_idx] + robot.data.body_link_pos_w[:, right_idx]),
        )
    left_idx = robot.body_names.index("panda_leftfinger")
    right_idx = robot.body_names.index("panda_rightfinger")
    return (
        "midpoint(panda_leftfinger,panda_rightfinger)",
        0.5 * (robot.data.body_link_pos_w[:, left_idx] + robot.data.body_link_pos_w[:, right_idx]),
    )


def _build_curobo_planner(
    robot: Articulation, sim: SimulationContext
) -> tuple[CuroboPlanner, Pose]:
    """Construct the cuRobo planner once so its MotionGen.warmup() pays its
    cost upfront — not on every phase boundary inside the sim loop."""
    stage = omni.usd.get_context().get_stage()
    planner_task = SimpleNamespace(scene=SimpleNamespace(stage=stage))
    arm_joint_names = [f"panda_joint{i}" for i in range(1, 8)]
    root_pose = Pose.from_list(
        torch.cat((robot.data.root_link_pos_w[0], robot.data.root_link_quat_w[0]))
    )
    planner = CuroboPlanner(
        task=planner_task,
        cfg=CuroboPlannerCfg(
            dt=sim.get_physics_dt(),
            all_joints_name=robot.joint_names,
            active_joints_name=arm_joint_names,
            robot_prime_path="/World/Robot",
            yaml_path=str(EMBODIMENTS_ROOT / "franka" / "curobo.yml"),
        ),
        robot_origin_pose=root_pose,
    )
    return planner, root_pose


def _capture_hand_fingertip_transform(robot: Articulation) -> tuple[torch.Tensor, str]:
    """Capture the constant panda_hand → fingertip-midpoint transform.

    The midpoint of the symmetric left/right gelpad cases (or fallback fingers)
    is invariant under finger opening, so this transform is geometric and can
    be reused across all pre-planned segments instead of being recomputed from
    live state inside each plan call.
    """
    hand_idx = robot.body_names.index("panda_hand")
    hand_pos_b, hand_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_link_pos_w,
        robot.data.root_link_quat_w,
        robot.data.body_link_pos_w[:, hand_idx],
        robot.data.body_link_quat_w[:, hand_idx],
    )
    case_body_names = ("gelsight_mini_case_left", "gelsight_mini_case_right")
    if all(name in robot.body_names for name in case_body_names):
        fingertip_frame_name = "midpoint(gelsight_mini_case_left,gelsight_mini_case_right)"
        left_idx = robot.body_names.index(case_body_names[0])
        right_idx = robot.body_names.index(case_body_names[1])
    else:
        fingertip_frame_name = "midpoint(panda_leftfinger,panda_rightfinger)"
        left_idx = robot.body_names.index("panda_leftfinger")
        right_idx = robot.body_names.index("panda_rightfinger")
    fingertip_pos_w = 0.5 * (
        robot.data.body_link_pos_w[:, left_idx] + robot.data.body_link_pos_w[:, right_idx]
    )
    fingertip_quat_w = robot.data.body_link_quat_w[:, hand_idx]
    fingertip_pos_b, fingertip_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_link_pos_w,
        robot.data.root_link_quat_w,
        fingertip_pos_w,
        fingertip_quat_w,
    )
    t_base_hand = _make_transform(hand_pos_b, hand_quat_b)
    t_base_fingertip = _make_transform(fingertip_pos_b, fingertip_quat_b)
    print(
        f"[INFO] captured panda_hand→{fingertip_frame_name} transform from initial pose"
    )
    return torch.linalg.inv(t_base_hand) @ t_base_fingertip, fingertip_frame_name


def _plan_panda_hand_to_target(
    robot: Articulation,
    planner: CuroboPlanner,
    root_pose: Pose,
    t_hand_fingertip: torch.Tensor,
    start_joint_pos: torch.Tensor,
    target_fingertip_pos_w: torch.Tensor,
    label: str,
) -> object:
    """Plan one cuRobo segment from `start_joint_pos` to a top-down grasp at
    `target_fingertip_pos_w`. `start_joint_pos` is a 7-vector of arm joints —
    pass the previous segment's `interpolated_plan.position[-1]` to chain
    segments without depending on live robot state."""
    target_fingertip_quat_w = torch.tensor(
        CUROBO_GRASP_TARGET_TOP_DOWN_QUAT_WXYZ,
        device=robot.device,
        dtype=torch.float32,
    ).unsqueeze(0)
    target_fingertip_pos_b, target_fingertip_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_link_pos_w,
        robot.data.root_link_quat_w,
        target_fingertip_pos_w,
        target_fingertip_quat_w,
    )
    t_base_target_fingertip = _make_transform(target_fingertip_pos_b, target_fingertip_quat_b)
    t_base_target_hand = t_base_target_fingertip @ torch.linalg.inv(t_hand_fingertip)
    target_hand_pos_b, target_hand_quat_b = _pose_from_transform(t_base_target_hand)

    target_hand_pose = Pose(
        p=target_hand_pos_b[0].detach().cpu().numpy(),
        q=target_hand_quat_b[0].detach().cpu().numpy(),
    )

    joint_vel = torch.zeros_like(start_joint_pos)
    result = planner.plan_path(
        curr_joint_pos=start_joint_pos,
        curr_joint_vel=joint_vel,
        target_ee_pose=target_hand_pose,
        real_robot_pose=root_pose,
    )

    print(f"[INFO] cuRobo target label={label}")
    print(
        f"[INFO] cuRobo target fingertip pos_w={target_fingertip_pos_w.squeeze().tolist()} "
        f"top_down quat_wxyz={target_fingertip_quat_w.squeeze().tolist()}"
    )
    print(
        f"[INFO] cuRobo converted target panda_hand pos_b={target_hand_pos_b.squeeze().tolist()} "
        f"quat_b={target_hand_quat_b.squeeze().tolist()}"
    )
    print(f"[INFO] cuRobo plan success={bool(result.success.item())}")
    if result.success.item():
        print(f"[INFO] cuRobo plan steps={result.interpolated_plan.position.shape[0]}")
        print(f"[INFO] cuRobo first joint pos={result.interpolated_plan.position[0].detach().cpu().tolist()}")
        print(f"[INFO] cuRobo final joint pos={result.interpolated_plan.position[-1].detach().cpu().tolist()}")
    else:
        status = getattr(result, "status", None)
        print(f"[WARN] cuRobo plan failed status={status}")
    return result


def _solve_random_gripper_init_ik(robot: Articulation, sim: SimulationContext) -> tuple[torch.Tensor, torch.Tensor]:
    sim_dt = sim.get_physics_dt()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)

    arm_joint_ids = torch.tensor(
        [robot.joint_names.index(f"panda_joint{i}") for i in range(1, 8)],
        device=robot.device,
        dtype=torch.long,
    )
    finger_joint_ids = torch.tensor(
        [idx for idx, name in enumerate(robot.joint_names) if name.startswith("panda_finger")],
        device=robot.device,
        dtype=torch.long,
    )
    joint_pos[:, finger_joint_ids] = FRANKA_FINGER_INIT_POS

    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim_dt)

    gripper_body_idx = robot.body_names.index(GRIPPER_IK_BODY_NAME)
    jacobian_body_idx = gripper_body_idx - 1
    target_pos_w = torch.tensor(
        GRIPPER_INIT_CENTER_W, device=robot.device, dtype=torch.float32
    ).unsqueeze(0)
    target_pos_w += (
        torch.rand_like(target_pos_w) * 2.0 - 1.0
    ) * GRIPPER_INIT_RANDOM_RANGE
    target_quat_w = robot.data.body_link_quat_w[:, gripper_body_idx].clone()

    ik_controller = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": IK_DLS_LAMBDA},
        ),
        num_envs=1,
        device=robot.device,
    )
    ik_controller.set_command(torch.cat((target_pos_w, target_quat_w), dim=-1))
    arm_soft_limits = robot.data.soft_joint_pos_limits[:, arm_joint_ids, :]

    for _ in range(IK_MAX_ITERS):
        curr_pos_w = robot.data.body_link_pos_w[:, gripper_body_idx]
        curr_quat_w = robot.data.body_link_quat_w[:, gripper_body_idx]
        jacobian = robot.root_physx_view.get_jacobians()[:, jacobian_body_idx, 0:6, arm_joint_ids]

        joint_pos[:, arm_joint_ids] = ik_controller.compute(
            curr_pos_w,
            curr_quat_w,
            jacobian,
            joint_pos[:, arm_joint_ids],
        )
        joint_pos[:, arm_joint_ids] = torch.clamp(
            joint_pos[:, arm_joint_ids],
            min=arm_soft_limits[..., 0],
            max=arm_soft_limits[..., 1],
        )
        joint_pos[:, finger_joint_ids] = FRANKA_FINGER_INIT_POS

        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        pos_error = target_pos_w - robot.data.body_link_pos_w[:, gripper_body_idx]
        if pos_error.norm(dim=-1).max() < IK_POSITION_TOL:
            break

    final_pos_error = target_pos_w - robot.data.body_link_pos_w[:, gripper_body_idx]
    print(f"[INFO] random gripper target pos_w={target_pos_w.squeeze().tolist()}")
    print(f"[INFO] IK joint_pos arm={joint_pos[:, arm_joint_ids].squeeze().tolist()}")
    print(f"[INFO] IK body={GRIPPER_IK_BODY_NAME}")
    print(f"[INFO] IK final gripper pos_w={robot.data.body_link_pos_w[:, gripper_body_idx].squeeze().tolist()}")
    print(f"[INFO] IK final position error={final_pos_error.norm(dim=-1).item():.6f} m")

    return joint_pos, joint_vel


def main() -> None:
    sim = SimulationContext(
        SimulationCfg(
            dt=1 / 120,
            physx=PhysxCfg(
                enable_ccd=True,
                # Contact buffer sizing from the IsaacLab gear-assembly tutorial
                # — required headroom for contact-rich PhysX scenes; safe to
                # leave on for single-env visualization too.
                gpu_max_rigid_contact_count=2**23,
                gpu_max_rigid_patch_count=2**23,
                gpu_collision_stack_size=2**31,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.75,
                dynamic_friction=0.75,
                restitution=0.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            ),
        )
    )
    # Camera framed so both the robot (at x=0) and the workspace (x≈0.5) fit.
    sim.set_camera_view(eye=(1.2, 0.9, 0.7), target=(0.4, 0.0, 0.1))

    entities = design_scene()
    tactile_sensors = {side: _make_gelsight_taxim_sensor(side) for side in ("left", "right")}
    realsense_camera = Camera(
        CameraCfg(
            prim_path=REALSENSE_CAMERA_PRIM_PATH,
            update_period=0.0,
            update_latest_camera_pose=True,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=1.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 10.0),
            ),
            width=REALSENSE_CAMERA_WIDTH,
            height=REALSENSE_CAMERA_HEIGHT,
        )
    )
    sim.reset()

    camera_eye = torch.tensor([REALSENSE_CAMERA_POS_W], device=realsense_camera.device, dtype=torch.float32)
    camera_target = torch.tensor([REALSENSE_CAMERA_TARGET_W], device=realsense_camera.device, dtype=torch.float32)
    realsense_camera.set_world_poses_from_view(camera_eye, camera_target)
    realsense_window = (
        DebugImageWindow("Realsense RGB", REALSENSE_CAMERA_WIDTH, REALSENSE_CAMERA_HEIGHT)
        if REALSENSE_DEBUG_VIS
        else None
    )

    robot: Articulation = entities["robot"]  # type: ignore[assignment]

    case_prims = {
        side: XFormPrim(
            prim_paths_expr=f"/World/Robot/gelsight_mini_case_{side}",
            name=f"case_{side}",
            usd=False,
        )
        for side in ("left", "right")
    }
    gelpad_prims = {
        side: XFormPrim(
            prim_paths_expr=f"/World/Robot/gelpad_{side}",
            name=f"gelpad_{side}",
            usd=False,
        )
        for side in ("left", "right")
    }

    # ────────────────────────────────────────────────────────────────────
    # Capture the case ↔ gelpad relative transform straight from USD via
    # pxr. We do not rely on Fabric / robot.data / XFormPrim for this step
    # because they reflect runtime state that may have already drifted by
    # the time we read it. Reading the USD-authored xformOps directly gives
    # the static geometric relationship the NVIDIA author baked in.
    # ────────────────────────────────────────────────────────────────────
    def _read_usd_local_pose(prim_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pos_wxyz=False, quat_wxyz) from a prim's USD-authored local xform."""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        mat = UsdGeom.Xformable(prim).GetLocalTransformation()
        t = mat.ExtractTranslation()
        q = mat.ExtractRotationQuat()  # Gf.Quatd: real + imag(x, y, z)
        pos = torch.tensor([t[0], t[1], t[2]], device=robot.device, dtype=torch.float32)
        # IsaacLab math_utils uses wxyz convention.
        quat = torch.tensor(
            [q.real, q.imaginary[0], q.imaginary[1], q.imaginary[2]],
            device=robot.device,
            dtype=torch.float32,
        )
        return pos.unsqueeze(0), quat.unsqueeze(0)

    gelpad_in_case_pos: dict[str, torch.Tensor] = {}
    gelpad_in_case_quat: dict[str, torch.Tensor] = {}
    for side in ("left", "right"):
        c_pos, c_quat = _read_usd_local_pose(f"/World/Robot/gelsight_mini_case_{side}")
        gp_pos, gp_quat = _read_usd_local_pose(f"/World/Robot/gelpad_{side}")
        c_quat_inv = math_utils.quat_inv(c_quat)
        gelpad_in_case_pos[side] = math_utils.quat_apply(c_quat_inv, gp_pos - c_pos)
        gelpad_in_case_quat[side] = math_utils.quat_mul(c_quat_inv, gp_quat)
        # Debug: print the USD-authored case + gelpad poses for sanity.
        print(
            f"[DEBUG] {side}: case_usd pos={c_pos.squeeze().tolist()} quat={c_quat.squeeze().tolist()}"
        )
        print(
            f"[DEBUG] {side}: gelpad_usd pos={gp_pos.squeeze().tolist()} quat={gp_quat.squeeze().tolist()}"
        )

    # Solve one random gripper initialization target and apply the resulting
    # arm joints. The finger joints remain fixed at FRANKA_FINGER_INIT_POS.
    _solve_random_gripper_init_ik(robot, sim)

    _, initial_gripper_pos_w = _get_gripper_task_pos_w(robot)
    initial_gripper_pos_w = initial_gripper_pos_w.clone()
    print(f"[INFO] captured initial gripper midpoint pos_w={initial_gripper_pos_w.squeeze().tolist()} (return target)")

    realsense_camera.update(sim.get_physics_dt())
    if realsense_window is not None and "rgb" in realsense_camera.data.output:
        realsense_window.update_rgb(realsense_camera.data.output["rgb"])
    for tactile_sensor in tactile_sensors.values():
        tactile_sensor.update(dt=sim.get_physics_dt(), force_recompute=True)
        _enable_tactile_debug_windows(tactile_sensor)

    cam_pos_w = realsense_camera.data.pos_w
    cam_quat_w = realsense_camera.data.quat_w_world
    base_pos_w = robot.data.root_link_pos_w
    base_quat_w = robot.data.root_link_quat_w
    cam_pos_b, cam_quat_b = math_utils.subtract_frame_transforms(
        base_pos_w, base_quat_w, cam_pos_w, cam_quat_w
    )
    print(f"[INFO] spawned: {list(entities.keys())}")
    print(
        f"[INFO] realsense camera pos_w={realsense_camera.data.pos_w.squeeze().tolist()} "
        f"target_w={list(REALSENSE_CAMERA_TARGET_W)}"
    )
    print(
        f"[INFO] realsense debug RGB window created={realsense_window is not None}"
    )
    print(
        f"[INFO] camera extrinsic T_base_camera pos={cam_pos_b.squeeze().tolist()} "
        f"quat_wxyz_world={cam_quat_b.squeeze().tolist()}"
    )
    print(
        "[INFO] realsense camera outputs: "
        + ", ".join(f"{name}={tuple(data.shape)}" for name, data in realsense_camera.data.output.items())
    )
    tactile_output_shapes = []
    for side, sensor in tactile_sensors.items():
        tactile_output_shapes.extend(
            f"{side}.{name}={tuple(data.shape)}" for name, data in sensor.data.output.items()
        )
    print("[INFO] tactile sensor outputs: " + ", ".join(tactile_output_shapes))
    medium_gear: Articulation = entities["medium"]  # type: ignore[assignment]
    gear_base: Articulation = entities["gear_base"]  # type: ignore[assignment]
    print(f"[DEBUG] medium gear center pos_w={medium_gear.data.root_pos_w.squeeze().tolist()}")
    cos_yaw = float(np.cos(BASE_YAW))
    sin_yaw = float(np.sin(BASE_YAW))
    # CUROBO_GRASP_TARGET_X_OFFSET is tuned in the gear's local frame at
    # yaw=0; rotate it into world by BASE_YAW so the grasp tracks the gear
    # when the base is yawed. Z is unaffected by pure z-axis yaw.
    grasp_target_pos_w = medium_gear.data.root_pos_w.clone()
    grasp_target_pos_w[:, 0] += cos_yaw * CUROBO_GRASP_TARGET_X_OFFSET
    grasp_target_pos_w[:, 1] += sin_yaw * CUROBO_GRASP_TARGET_X_OFFSET
    grasp_target_pos_w[:, 2] += CUROBO_GRASP_TARGET_Z_OFFSET
    medium_dx, medium_dy, medium_dz = GEAR_BASE_OFFSETS["medium"]
    # Rotate the gear-base offset and the prep-X offset by BASE_YAW so the
    # translate target lands on the (rotated) peg location rather than the
    # unrotated reference frame.
    medium_dxr = cos_yaw * medium_dx - sin_yaw * medium_dy
    medium_dyr = sin_yaw * medium_dx + cos_yaw * medium_dy
    prep_x_dxr = cos_yaw * MEDIUM_GEAR_PREP_X_OFFSET
    prep_x_dyr = sin_yaw * MEDIUM_GEAR_PREP_X_OFFSET
    medium_lift_target_pos_w = grasp_target_pos_w.clone()
    medium_lift_target_pos_w[:, 2] = medium_gear.data.root_pos_w[:, 2] - medium_dz + MEDIUM_GEAR_PREP_Z_OFFSET
    medium_translate_target_pos_w = medium_lift_target_pos_w.clone()
    medium_translate_target_pos_w[:, 0] = medium_gear.data.root_pos_w[:, 0] - medium_dxr + prep_x_dxr
    medium_translate_target_pos_w[:, 1] = medium_gear.data.root_pos_w[:, 1] - medium_dyr + prep_x_dyr
    medium_lower_target_pos_w = medium_translate_target_pos_w.clone()
    medium_lower_target_pos_w[:, 2] -= MEDIUM_GEAR_LOWER_Z_OFFSET
    print(
        f"[INFO] medium gear lower target pos_w={medium_lower_target_pos_w.squeeze().tolist()} "
        f"with z_offset=-{MEDIUM_GEAR_LOWER_Z_OFFSET}"
    )
    print(
        f"[INFO] medium gear lift target pos_w={medium_lift_target_pos_w.squeeze().tolist()} "
        f"with z_offset={MEDIUM_GEAR_PREP_Z_OFFSET}"
    )
    print(
        f"[INFO] medium gear translate target pos_w={medium_translate_target_pos_w.squeeze().tolist()} "
        f"by removing GEAR_BASE_OFFSETS[medium]={(medium_dx, medium_dy, medium_dz)} "
        f"and adding x_offset={MEDIUM_GEAR_PREP_X_OFFSET}"
    )
    # All segment targets are ground-truth-known up front — build the cuRobo
    # planner once (paying the warmup cost now) and plan every segment back to
    # back, chaining each plan's final joint config as the next plan's start.
    # Phase transitions in the sim loop then just swap in a precomputed
    # trajectory and don't stall on solve+warmup.
    planner, root_pose = _build_curobo_planner(robot, sim)
    t_hand_fingertip, _ = _capture_hand_fingertip_transform(robot)
    arm_n = robot.num_joints - 2
    pre_plan_start_joint_pos = robot.data.joint_pos[0, :arm_n].detach().clone()
    segment_targets = [
        ("medium_grasp", grasp_target_pos_w),
        ("medium_lift", medium_lift_target_pos_w),
        ("medium_translate", medium_translate_target_pos_w),
        ("medium_lower", medium_lower_target_pos_w),
        ("medium_return", initial_gripper_pos_w),
    ]
    precomputed_positions: dict[str, torch.Tensor] = {}
    for label, target in segment_targets:
        plan = _plan_panda_hand_to_target(
            robot,
            planner,
            root_pose,
            t_hand_fingertip,
            pre_plan_start_joint_pos,
            target,
            label,
        )
        if not plan.success.item():
            print(
                f"[WARN] precompute halted at segment={label}; later segments will be skipped"
            )
            break
        positions = plan.interpolated_plan.position.to(device=robot.device, dtype=torch.float32)
        precomputed_positions[label] = positions
        pre_plan_start_joint_pos = positions[-1].clone()
    print(
        f"[INFO] precomputed plans available for: {list(precomputed_positions.keys())}"
    )

    arm_joint_ids = torch.tensor(
        [robot.joint_names.index(f"panda_joint{i}") for i in range(1, 8)],
        device=robot.device,
        dtype=torch.long,
    )
    finger_joint_ids = torch.tensor(
        [idx for idx, name in enumerate(robot.joint_names) if name.startswith("panda_finger")],
        device=robot.device,
        dtype=torch.long,
    )
    curobo_positions = precomputed_positions.get("medium_grasp")
    curobo_step = 0
    curobo_phase = "medium_grasp"
    curobo_done_printed = False
    grasp_close_step = 0
    grasp_close_done_printed = False
    grasp_open_step = 0
    grasp_open_done_printed = False
    lift_plan_started = False
    translate_plan_started = False
    lower_plan_started = False
    return_plan_started = False
    grasp_arrival_wait_printed = False
    lift_arrival_wait_printed = False
    translate_arrival_wait_printed = False
    lower_arrival_wait_printed = False
    if curobo_positions is not None:
        print(
            f"[INFO] executing cuRobo {curobo_phase} trajectory in sim with "
            f"{curobo_positions.shape[0]} waypoints"
        )
    print(
        "[INFO] gelpad↔case offsets captured from USD: "
        + ", ".join(
            f"{side}={gelpad_in_case_pos[side].squeeze().tolist()}" for side in ("left", "right")
        )
    )
    # Sanity-check: PhysX-driven link world pose. Should reflect the IK pose
    # even though the GUI gizmo for panda_link8 may still show the zero-pose
    # location (Fabric-vs-USD discrepancy).
    panda_hand_idx = robot.find_bodies("panda_hand")[0][0]
    panda_link8_idx = robot.find_bodies("panda_link8")[0][0]
    print(
        f"[DEBUG] panda_link8 world pos (from PhysX): "
        f"{robot.data.body_link_pos_w[:, panda_link8_idx].squeeze().tolist()}"
    )
    print("[INFO] close the Isaac Sim window (or Ctrl+C in terminal) to exit.")

    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        if curobo_positions is not None:
            if curobo_step < curobo_positions.shape[0]:
                robot.set_joint_position_target(
                    curobo_positions[curobo_step].unsqueeze(0), joint_ids=arm_joint_ids
                )
                curobo_step += 1
            elif not curobo_done_printed:
                if curobo_phase == "medium_grasp":
                    task_frame_name, task_pos_w = _get_gripper_task_pos_w(robot)
                    grasp_arrival_error = (task_pos_w - grasp_target_pos_w).norm(dim=-1).item()
                    if grasp_arrival_error < GRASP_ARRIVAL_POS_TOL:
                        print(
                            f"[INFO] cuRobo medium_grasp reached {task_frame_name}; "
                            f"position error={grasp_arrival_error:.6f} m, closing gripper"
                        )
                        print(
                            f"[INFO] cuRobo actual final panda_hand pos_w="
                            f"{robot.data.body_link_pos_w[:, panda_hand_idx].squeeze().tolist()}"
                        )
                        curobo_done_printed = True
                    elif not grasp_arrival_wait_printed:
                        print(
                            f"[INFO] cuRobo medium_grasp waypoints sent; waiting for {task_frame_name} "
                            f"arrival error < {GRASP_ARRIVAL_POS_TOL} m "
                            f"(current={grasp_arrival_error:.6f} m)"
                        )
                        grasp_arrival_wait_printed = True
                elif curobo_phase == "medium_lift":
                    task_frame_name, task_pos_w = _get_gripper_task_pos_w(robot)
                    lift_arrival_error = (task_pos_w - medium_lift_target_pos_w).norm(dim=-1).item()
                    if lift_arrival_error < MEDIUM_GEAR_LIFT_POS_TOL:
                        print(
                            f"[INFO] cuRobo medium_lift reached {task_frame_name}; "
                            f"position error={lift_arrival_error:.6f} m, starting horizontal translate"
                        )
                        curobo_done_printed = True
                    elif not lift_arrival_wait_printed:
                        print(
                            f"[INFO] cuRobo medium_lift waypoints sent; waiting for {task_frame_name} "
                            f"arrival error < {MEDIUM_GEAR_LIFT_POS_TOL} m "
                            f"(current={lift_arrival_error:.6f} m)"
                        )
                        lift_arrival_wait_printed = True
                elif curobo_phase == "medium_translate":
                    task_frame_name, task_pos_w = _get_gripper_task_pos_w(robot)
                    translate_arrival_error = (task_pos_w - medium_translate_target_pos_w).norm(dim=-1).item()
                    if translate_arrival_error < MEDIUM_GEAR_CONTACT_MOVE_TOL:
                        print(
                            f"[INFO] cuRobo medium_translate reached {task_frame_name}; "
                            f"position error={translate_arrival_error:.6f} m, starting vertical lowering"
                        )
                        curobo_done_printed = True
                    elif not translate_arrival_wait_printed:
                        print(
                            f"[INFO] cuRobo medium_translate waypoints sent; waiting for {task_frame_name} "
                            f"arrival error < {MEDIUM_GEAR_CONTACT_MOVE_TOL} m "
                            f"(current={translate_arrival_error:.6f} m)"
                        )
                        translate_arrival_wait_printed = True
                elif curobo_phase == "medium_lower":
                    task_frame_name, task_pos_w = _get_gripper_task_pos_w(robot)
                    lower_arrival_error = (task_pos_w - medium_lower_target_pos_w).norm(dim=-1).item()
                    if lower_arrival_error < MEDIUM_GEAR_LIFT_POS_TOL:
                        print(
                            f"[INFO] cuRobo medium_lower reached {task_frame_name}; "
                            f"position error={lower_arrival_error:.6f} m, opening gripper"
                        )
                        curobo_done_printed = True
                    elif not lower_arrival_wait_printed:
                        print(
                            f"[INFO] cuRobo medium_lower waypoints sent; waiting for {task_frame_name} "
                            f"arrival error < {MEDIUM_GEAR_LIFT_POS_TOL} m "
                            f"(current={lower_arrival_error:.6f} m)"
                        )
                        lower_arrival_wait_printed = True
                else:
                    print("[INFO] cuRobo medium_return trajectory execution finished; holding final arm target")
                    print(
                        f"[INFO] cuRobo actual final panda_hand pos_w="
                        f"{robot.data.body_link_pos_w[:, panda_hand_idx].squeeze().tolist()}"
                    )
                    curobo_done_printed = True
        if curobo_phase == "medium_grasp" and curobo_done_printed and grasp_close_step < GRASP_CLOSE_STEPS:
            alpha = float(grasp_close_step + 1) / float(GRASP_CLOSE_STEPS)
            finger_pos = FRANKA_FINGER_INIT_POS + alpha * (FRANKA_FINGER_GRASP_POS - FRANKA_FINGER_INIT_POS)
            finger_targets = torch.full(
                (1, finger_joint_ids.numel()),
                finger_pos,
                device=robot.device,
                dtype=torch.float32,
            )
            robot.set_joint_position_target(finger_targets, joint_ids=finger_joint_ids)
            grasp_close_step += 1
        elif curobo_phase == "medium_grasp" and curobo_done_printed and not grasp_close_done_printed:
            print(f"[INFO] gripper closed to panda_finger target={FRANKA_FINGER_GRASP_POS}")
            grasp_close_done_printed = True
        if grasp_close_done_printed:
            closed_finger_targets = torch.full(
                (1, finger_joint_ids.numel()),
                FRANKA_FINGER_GRASP_POS,
                device=robot.device,
                dtype=torch.float32,
            )
            robot.set_joint_position_target(closed_finger_targets, joint_ids=finger_joint_ids)
        if grasp_close_done_printed and not lift_plan_started:
            lift_plan_started = True
            lift_positions = precomputed_positions.get("medium_lift")
            if lift_positions is not None:
                curobo_phase = "medium_lift"
                curobo_positions = lift_positions
                curobo_step = 0
                curobo_done_printed = False
                print(
                    f"[INFO] executing cuRobo {curobo_phase} trajectory in sim with "
                    f"{curobo_positions.shape[0]} waypoints"
                )
            else:
                print("[WARN] cuRobo medium_lift plan unavailable; holding grasp pose")
        if curobo_phase == "medium_lift" and curobo_done_printed and not translate_plan_started:
            translate_plan_started = True
            translate_positions = precomputed_positions.get("medium_translate")
            if translate_positions is not None:
                curobo_phase = "medium_translate"
                curobo_positions = translate_positions
                curobo_step = 0
                curobo_done_printed = False
                print(
                    f"[INFO] executing cuRobo {curobo_phase} trajectory in sim with "
                    f"{curobo_positions.shape[0]} waypoints"
                )
            else:
                print("[WARN] cuRobo medium_translate plan unavailable; holding lift pose")
        if curobo_phase == "medium_translate" and curobo_done_printed and not lower_plan_started:
            lower_plan_started = True
            lower_positions = precomputed_positions.get("medium_lower")
            if lower_positions is not None:
                curobo_phase = "medium_lower"
                curobo_positions = lower_positions
                curobo_step = 0
                curobo_done_printed = False
                print(
                    f"[INFO] executing cuRobo {curobo_phase} trajectory in sim with "
                    f"{curobo_positions.shape[0]} waypoints"
                )
            else:
                print("[WARN] cuRobo medium_lower plan unavailable; holding translate pose")
        if curobo_phase == "medium_lower" and curobo_done_printed and grasp_open_step < GRASP_OPEN_STEPS:
            alpha = float(grasp_open_step + 1) / float(GRASP_OPEN_STEPS)
            finger_pos = FRANKA_FINGER_GRASP_POS + alpha * (FRANKA_FINGER_INIT_POS - FRANKA_FINGER_GRASP_POS)
            finger_targets = torch.full(
                (1, finger_joint_ids.numel()),
                finger_pos,
                device=robot.device,
                dtype=torch.float32,
            )
            robot.set_joint_position_target(finger_targets, joint_ids=finger_joint_ids)
            grasp_open_step += 1
        elif curobo_phase == "medium_lower" and curobo_done_printed and not grasp_open_done_printed:
            print(f"[INFO] gripper opened to panda_finger target={FRANKA_FINGER_INIT_POS}")
            grasp_open_done_printed = True
        if grasp_open_done_printed:
            opened_finger_targets = torch.full(
                (1, finger_joint_ids.numel()),
                FRANKA_FINGER_INIT_POS,
                device=robot.device,
                dtype=torch.float32,
            )
            robot.set_joint_position_target(opened_finger_targets, joint_ids=finger_joint_ids)
        if grasp_open_done_printed and not return_plan_started:
            return_plan_started = True
            return_positions = precomputed_positions.get("medium_return")
            if return_positions is not None:
                curobo_phase = "medium_return"
                curobo_positions = return_positions
                curobo_step = 0
                curobo_done_printed = False
                print(
                    f"[INFO] executing cuRobo {curobo_phase} trajectory in sim with "
                    f"{curobo_positions.shape[0]} waypoints"
                )
            else:
                print("[WARN] cuRobo medium_return plan unavailable; holding lower pose")
        robot.write_data_to_sim()
        sim.step()
        sim.render()
        robot.update(sim_dt)
        realsense_camera.update(sim_dt)
        if realsense_window is not None and "rgb" in realsense_camera.data.output:
            realsense_window.update_rgb(realsense_camera.data.output["rgb"])
        for tactile_sensor in tactile_sensors.values():
            tactile_sensor.update(dt=sim_dt, force_recompute=True)
        # gelpad_world = case_world ∘ gelpad_in_case   (read case pose via the
        # same XFormPrim source we used at capture time → no frame mismatch).
        for side in ("left", "right"):
            c_pos, c_quat = case_prims[side].get_world_poses()
            gp_pos = c_pos + math_utils.quat_apply(c_quat, gelpad_in_case_pos[side])
            gp_quat = math_utils.quat_mul(c_quat, gelpad_in_case_quat[side])
            gelpad_prims[side].set_world_poses(positions=gp_pos, orientations=gp_quat)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
