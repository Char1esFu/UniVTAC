"""Gear-assembly task: pick up the medium factory gear and place it on its peg.

Scene
-----
  gear_base   : UIPC AffineBody, kinematic (bolted to the table); anchors the
                three pegs and is the surface the medium gear rests on.
  gear_small  : UIPC AffineBody, dynamic; sits on its peg, free to spin so it
                can mesh with the medium gear.
  gear_large  : UIPC AffineBody, dynamic; sits on its peg, free to spin so it
                can mesh with the medium gear.
  gear_medium : UIPC AffineBody, dynamic; staged Y=+0.15 m aside — pick-and-place
                target.

All four gears live in UIPC. This is required (not just for the gelpad/tactile
contact, which only touches gear_medium) so that the grasped medium gear
physically rests on gear_base, slides onto its peg, and meshes with the
neighbouring small/large gears — UIPC contact only happens between UIPC bodies.
It also makes every gear visible to cuRobo, which builds its collision world
from the UIPC actors (envs/robot/curobo_planner.py::get_curr_world_cfg), so the
arm avoids them while planning.

Each gear is tetrahedralized once by float-tetwild on first load and the tet
data is cached back into its USD (see uipc_object.py), so subsequent runs are
fast.

Episode structure
-----------------
  pre_move   — grasp the medium gear from its staged position.
  _play_once — transport up-and-over to a standoff above the peg → plain cuRobo
               plan down to the seated pose → release.

Notes
-----
  Gear USDs must have been re-centred by scripts/usd_related/recenter_gears.py
  so each USD origin coincides with the gear's geometry bbox centre.
"""

from ._base_task import *
import numpy as np
import torch


# ── Scene geometry ─────────────────────────────────────────────────────────────

GEAR_BASE_POS = (0.4, 0.0, 0.005)

GEAR_BASE_OFFSETS: dict[str, tuple[float, float, float]] = {
    "small":  ( 0.05075, 0.0,  0.0175),
    "medium": ( 0.02025, 0.15, 0.01),
    "large":  (-0.03025, 0.0,  0.0175),
}

MEDIUM_PEG_OFFSET = (0.02025, 0.0, 0.0175)

GRASP_Z_MIN = 0.001
GRASP_Z_MAX = 0.002
LIFT_Z      = 0.05

# UIPC refuses to start if any two surfaces interpenetrate at the initial frame
# (SimplicialSurfaceIntersectionCheck). The gears sit flush on the base pegs, so
# spawn the dynamic neighbours with a small vertical clearance and let them drop
# onto their pegs during the reset settling steps.
GEAR_SEAT_CLEARANCE = 0.0001

SUCCESS_XY_TOL = 0.005
SUCCESS_Z_MAX  = 0.025


# ── Config ─────────────────────────────────────────────────────────────────────

@configclass
class TaskCfg(BaseTaskCfg):
    pass


# ── Task ───────────────────────────────────────────────────────────────────────

class Task(BaseTask):

    # ── scene setup ───────────────────────────────────────────────────────────

    def create_actors(self):
        """Register all four gears with UIPC.

        gear_base is kinematic (fixed to the world); gear_small/large/medium are
        dynamic affine rigid bodies. small/large sit on their pegs and stay free
        to spin so the medium gear can mesh with them during assembly.
        """
        bx, by, bz = GEAR_BASE_POS

        self.gear_base = self._actor_manager.add_from_usd_file(
            name='gear_base',
            asset_path="factory_gear_base.usd",
            pose=Pose([bx, by, bz], [1, 0, 0, 0]),
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(kinematic=True),
        )

        for size in ('small', 'large'):
            dx, dy, dz = GEAR_BASE_OFFSETS[size]
            setattr(self, f'gear_{size}', self._actor_manager.add_from_usd_file(
                name=f'gear_{size}',
                asset_path=f"factory_gear_{size}.usd",
                pose=Pose([bx + dx, by + dy, bz + dz + GEAR_SEAT_CLEARANCE], [1, 0, 0, 0]),
            ))

        mx, my, mz = GEAR_BASE_OFFSETS['medium']
        self.gear_medium = self._actor_manager.add_from_usd_file(
            name='gear_medium',
            asset_path="factory_gear_medium.usd",
            pose=Pose([bx + mx, by + my, bz + mz], [1, 0, 0, 0]),
        )

    # ── reset ─────────────────────────────────────────────────────────────────

    def _reset_actors(self):
        bx, by, bz = GEAR_BASE_POS
        self._set_gripper_fully_open()

        # re-seat the dynamic neighbours on their pegs (they may have been
        # nudged during the previous episode); gear_base is kinematic.
        for size in ('small', 'large'):
            dx, dy, dz = GEAR_BASE_OFFSETS[size]
            getattr(self, f'gear_{size}').set_pose(
                Pose([bx + dx, by + dy, bz + dz + GEAR_SEAT_CLEARANCE], [1, 0, 0, 0])
            )

        mx, my, mz = GEAR_BASE_OFFSETS['medium']
        noise = self.create_noise([0.01, 0.01, 0.0])
        self.gear_medium.set_pose(
            Pose([bx + mx, by + my, bz + mz], [1, 0, 0, 0]).add_offset(noise)
        )

    def _set_gripper_fully_open(self):
        qpos = torch.full(
            (2,),
            self._robot_manager.gripper_max_qpos,
            device=self._robot_manager.device,
        )
        qvel = torch.zeros_like(qpos)
        self._robot_manager.set_gripper(qpos, qvel)

    # ── pre-episode: grasp the staged medium gear ─────────────────────────────

    def pre_move(self):
        self.delay(1)

        grasp_z = self.rng.uniform(GRASP_Z_MIN, GRASP_Z_MAX)
        target_pose = self.gear_medium.get_pose().add_bias([0.0, 0.0, grasp_z])
        cpose = construct_grasp_pose(target_pose.p, [0, 0, 1], [1, 0, 0])
        cid = self.gear_medium.register_point(cpose, type='contact')

        self.move(self.atom.grasp_actor(
            self.gear_medium,
            contact_point_id=cid,
            pre_dis=0.03,
            dis=0.0,
            is_close=False,
        ))
        self.move(self.atom.close_gripper(0.0))

        self.origin_inhand_pose = self.gear_medium.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose()
        )

    # ── episode: transport over peg → descend → wiggle → release ──────────────

    def _play_once(self):
        bx, by, bz = GEAR_BASE_POS
        mx, my, mz = MEDIUM_PEG_OFFSET
        peg_xy   = np.array([bx + mx, by + my])
        seat_z   = bz + mz

        # 1. transport: one plan that lifts up and over to a standoff above the
        #    peg axis, merging the old lift + horizontal translate. cuRobo curves
        #    up to the lift height first, then moves horizontally over the peg.
        #    The endpoint carries some open-loop xy error — that's fine, the
        #    insertion segment re-reads the gear and corrects it.
        gear_pose = self.gear_medium.get_pose()
        dx, dy = (peg_xy - gear_pose.p[:2])
        self.move(
            self.atom.move_by_displacement(x=float(dx), y=float(dy), z=LIFT_Z),
            tag='peg_transport',
        )

        # 2. insert: re-read the gear's actual pose and re-target the ground-truth
        #    seated pose with a plain cuRobo plan (no pre_dis straight descent).
        #    cuRobo finds its own path down to the lower pose. Slow so the dynamic
        #    small/large gears have time to rotate and accept the medium gear.
        gear_pose = self.gear_medium.get_pose()
        dx, dy = (peg_xy - gear_pose.p[:2])
        descend = float(gear_pose.p[2] - seat_z)
        actions = self.atom.move_by_displacement(x=float(dx), y=float(dy), z=-descend)
        self.move(actions, time_dilation_factor=0.3, tag='peg_insert')

        # 3. release and let it settle onto the peg
        self.move(self.atom.open_gripper())
        self.delay(20, is_save=False)

    # ── success criterion ─────────────────────────────────────────────────────

    def check_success(self) -> bool:
        bx, by, bz = GEAR_BASE_POS
        mx, my, mz = MEDIUM_PEG_OFFSET
        target_xy = np.array([bx + mx, by + my])

        gear_pose = self.gear_medium.get_pose()
        xy_err    = np.linalg.norm(gear_pose.p[:2] - target_xy)
        z_ok      = gear_pose.p[2] < bz + mz + SUCCESS_Z_MAX

        return bool(xy_err < SUCCESS_XY_TOL and z_ok)
