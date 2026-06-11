"""Gear-assembly task: pick up the medium factory gear and place it on its peg.

All four gears live in UIPC so the grasped medium gear can rest on the base,
slide onto its peg and mesh with the small/large gears — UIPC contact only
happens between UIPC bodies. Episode: grasp (pre_move) → transport over the
peg → closed-loop servo descent onto the peg axis → wiggle to mesh the teeth
→ seat → release.

Gear USDs must be re-centred (scripts/usd_related/recenter_gears.py) so each
USD origin coincides with the geometry bbox centre.
"""

from ._base_task import *
import numpy as np
import torch
import transforms3d as t3d


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

# Axial face width of the toothed rim. The descent stops this far above the
# seated pose: pressing to full depth jams tooth tips before they can mesh.
GEAR_TOOTH_WIDTH = 0.010

GEAR_HALF_HEIGHT = 0.0125  # gear is 0.025 m tall, USD origin at bbox centre

# Servo descent: re-read the gear each step and re-correct toward the peg
# axis, so the compliant grasp's drift cannot accumulate and short plans
# cannot bow into the neighbouring gears.
INSERT_STEP_Z    = 0.01    # depth per step
INSERT_MAX_STEPS = 12
INSERT_XY_TOL    = 0.001   # axis tolerance required to finish the descent
INSERT_STALL_MIN = 0.0005  # smaller per-step gain ⇒ tooth-tip jam, go wiggle

# Wiggle: twist about the gear axis, then either pry the gear level (tilt
# above threshold ⇒ one side meshed, the other on a tooth tip) or press on.
WIGGLE_MAX_STEPS    = 8
WIGGLE_YAW_DEG      = 9.0     # one tooth pitch (40 teeth, from the USD)
WIGGLE_TILT_MAX_DEG = 3.0
WIGGLE_PRESS_STEP   = 0.001
WIGGLE_SEAT_TOL     = 0.0005

# UIPC rejects interpenetrating initial states — spawn the neighbour gears
# slightly above their pegs and let them settle during reset.
GEAR_SEAT_CLEARANCE = 0.0001

SUCCESS_XY_TOL = 0.005
SUCCESS_Z_MAX  = 0.025


# ── Config ─────────────────────────────────────────────────────────────────────

@configclass
class TaskCfg(BaseTaskCfg):
    pass


# ── Task ───────────────────────────────────────────────────────────────────────

class Task(BaseTask):

    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        # UIPC default friction (0.5) lets the gear shear out of the gelpads;
        # 2.5 matches the other grasping tasks. Global — also raises gear↔peg
        # friction, so lower it if the released gear stops sliding/meshing.
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    # ── scene setup ───────────────────────────────────────────────────────────

    def create_actors(self):
        """gear_base is kinematic; the gears are dynamic so they can spin/mesh."""
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

        # re-seat the neighbours (the previous episode may have nudged them)
        for size in ('small', 'large'):
            dx, dy, dz = GEAR_BASE_OFFSETS[size]
            getattr(self, f'gear_{size}').set_pose(
                Pose([bx + dx, by + dy, bz + dz + GEAR_SEAT_CLEARANCE], [1, 0, 0, 0])
            )

        mx, my, mz = GEAR_BASE_OFFSETS['medium']
        noise = self.create_noise([0.005, 0.005, 0.0])
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

        # the grasped gear moves with the hand and meshing contact is wanted —
        # keep only gear_base as a cuRobo obstacle while it is held
        self._robot_manager.planner.ignore_actors.update(
            {'gear_medium', 'gear_small', 'gear_large'})

        self.origin_inhand_pose = self.gear_medium.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose()
        )

    # ── episode helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _level_gear_quat(gear_pose: Pose) -> np.ndarray:
        """Level orientation: shortest arc taking the gear axis to world +z."""
        z_now = t3d.quaternions.quat2mat(gear_pose.q)[:, 2]
        axis = np.cross(z_now, [0.0, 0.0, 1.0])
        sin_a = np.linalg.norm(axis)
        if sin_a > 1e-8:
            q_fix = t3d.quaternions.axangle2quat(axis / sin_a, np.arctan2(sin_a, z_now[2]))
        else:
            q_fix = np.array([1.0, 0.0, 0.0, 0.0])
        return t3d.quaternions.qmult(q_fix, gear_pose.q)

    def _measure_medium_peg(self) -> tuple[np.ndarray, float]:
        """Peg axis xy and top z, measured from the base mesh.

        MEDIUM_PEG_OFFSET only windows the search; the xy bbox centre of the
        shaft vertices is the axis (centroid would be biased by uneven vertex
        density).
        """
        bx, by, bz = GEAR_BASE_POS
        mx, my, mz = MEDIUM_PEG_OFFSET
        expected = np.array([bx + mx, by + my])
        verts = np.asarray(self.gear_base.vertices)
        lateral = np.linalg.norm(verts[:, :2] - expected, axis=1)
        shaft = verts[(lateral < 0.010) & (verts[:, 2] > bz + mz)]
        if len(shaft) == 0:
            self.logger.warning(
                f'medium peg shaft not found near xy={expected} — using '
                f'MEDIUM_PEG_OFFSET as-is, keeping levelling on all the way')
            return expected, np.inf
        measured = (shaft[:, :2].min(axis=0) + shaft[:, :2].max(axis=0)) / 2
        top_z = float(shaft[:, 2].max())
        self.logger.info(
            f'medium peg axis xy: measured ({measured[0]:.4f}, {measured[1]:.4f}) m, '
            f'configured ({expected[0]:.4f}, {expected[1]:.4f}) m, '
            f'delta {np.linalg.norm(measured - expected) * 1000:.2f} mm, '
            f'top z {top_z:.4f} m ({len(shaft)} shaft vertices)'
        )
        return measured, top_z

    def _debug_vis_peg_axis(self, peg_xy: np.ndarray):
        """Column of green markers along the medium peg axis (--debug_vis)."""
        if not self.cfg.debug_vis:
            return
        bz = GEAR_BASE_POS[2]
        for i, z in enumerate(np.linspace(bz, bz + 0.08, 17)):
            add_visual_box(
                Pose([peg_xy[0], peg_xy[1], float(z)], [1, 0, 0, 0]),
                name=f'medium_peg_axis_{i}',
                size=0.0015,
                color=np.array([0.0, 255.0, 0.0]),
            )

    def _bottom_face_tilt(self) -> tuple[float, np.ndarray]:
        """Bottom-face tilt (deg) and its lowest point (= meshed side, pry pivot)."""
        verts = np.asarray(self.gear_medium.vertices)
        axis = self.gear_medium.get_pose().to_transformation_matrix()[:3, 2]
        proj = verts @ axis
        ring = verts[proj < proj.min() + 5e-4]
        low_pt = ring[np.argmin(ring[:, 2])]
        high_pt = ring[np.argmax(ring[:, 2])]
        chord = high_pt - low_pt
        run = np.linalg.norm(chord[:2])
        tilt_deg = float(np.degrees(np.arctan2(chord[2], run))) if run > 1e-9 else 0.0
        return tilt_deg, low_pt

    def _pry_gear_level(self, low_pt: np.ndarray):
        """Rotate the gear level about its lowest point, so the meshed side
        stays engaged while the high side swings down."""
        gear_pose = self.gear_medium.get_pose()
        q_level = self._level_gear_quat(gear_pose)
        r_fix = t3d.quaternions.quat2mat(
            t3d.quaternions.qmult(q_level, t3d.quaternions.qinverse(gear_pose.q)))
        gear_target = Pose(low_pt + r_fix @ (gear_pose.p - low_pt), q_level)
        ee_target = self.atom.get_arm_pose().rebase(to_coord=gear_pose).rebase(from_coord=gear_target)
        self.move(self.atom.move_to_pose(ee_target), time_dilation_factor=0.3, tag='peg_pry')

    # ── episode ───────────────────────────────────────────────────────────────

    def _play_once(self):
        peg_xy, peg_top_z = self._measure_medium_peg()
        seat_z = GEAR_BASE_POS[2] + MEDIUM_PEG_OFFSET[2]
        self._debug_vis_peg_axis(peg_xy)

        # 1. transport: lift up and over to a standoff above the peg
        gear_pose = self.gear_medium.get_pose()
        dx, dy = (peg_xy - gear_pose.p[:2])
        self.move(
            self.atom.move_by_displacement(x=float(dx), y=float(dy), z=LIFT_Z),
            tag='peg_transport',
        )

        # 2. insert: servo descent to one tooth width above the seated height.
        #    Gear-centric with re-levelling while above the peg top (free
        #    space, fast); translation-only and slower once the hole is on the
        #    shaft. delay=False: the next step re-reads the gear anyway.
        insert_z = seat_z + GEAR_TOOTH_WIDTH
        for _ in range(INSERT_MAX_STEPS):
            gear_pose = self.gear_medium.get_pose()
            if self.cfg.debug_vis:
                # bottom-face centre (blue) vs the green peg-axis column
                add_visual_box(
                    gear_pose.add_bias([0.0, 0.0, -GEAR_HALF_HEIGHT]),
                    name='gear_bottom_centre',
                    size=0.002,
                    color=np.array([0.0, 0.0, 255.0]),
                )
            xy_err = float(np.linalg.norm(peg_xy - gear_pose.p[:2]))
            at_depth = gear_pose.p[2] - insert_z < 1e-4
            if at_depth and xy_err < INSERT_XY_TOL:
                break

            target_z = max(gear_pose.p[2] - INSERT_STEP_Z, insert_z)
            on_shaft = (gear_pose.p[2] - GEAR_HALF_HEIGHT) < peg_top_z
            pre_z = gear_pose.p[2]
            if on_shaft:
                dx, dy = peg_xy - gear_pose.p[:2]
                self.move(
                    self.atom.move_by_displacement(
                        x=float(dx), y=float(dy), z=float(target_z - gear_pose.p[2])),
                    time_dilation_factor=0.3,
                    constraint_pose=[1, 1, 1, 0, 0, 0],
                    tag='peg_insert',
                    delay=False,
                )
            else:
                gear_target = Pose(
                    [peg_xy[0], peg_xy[1], target_z],
                    self._level_gear_quat(gear_pose),
                )
                ee_target = self.atom.get_arm_pose().rebase(to_coord=gear_pose).rebase(from_coord=gear_target)
                self.move(
                    self.atom.move_to_pose(ee_target),
                    time_dilation_factor=0.6,
                    tag='peg_insert',
                    delay=False,
                )
            if not self.plan_success:
                return
            # no descent gained ⇒ tooth-tip jam, only the wiggle resolves it
            if on_shaft and pre_z - self.gear_medium.get_pose().p[2] < INSERT_STALL_MIN:
                break

        # 3. wiggle: alternate one-tooth-pitch twists; after each twist either
        #    pry the gear level or press deeper. Constraints free exactly the
        #    axes each plan moves along (hold_partial_pose needs the held
        #    components to match the start).
        for i in range(WIGGLE_MAX_STEPS):
            yaw_deg = WIGGLE_YAW_DEG if i % 2 == 0 else -WIGGLE_YAW_DEG
            # rpy_coord='local' yaws in place; 'world' would orbit the origin
            self.move(
                self.atom.move_by_displacement(
                    rpy=[0.0, 0.0, float(np.deg2rad(yaw_deg))],
                ),
                time_dilation_factor=0.3,
                constraint_pose=[0, 0, 0, 1, 1, 1],
                tag='peg_wiggle',
            )

            tilt_deg, low_pt = self._bottom_face_tilt()
            if tilt_deg > WIGGLE_TILT_MAX_DEG:
                self._pry_gear_level(low_pt)
            else:
                gear_pose = self.gear_medium.get_pose()
                dx, dy = peg_xy - gear_pose.p[:2]
                dz = max(gear_pose.p[2] - WIGGLE_PRESS_STEP, seat_z) - gear_pose.p[2]
                self.move(
                    self.atom.move_by_displacement(x=float(dx), y=float(dy), z=float(dz)),
                    time_dilation_factor=0.3,
                    constraint_pose=[1, 1, 1, 0, 0, 0],
                    tag='peg_wiggle_press',
                )

            if not self.plan_success:
                return
            if self.gear_medium.get_pose().p[2] - seat_z < WIGGLE_SEAT_TOL:
                break

        # 4. seat: press down to the seated height
        gear_pose = self.gear_medium.get_pose()
        dx, dy = peg_xy - gear_pose.p[:2]
        dz = seat_z - gear_pose.p[2]
        self.move(
            self.atom.move_by_displacement(x=float(dx), y=float(dy), z=float(dz)),
            time_dilation_factor=0.3,
            constraint_pose=[1, 1, 1, 0, 0, 0],
            tag='peg_seat',
        )

        # 5. release; the gears become cuRobo obstacles again
        self.move(self.atom.open_gripper())
        self._robot_manager.planner.ignore_actors.difference_update(
            {'gear_medium', 'gear_small', 'gear_large'})
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
