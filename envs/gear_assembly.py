from ._base_task import *
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Gear-assembly data-collection task.
#
# Assets (assets/objects/, loaded via ActorManager.add_from_usd_file which
# resolves paths relative to OBJECTS_ROOT):
#   factory_gear_base.usd    – base plate with three vertical pegs
#   factory_gear_small.usd    – small gear (authored on its peg)
#   factory_gear_medium.usd   – medium gear (authored on its peg) → manipulated
#   factory_gear_large.usd    – large gear (authored on its peg)
#
# Key geometric fact (measured from each USD's bounding box): every gear USD is
# authored in the *base's* coordinate frame, sitting at its own peg location.
# The Factory peg offsets in the base-local frame are
#     small=(0.05075, 0, 0), medium=(0.02025, 0, 0), large=(-0.03025, 0, 0).
# Consequences we exploit:
#   • Spawning base + small + large at the *same* pose places the two side
#     gears onto their pegs automatically — no per-gear offset math.
#   • The medium gear's fully-assembled target pose is simply the base's pose
#     (its USD origin coincides with the base origin when seated). So success =
#     medium_gear.get_pose() ≈ gear_base.get_pose().
#
# Lifecycle (mirrors insert_HDMI / insert_hole; only _play_once is *saved* —
# the grasp in pre_move runs with in_pre_move=True and is never recorded, which
# is why, like every other task, the recorded episode begins with the gear
# already pinched between the GelSight pads):
#   create_actors()  → spawn base/small/large (base frame) + medium (parked)
#   _reset_actors()  → randomize base pose (carries side gears + pegs) + medium
#   pre_move()       → grasp medium top-down, lift, hover above its peg  [unsaved]
#   _play_once()     → cuRobo-planned vertical insertion onto the medium peg [saved]
# ─────────────────────────────────────────────────────────────────────────────

# Base plate spawn origin (USD bottom face at local z=0). Sits just above the
# plate top like the insert-task slots (which spawn at z=0.002).
GEAR_BASE_POS = (0.5, 0.0, 0.002)

# Densities. Heavy base/side gears so they stay put under insertion contact
# loads (same trick the insert tasks use for the static slot).
BASE_DENSITY = 1e5
SIDE_GEAR_DENSITY = 1e4

# Medium gear parking spot on the table, in front of the robot and clear of the
# base, so the top-down grasp plans collision-free. USD bottom face is at local
# z=0.005, so a small spawn height lets it settle flat onto the plate.
MEDIUM_PARK_POS = (0.40, -0.16, 0.0)

# Medium gear geometric centre relative to its USD spawn origin, from the USD
# bounding box (x∈[-0.0007,0.0412], y∈[-0.021,0.021], z∈[0.005,0.03]):
# centre ≈ (0.0203, 0.0, 0.0175), mid-thickness at local z=0.0175.
MEDIUM_CENTER_LOCAL = (0.0203, 0.0, 0.0175)

# Insertion standoffs (along the vertical peg axis), mirroring insert_HDMI.
PRE_INSERT_PRE_DIS = 0.05   # hover approach distance in pre_move
PRE_INSERT_DIS = 0.02       # hover height above the seated pose in pre_move
INSERT_PRE_DIS = 0.02       # final descent approach distance in _play_once
INSERT_DIS = 0.003          # final clearance above seated pose before the push


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.554, 1.0, 0.150), rot=(0, 0, 0.707, 0.707), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.94, focus_distance=1.0, horizontal_aperture=2.688, clipping_range=(0.01, 100.0)
            ),
            width=480,
            height=270,
            update_period=1/120
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,  # use existing camera
            width=480,
            height=270,
            update_period=1/120,
        )
    ]
    step_lim = 600


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        # Contact-rich peg insertion benefits from firmer friction, like the
        # other insert tasks, so the gear doesn't slip in the gripper mid-descent.
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        base_pose = Pose(list(GEAR_BASE_POS), [1, 0, 0, 0])
        park_pose = Pose(list(MEDIUM_PARK_POS), [1, 0, 0, 0])

        # Base plate + pegs (heavy → effectively static, like the insert slot).
        self.gear_base = self._actor_manager.add_from_usd_file(
            name='gear_base',
            asset_path="factory_gear_base.usd",
            pose=base_pose,
            density=BASE_DENSITY,
        )
        # Side gears authored in the base frame → spawning at base_pose seats
        # them on their own pegs.
        self.small_gear = self._actor_manager.add_from_usd_file(
            name='small_gear',
            asset_path="factory_gear_small.usd",
            pose=base_pose,
            density=SIDE_GEAR_DENSITY,
        )
        self.large_gear = self._actor_manager.add_from_usd_file(
            name='large_gear',
            asset_path="factory_gear_large.usd",
            pose=base_pose,
            density=SIDE_GEAR_DENSITY,
        )
        # Medium gear: the manipulated object, parked flat on the table.
        self.medium_gear = self._actor_manager.add_from_usd_file(
            name='medium_gear',
            asset_path="factory_gear_medium.usd",
            pose=park_pose,
        )

    def _reset_actors(self):
        # Randomize the whole base assembly (base + small/large gears + pegs) on
        # the table: small xy shift + yaw, like the slot randomization in the
        # insert tasks. All three share one pose because they're authored in the
        # same frame, so the side gears ride along onto their pegs.
        base_offset = self.create_noise([0.02, 0.02, 0.0], euler=[0, 0, np.pi / 12])
        base_pose = Pose(list(GEAR_BASE_POS), [1, 0, 0, 0]).add_offset(base_offset)
        self.gear_base.set_pose(base_pose)
        self.small_gear.set_pose(base_pose)
        self.large_gear.set_pose(base_pose)
        self.metadata['base_offset'] = base_offset.tolist()

        # Randomize the medium gear's in-hand pose by perturbing where it parks
        # (small xy shift + yaw) so the grasp is not identical every episode.
        medium_offset = self.create_noise([0.02, 0.02, 0.0], euler=[0, 0, np.pi / 9])
        medium_pose = Pose(list(MEDIUM_PARK_POS), [1, 0, 0, 0]).add_offset(medium_offset)
        self.medium_gear.set_pose(medium_pose)
        self.metadata['medium_offset'] = medium_offset.tolist()

    def pre_move(self):
        self.delay(10)

        # ── Grasp the medium gear top-down from the table (NOT saved) ──────────
        self.move(self.atom.open_gripper(1.0))

        gear_center = self.medium_gear.get_pose().add_bias(MEDIUM_CENTER_LOCAL, coord='local')
        grasp_rotate = self.rng.uniform(-np.pi / 12, np.pi / 12)
        cpose = construct_grasp_pose(
            gear_center.p,
            [0, 0, 1],   # approach straight down (grasp from +z)
            [1, 0, 0],   # gripper opening axis
        ).add_rotation([0, 0, grasp_rotate], coord='local')  # spin about the approach axis
        cid = self.medium_gear.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.medium_gear,
            contact_point_id=cid,
            pre_dis=0.06, dis=0.0,
            is_close=False,
        ))
        self.move(self.atom.close_gripper())
        self.origin_inhand_pose = self.medium_gear.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose())

        # Lift clear of the table.
        self.move(self.atom.move_by_displacement(z=0.08))

        # ── Hover above the medium peg (with target-axis noise) ───────────────
        # Assembled medium-gear pose == base pose (USD authored in base frame).
        # Keep only the base's planar yaw so the seating axis stays vertical.
        base_pose = self.gear_base.get_pose()
        base_pose.q = (1, 0, 0, 0)
        self.assembly_pose = base_pose

        # Contact-rich noise on the insertion target axis (xy misalignment + yaw).
        insert_noise = self.create_noise([0.004, 0.004, 0.0], euler=[0, 0, np.pi / 36])
        self.noise_assembly_pose = self.assembly_pose.add_offset(insert_noise)
        self.metadata['insert_noise'] = insert_noise.tolist()

        self.move(self.atom.place_actor(
            self.medium_gear,
            target_pose=self.noise_assembly_pose,
            pre_dis=PRE_INSERT_PRE_DIS,
            dis=PRE_INSERT_DIS,
            is_open=False,
        ))

    def _play_once(self):
        # Final cuRobo-planned vertical insertion onto the medium peg (SAVED).
        self.move(self.atom.place_actor(
            self.medium_gear,
            target_pose=self.noise_assembly_pose,
            pre_dis=INSERT_PRE_DIS,
            dis=INSERT_DIS,
            is_open=False,
        ), time_dilation_factor=0.5)

        # Press the gear down onto the peg, locked to the vertical axis. Randomize
        # the press depth / contact force for more diverse tactile data.
        push_depth = self.rng.uniform(0.006, 0.012)
        self.metadata['push_depth'] = float(push_depth)
        self.move(self.atom.move_by_displacement(
            z=push_depth, xyz_coord='local'
        ), time_dilation_factor=0.5, constraint_pose=[1, 1, 1, 1, 1, 0])
        self.move(self.atom.move_by_displacement(
            z=0.003, xyz_coord='local'
        ), time_dilation_factor=0.5, constraint_pose=[1, 1, 1, 1, 1, 0])
        self.delay(20, is_save=True)

    def check_early_stop(self):
        # Abort if the gear slipped substantially in the gripper during descent.
        inhand_pose = self.medium_gear.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose())
        inhand_bias = np.abs(self.origin_inhand_pose[2] - inhand_pose[2])
        if inhand_bias > 0.03:
            self.metadata['early_stop'] = True
            self.metadata['inhand_bias'] = float(inhand_bias)
            return True
        return False

    def check_success(self, z_threshold=0.006):
        # Assembled when the medium gear's frame returns to the base frame:
        # planar offset small, seated down the peg, and axis still vertical.
        rel_pose = self.medium_gear.get_pose().rebase(self.gear_base.get_pose())
        self.metadata['rel_pose'] = rel_pose.tolist()
        return np.all(np.abs(rel_pose.p[:2]) < np.array([0.006, 0.006])) \
            and rel_pose.p[2] < z_threshold \
            and np.dot(rel_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])) > 0.965  # 15°
