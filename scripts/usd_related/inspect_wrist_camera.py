"""Inspect the Franka GelSight-Mini gripper USD and dump the wrist-camera
mounting parameters (the transform of the WristCamera prim relative to the arm
link it is parented under).

Same no-Kit pxr bootstrap as read_usd.py — reads the binary USD crate directly
without launching Isaac Sim. Use it to find, and as a template to modify, the
wrist camera's mount pose.

Usage:
    python scripts/usd_related/inspect_wrist_camera.py
"""

import glob
import os
import sys


def _bootstrap_isaac_pxr() -> str:
    matches = glob.glob(
        os.path.join(
            sys.prefix, "lib", "python*", "site-packages", "isaacsim",
            "extscache", "omni.usd.libs-*", "pxr",
        )
    )
    if not matches:
        raise RuntimeError(
            "Could not find Isaac built-in pxr under "
            f"{sys.prefix}/lib/python*/site-packages/isaacsim/extscache/"
        )
    pxr_dir = matches[0]
    libs_bin = os.path.join(os.path.dirname(pxr_dir), "bin")
    conda_lib = os.path.join(sys.prefix, "lib")
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if libs_bin not in current.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            p for p in (libs_bin, conda_lib, current) if p
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)
    sys.path.insert(0, os.path.dirname(pxr_dir))
    return pxr_dir


_bootstrap_isaac_pxr()

from pxr import Usd, UsdGeom, Gf  # noqa: E402

# Direct path (importing tacex_assets would pull in Kit-only isaaclab modules,
# which the no-Kit pxr bootstrap above intentionally avoids).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROBOT_USD = os.path.join(
    _PROJECT_ROOT,
    "third_party/TacEx/source/tacex_assets/tacex_assets/data",
    "Robots/Franka/GelSight_Mini/Gripper/uipc_gelpads_high_res_wrist.usd",
)


HAND_PRIM = "/panda/panda_hand"
CAM_PRIM = "/panda/WristCamera/Camera"


def _fmt_mat(m: Gf.Matrix4d) -> str:
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat()
    im = q.GetImaginary()
    return (
        f"pos=({t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f})  "
        f"quat_wxyz=({q.GetReal():.6f}, {im[0]:.6f}, {im[1]:.6f}, {im[2]:.6f})"
    )


def _quat_mul(a, b):
    """Hamilton product of two (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _normalize_sign(q, eps: float = 1e-9):
    """Pick a canonical sign (q and -q are the same rotation): make the first
    component whose magnitude exceeds eps positive. Using w alone is ambiguous
    when w == 0 (e.g. a pure 180 deg rotation)."""
    for c in q:
        if abs(c) > eps:
            return tuple(-x for x in q) if c < 0 else q
    return q


def _opengl_to_ros(q_wxyz):
    """OpenGL/USD camera frame -> ROS camera frame: rotate 180 deg about local
    X (right-multiply by the unit-X quaternion). pos is convention-independent."""
    return _normalize_sign(_quat_mul(q_wxyz, (0.0, 1.0, 0.0, 0.0)))


def _print_cfg_offset(stage) -> None:
    """Compute the WristCamera/Camera pose expressed in the panda_hand frame and
    print it in both OpenGL (USD-native) and ROS convention -- i.e. the exact
    numbers that go into CameraCfg.OffsetCfg(pos=..., rot=..., convention=...)
    in envs/_base_task.py."""
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    w_hand = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(HAND_PRIM))
    w_cam = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(CAM_PRIM))
    # Gf uses row-vector convention: world = local * L2W, so the camera pose in
    # the hand frame is  cam_in_hand = W_cam * inv(W_hand).
    cam_in_hand = w_cam * w_hand.GetInverse()
    t = cam_in_hand.ExtractTranslation()
    q = cam_in_hand.ExtractRotationQuat()
    im = q.GetImaginary()
    q_opengl = _normalize_sign((q.GetReal(), im[0], im[1], im[2]))
    q_ros = _opengl_to_ros(q_opengl)

    print("\n" + "=" * 78)
    print(f"WristCamera/Camera pose in panda_hand frame  (-> CameraCfg.OffsetCfg)")
    print("=" * 78)
    print(f"  pos                      = ({t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f})")
    print(f"  rot_wxyz (convention=opengl, USD-native) = "
          f"({q_opengl[0]:.6f}, {q_opengl[1]:.6f}, {q_opengl[2]:.6f}, {q_opengl[3]:.6f})")
    print(f"  rot_wxyz (convention=ros)                = "
          f"({q_ros[0]:.6f}, {q_ros[1]:.6f}, {q_ros[2]:.6f}, {q_ros[3]:.6f})")


def main() -> None:
    stage = Usd.Stage.Open(ROBOT_USD)
    print("=" * 78)
    print("robot USD :", ROBOT_USD)
    dp = stage.GetDefaultPrim()
    print("defaultPrim:", dp.GetPath() if dp else None)

    # 1) Find every Camera prim and anything named *Wrist*.
    cam_prims, wrist_prims = [], []
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Camera":
            cam_prims.append(prim)
        if "wrist" in prim.GetName().lower():
            wrist_prims.append(prim)

    print("\n--- Camera prims ---")
    for p in cam_prims:
        print(" ", p.GetPath(), "  type:", p.GetTypeName())
    print("\n--- *wrist* prims ---")
    for p in wrist_prims:
        print(" ", p.GetPath(), "  type:", p.GetTypeName())

    # 2) For the WristCamera subtree, dump parent chain + local/relative xforms.
    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    targets = wrist_prims + [p for p in cam_prims if p not in wrist_prims]
    for prim in targets:
        print("\n" + "=" * 78)
        print("prim       :", prim.GetPath(), "  type:", prim.GetTypeName())
        parent = prim.GetParent()
        print("parent     :", parent.GetPath(), "  type:", parent.GetTypeName())

        # Local transform = the mounting pose relative to the parent prim.
        if prim.IsA(UsdGeom.Xformable):
            local = UsdGeom.Xformable(prim).GetLocalTransformation()
            print("LOCAL xform (relative to parent):")
            print("   ", _fmt_mat(local))
            ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
            print("authored xformOps:")
            for op in ops:
                print(f"    {op.GetOpName():28s} = {op.Get()}")

        # Also show the world transform (relative to the USD default prim root).
        world = xf_cache.GetLocalToWorldTransform(prim)
        print("WORLD xform (relative to robot root):")
        print("   ", _fmt_mat(world))

    # 3) The numbers that actually go into CameraCfg.OffsetCfg.
    _print_cfg_offset(stage)


if __name__ == "__main__":
    main()
