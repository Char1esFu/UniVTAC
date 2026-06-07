"""Re-center the factory gear USDs so each USD origin coincides with the gear's
own geometry center (bbox center -> 0,0,0).

The gears originally bake a Factory peg offset into their *mesh vertices*
(small +0.05075, medium +0.02025, large -0.03025 in local X), so each gear's
origin sits at the shared gear-base frame rather than at the gear itself. This
script bakes the offset out: it subtracts each gear's world-bbox center from
every Mesh's points (and `extent`), permanently moving the geometry so the
origin lands at the geometry center.

Only the three gears are touched; factory_gear_base is left as-is (its XY center
is already the origin and the gear-offset math is anchored to it).

Originals are backed up to assets/objects/orig_backup/ before editing.

Usage:
    python scripts/recenter_gears.py            # apply (with backup)
    python scripts/recenter_gears.py --dry-run  # report only, no writes
"""

import argparse
import glob
import os
import shutil
import sys


def _bootstrap_isaac_pxr() -> None:
    """Make Isaac's built-in pxr importable without launching Kit (re-exec once
    so LD_LIBRARY_PATH is in place for the native libs)."""
    matches = glob.glob(
        os.path.join(
            sys.prefix, "lib", "python*", "site-packages", "isaacsim",
            "extscache", "omni.usd.libs-*", "pxr",
        )
    )
    if not matches:
        raise RuntimeError("Could not find Isaac built-in pxr under extscache/")
    pxr_dir = matches[0]
    libs_bin = os.path.join(os.path.dirname(pxr_dir), "bin")
    if libs_bin not in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            p for p in (libs_bin, os.path.join(sys.prefix, "lib"),
                        os.environ.get("LD_LIBRARY_PATH", "")) if p
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)
    sys.path.insert(0, os.path.dirname(pxr_dir))


_bootstrap_isaac_pxr()

from pxr import Gf, Usd, UsdGeom  # noqa: E402

OBJECTS_DIR = "assets/objects"
GEARS = ["factory_gear_small.usd", "factory_gear_medium.usd", "factory_gear_large.usd"]
BACKUP_DIR = os.path.join(OBJECTS_DIR, "orig_backup")


def _bbox_center(stage: Usd.Stage) -> Gf.Vec3d:
    dp = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    rng = cache.ComputeWorldBound(dp).ComputeAlignedRange()
    return (rng.GetMin() + rng.GetMax()) * 0.5


def recenter(path: str, dry_run: bool) -> None:
    stage = Usd.Stage.Open(path)
    offset = _bbox_center(stage)
    print(f"\n{os.path.basename(path)}")
    print(f"  bbox center (offset to remove) = "
          f"({offset[0]:.5f}, {offset[1]:.5f}, {offset[2]:.5f})")

    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    off_f = Gf.Vec3f(offset[0], offset[1], offset[2])
    for prim in meshes:
        mesh = UsdGeom.Mesh(prim)
        pts_attr = mesh.GetPointsAttr()
        pts = pts_attr.Get()
        if pts is None:
            print(f"  - {prim.GetName()}: no points, skipped")
            continue
        new_pts = [p - off_f for p in pts]
        # extent shifts by the same uniform translation
        ext_attr = mesh.GetExtentAttr()
        ext = ext_attr.Get()
        print(f"  - {prim.GetName()}: {len(pts)} points shifted")
        if not dry_run:
            pts_attr.Set(new_pts)
            if ext is not None:
                ext_attr.Set([ext[0] - off_f, ext[1] - off_f])

    if dry_run:
        print("  (dry-run: nothing written)")
        return
    stage.GetRootLayer().Save()
    print("  saved.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        for g in GEARS:
            src = os.path.join(OBJECTS_DIR, g)
            dst = os.path.join(BACKUP_DIR, g)
            if not os.path.exists(dst):  # don't clobber an existing backup
                shutil.copy2(src, dst)
                print(f"backup: {src} -> {dst}")
            else:
                print(f"backup exists, kept: {dst}")

    for g in GEARS:
        recenter(os.path.join(OBJECTS_DIR, g), args.dry_run)


if __name__ == "__main__":
    main()
