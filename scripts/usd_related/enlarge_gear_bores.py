"""Enlarge the central bore of the factory gear USDs so the bore↔peg radial
clearance exceeds the UIPC global d_hat (0.5 mm).

Why
---
The gears are AffineBody rigid bodies. Their bore↔peg radial clearance is only
~0.11-0.25 mm — smaller than the global IPC contact thickness d_hat (0.5 mm),
which cannot be set per-pair in libuipc. So the contact barrier stays engaged
all the way around the bore and its faceted radial cushion holds the light
(19 g) neighbour gears against gravity and stops them spinning. Softening the
pairwise contact resistance instead breaks medium-gear insertion and makes the
IPC solver crawl, so the only clean fix is geometric: open the bore until the
clearance comfortably exceeds d_hat.

What it does
------------
For each gear Mesh it finds the bore axis (the gears are re-centred so the axis
is ~the local origin), then pushes every vertex inside `CUTOFF_R` radially
OUTWARD by `DELTA_R`, enlarging the hole while leaving the teeth (large radius)
untouched. That is all — no backup, no tet-cache clearing: run this on freshly
recentred copies that have no cached `tet_*` attrs yet, so float-tetwild
regenerates the tet/collision mesh from the new bore on the next sim load.

Workflow
--------
    # 1. inspect: print the radial histogram, pick CUTOFF_R / DELTA_R, no writes
    python scripts/usd_related/enlarge_gear_bores.py --dry-run

    # 2. apply
    python scripts/usd_related/enlarge_gear_bores.py
"""

import argparse
import glob
import math
import os
import sys


def _bootstrap_isaac_pxr() -> None:
    """Make Isaac's built-in pxr importable without launching Kit."""
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

# ── Tunables (confirm with --dry-run first) ─────────────────────────────────────
# Vertices with radial distance < CUTOFF_R from the bore axis are treated as the
# bore wall/rim and pushed outward by DELTA_R. Pick CUTOFF_R between the bore
# wall (~5 mm) and where the gear body starts (read the histogram!). DELTA_R is
# how much to open the bore radius; bore ~5 mm + DELTA_R should clear the peg
# (4.75 mm) by > d_hat (0.5 mm), so DELTA_R ≈ 0.9 mm → clearance ≈ 1.15 mm.
CUTOFF_R = 0.0065   # m
DELTA_R  = 0.0006   # m
INNER_CLUSTER_R = 0.008   # m, used only to locate the bore axis


def _bore_axis(pts) -> Gf.Vec2f:
    """Median XY of the inner cluster ≈ bore axis (robust to a small off-centre)."""
    inner = [(p[0], p[1]) for p in pts
             if math.hypot(p[0], p[1]) < INNER_CLUSTER_R]
    if not inner:
        return Gf.Vec2f(0.0, 0.0)
    inner.sort(key=lambda xy: xy[0]); cx = inner[len(inner) // 2][0]
    inner.sort(key=lambda xy: xy[1]); cy = inner[len(inner) // 2][1]
    return Gf.Vec2f(cx, cy)


def _histogram(radials) -> None:
    edges = [0, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 10, 15, 25, 1e9]  # mm
    rmm = sorted(r * 1000 for r in radials)
    print("    radial histogram (mm bins -> vertex count):")
    i = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        c = 0
        while i < len(rmm) and rmm[i] < hi:
            c += 1; i += 1
        if c:
            print(f"      [{lo:>4}, {hi if hi < 1e8 else '∞':>4}) : {c}")


def _bore_z_profile(pts, radials) -> None:
    """Print radial(min/max) of the bore-region verts (radial < CUTOFF_R) split
    into z-layers. A chamfer shows up as the radial RANGE widening toward the
    end faces (top/bottom z-layers). If max radial stays well below CUTOFF_R the
    whole feature (wall + chamfer) shifts out intact with no boundary tearing."""
    br = [(p[2], r) for p, r in zip(pts, radials) if 5e-4 < r < CUTOFF_R]
    if not br:
        print("    bore z-profile: (no bore-region verts)"); return
    zmin = min(z for z, _ in br); zmax = max(z for z, _ in br)
    span = (zmax - zmin) or 1.0
    nlayers = 8
    print(f"    bore z-profile ({len(br)} verts, z {zmin*1000:.1f}..{zmax*1000:.1f}mm; "
          f"radial min/max per z-layer, CUTOFF={CUTOFF_R*1000:.1f}mm):")
    for L in range(nlayers):
        lo = zmin + span * L / nlayers
        hi = zmin + span * (L + 1) / nlayers
        rs = [r for z, r in br if (lo <= z < hi) or (L == nlayers - 1 and z == hi)]
        if rs:
            print(f"      z[{lo*1000:6.2f},{hi*1000:6.2f}] : R {min(rs)*1000:.3f}..{max(rs)*1000:.3f}mm  (n={len(rs)})")


def process(path: str, dry_run: bool) -> None:
    stage = Usd.Stage.Open(path)
    print(f"\n{os.path.basename(path)}")
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    for prim in meshes:
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if pts is None:
            print(f"  - {prim.GetName()}: no points, skipped"); continue
        axis = _bore_axis(pts)
        radials = [math.hypot(p[0] - axis[0], p[1] - axis[1]) for p in pts]
        bore_r = min(radials)
        print(f"  - {prim.GetName()}: {len(pts)} points, axis=({axis[0]:.4f},{axis[1]:.4f}), "
              f"bore_inner_R={bore_r*1000:.3f}mm")
        _histogram(radials)
        _bore_z_profile(pts, radials)

        moved = 0
        new_pts = list(pts)
        for k, (p, r) in enumerate(zip(pts, radials)):
            if r < CUTOFF_R and r > 5e-4:          # bore region, skip near-axis verts
                f = (r + DELTA_R) / r
                new_pts[k] = Gf.Vec3f(axis[0] + (p[0] - axis[0]) * f,
                                      axis[1] + (p[1] - axis[1]) * f,
                                      p[2])
                moved += 1
        print(f"      -> {moved} bore verts pushed out by {DELTA_R*1000:.2f}mm "
              f"(new bore_inner_R≈{(bore_r+DELTA_R)*1000:.3f}mm)")

        if not dry_run:
            mesh.GetPointsAttr().Set(new_pts)

    if dry_run:
        print("  (dry-run: nothing written)")
        return
    stage.GetRootLayer().Save()
    print("  saved (tet mesh will regenerate on next sim load).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for g in GEARS:
        process(os.path.join(OBJECTS_DIR, g), args.dry_run)


if __name__ == "__main__":
    main()
