"""Read the factory gear USD bounding boxes to inspect each USD's reference
center (origin) vs. its geometry center.

Uses Isaac Sim's built-in pxr/USD — no extra packages. We bootstrap a headless
Kit app first (same pattern as scripts/visualize_gear_assembly.py) so that the
Isaac-shipped `pxr` modules are on sys.path before we import them.

Usage:
    python scripts/usd_related/test_gear_usd.py
"""

from isaaclab.app import AppLauncher

# Headless Kit bootstrap — this is what puts Isaac's built-in pxr on sys.path.
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import os

from pxr import Usd, UsdGeom  # noqa: E402

root = "assets/objects"
files = [
    "factory_gear_base.usd",
    "factory_gear_small.usd",
    "factory_gear_medium.usd",
    "factory_gear_large.usd",
]
for f in files:
    stage = Usd.Stage.Open(os.path.join(root, f))
    dp = stage.GetDefaultPrim()
    print("=" * 72)
    print(f, "  defaultPrim:", dp.GetName() if dp else None)
    bbcache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    rp = dp or stage.GetPseudoRoot()
    rng = bbcache.ComputeWorldBound(rp).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    c = (mn + mx) * 0.5
    s = mx - mn
    print("  bbox min   : (%9.5f, %9.5f, %9.5f)" % (mn[0], mn[1], mn[2]))
    print("  bbox max   : (%9.5f, %9.5f, %9.5f)" % (mx[0], mx[1], mx[2]))
    print(
        "  bbox center: (%9.5f, %9.5f, %9.5f)   <- geom center rel. to USD origin"
        % (c[0], c[1], c[2])
    )
    print("  bbox size  : (%9.5f, %9.5f, %9.5f)" % (s[0], s[1], s[2]))

simulation_app.close()
