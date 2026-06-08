"""Read the factory gear USD bounding boxes to inspect each USD's reference
center (origin) vs. its geometry center.

Uses Isaac Sim's built-in pxr/USD with **no extra packages and without starting
Kit** (no SimulationApp / GPU). Isaac ships its pxr under
``isaacsim/extscache/omni.usd.libs-*/`` with native ``.so`` files in ``bin/``.
Since ``LD_LIBRARY_PATH`` is only read at process start, we inject it on the
first run and re-exec once; the second pass puts the built-in ``pxr`` on
``sys.path`` and imports it directly.

Usage:
    python scripts/usd_related/read_gear_usd.py
"""

import glob
import os
import sys


def _bootstrap_isaac_pxr() -> str:
    """Make Isaac's built-in pxr importable without launching Kit.

    Returns the pxr directory that gets prepended to sys.path. Re-execs the
    process once (the first time) so the dynamic linker picks up the native
    libs via LD_LIBRARY_PATH.
    """
    # Locate the Isaac-shipped pxr (version hash globbed, not hardcoded).
    matches = glob.glob(
        os.path.join(
            sys.prefix,
            "lib",
            "python*",
            "site-packages",
            "isaacsim",
            "extscache",
            "omni.usd.libs-*",
            "pxr",
        )
    )
    if not matches:
        raise RuntimeError(
            "Could not find Isaac built-in pxr under "
            f"{sys.prefix}/lib/python*/site-packages/isaacsim/extscache/"
        )
    pxr_dir = matches[0]
    libs_bin = os.path.join(os.path.dirname(pxr_dir), "bin")  # native .so files
    conda_lib = os.path.join(sys.prefix, "lib")  # libpython3.10.so.1.0 etc.

    # LD_LIBRARY_PATH is consumed at process startup → inject + re-exec once.
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if libs_bin not in current.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            p for p in (libs_bin, conda_lib, current) if p
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)

    # Second pass: linker is set up. Put the *parent* of the pxr package dir on
    # sys.path so `import pxr` resolves to the Isaac build.
    libs_root = os.path.dirname(pxr_dir)
    sys.path.insert(0, libs_root)
    return pxr_dir


_bootstrap_isaac_pxr()

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
