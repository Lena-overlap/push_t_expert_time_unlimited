"""Check direct dependencies and real calibration files without running physics."""
import importlib.util
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
failed = False
for name in ("numpy", "scipy", "shapely", "mujoco", "robosuite", "imageio", "imageio_ffmpeg", "PIL"):
    present = importlib.util.find_spec(name) is not None
    print(f"{name}: {'OK' if present else 'MISSING'}")
    failed |= not present
for path in (ROOT / "config" / "contact_model.json", ROOT / "config" / "contact_response_model.json"):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if path.name == "contact_model.json":
            assert data.get("version") == 1 and isinstance(data.get("yaw_bins"), int) and data["yaw_bins"] > 0
            assert data.get("environment") in (None, "random")
            assert any(e.get("reliable") for e in data["entries"]), "No reliable contacts"
        print(f"{path.name}: OK (basic JSON check)")
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        print(f"{path.name}: MISSING/INVALID ({exc})")
        failed = True
if failed:
    print("Use the existing working Python environment and copy the original contact_model.json into config/.")
sys.exit(int(failed))
