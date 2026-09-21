"""Importing the package must be side-effect-free (issue #8 done-when)."""

import subprocess
import sys

PROBE = (
    "import sys, json\n"
    "sys.path.insert(0, 'src')\n"
    "import local_judge\n"
    "banned = [m for m in sys.modules if m.split('.')[0] in\n"
    "          ('requests', 'httpx', 'urllib3', 'socketserver', 'asyncio', 'sqlite3', 'subprocess', 'ctypes')]\n"
    "assert not banned, banned\n"
    "assert isinstance(local_judge.__all__, list) and local_judge.__all__\n"
    "print(json.dumps(local_judge.__all__))\n"
)


def test_import_is_side_effect_free():
    proc = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_public_surface_is_exported():
    import local_judge

    for name in ("RequestValidator", "RequestEnvelope", "StructuralError", "StructuralCode",
                 "RejectionResponse", "CompletedResponse", "ResultEntry", "TraceRecord"):
        assert hasattr(local_judge, name), name
