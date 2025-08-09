import os
import shutil
import subprocess
import sys


POTRACE_CMD = "potrace"


def _local_potrace_candidate() -> str | None:
    # Look for bundled binary under project ./bin
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    bin_dir = os.path.join(root, "bin")
    candidates = ["potrace", "potrace.exe"] if os.name == "nt" else ["potrace"]
    for name in candidates:
        cand = os.path.join(bin_dir, name)
        if os.path.isfile(cand):
            return cand
    return None


def ensure_potrace_available() -> None:
    global POTRACE_CMD
    exe = shutil.which("potrace")
    if exe:
        POTRACE_CMD = exe
        return
    local = _local_potrace_candidate()
    if local:
        POTRACE_CMD = local
        return
    raise RuntimeError(
        "Potrace is not available. Install it and ensure 'potrace' is on PATH, or place the binary under 'bin/'."
    )


def potrace_to_svg(pbm_path: str, svg_path: str, turdsize: int = 2, alphamax: float = 1.0) -> None:
    """Invoke Potrace to convert PBM to SVG.

    - turdsize: suppress speckles smaller than this number of pixels
    - alphamax: corner threshold (higher = smoother)
    """
    if not os.path.exists(pbm_path):
        raise FileNotFoundError(pbm_path)

    cmd = [
        POTRACE_CMD,
        "-s",  # output SVG
        "-o",
        svg_path,
        "--turdsize",
        str(turdsize),
        "--alphamax",
        str(alphamax),
        pbm_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Potrace failed: {result.stderr.strip()}")


