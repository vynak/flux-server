"""WSGI entrypoint: build the app, load the models, hand it to gunicorn.

    gunicorn --workers 1 --threads 4 --timeout 1800 wsgi:app
    python wsgi.py            # single process, for local use

One worker on purpose: every worker loads its own copy of the models. Use
threads to serve concurrent requests instead.

This is a CPU service. FLUX is a 4B model: it fits in RAM, not in a hurry.
"""

import os
from pathlib import Path

# Load .env first: it cannot override real environment variables (python-dotenv
# does not overwrite by default), so the setdefault calls below only fill the
# values .env and the process environment left unset.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass  # .env support is optional; environment variables work on their own

# Keep the weights next to the project: on Windows, writing to
# %USERPROFILE%\.cache is blocked by Defender's controlled folder access on
# some machines. Respects an explicit HF_HOME.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".hf-cache"))
# The hub's default socket read timeout is only 10 s: too aggressive on a slow link.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

try:
    from app import create_app
except ModuleNotFoundError as exc:  # the most common first-run failure
    raise SystemExit(
        f"Missing dependency: {exc.name}\n\n"
        "You are probably running the global Python instead of the project venv.\n"
        "From the repository root, either activate it:\n"
        r"    .\.venv\Scripts\Activate.ps1      (PowerShell)" "\n"
        "or call its interpreter directly (works even if activation is blocked):\n"
        r"    .\.venv\Scripts\python.exe -m pip install -r requirements.txt" "\n"
        r"    .\.venv\Scripts\python.exe wsgi.py" "\n"
    ) from None

app = create_app()
try:
    app.extensions["repaint"].load()
except RuntimeError as exc:
    raise SystemExit(f"\nCould not load the models.\n\n{exc}\n") from None

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
