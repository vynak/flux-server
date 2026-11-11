"""Download the model weights before starting the server.

    python scripts/download_models.py                # both models
    python scripts/download_models.py --only gen     # just the diffusion model
    python scripts/download_models.py --check        # verify cached weights, download nothing

huggingface_hub keeps in-flight bytes in ``blobs/<oid>.incomplete`` and resumes
them with an HTTP Range request on the next attempt, so an interrupted run keeps
its progress: rerun the same command to continue. Transfers are serialized
(``max_workers=1``) because parallel LFS streams on a flaky connection are more
likely to abort.

Model ids come from the same environment variables as app/config.py
(FLUX_SEG_MODEL / FLUX_GEN_MODEL); a .env file is read like wsgi.py does. This
script only needs the stdlib plus huggingface_hub (and optional python-dotenv).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Keep in sync with app/config.py.
SEG_MODEL_DEFAULT = "nvidia/segformer-b5-finetuned-ade-640-640"
GEN_MODEL_DEFAULT = "black-forest-labs/FLUX.2-klein-4B"

DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 5

# The same name-based set app/pipeline.py walks, plus the httpx exceptions used
# by huggingface_hub 2.x (ConnectError / ReadError / RemoteProtocolError). Names
# instead of classes so this works with either HTTP stack and never imports
# requests.
TRANSIENT_ERROR_NAMES = frozenset(
    {
        "ChunkedEncodingError",
        "ConnectionError",
        "ReadTimeout",
        "ConnectTimeout",
        "Timeout",
        "TimeoutError",
        "ProtocolError",
        "IncompleteRead",
        "SSLError",
        "ConnectionResetError",
        "RemoteDisconnected",
        "BrokenPipeError",
        "RemoteProtocolError",
        "ConnectError",
        "ReadError",
    }
)

# Files that count as model weights when checking the cache offline.
WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".ckpt", ".pt"})
# model_index.json values are [library, class]; these classes carry no weights.
WEIGHTLESS_CLASS_HINTS = ("Tokenizer", "Scheduler", "Processor", "FeatureExtractor")


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable, no huggingface_hub import)
# --------------------------------------------------------------------------- #


def _component_names(model_index: dict) -> list[str]:
    """Return component directories: keys whose value is ``[library, class]``."""
    return [
        name
        for name, value in model_index.items()
        if not name.startswith("_") and isinstance(value, list)
    ]


def _gen_allow_patterns(model_index: dict) -> list[str]:
    """Files DiffusionPipeline needs.

    Only the component directories are fetched, so the duplicate repo-root
    ``flux-2-klein-4b.safetensors`` (another 7.75 GB) and the preview JPEGs are
    skipped.
    """
    return ["model_index.json", *(f"{name}/*" for name in _component_names(model_index))]


def _seg_allow_patterns() -> list[str]:
    """SegFormer files.

    The default revision of the nvidia repo ships ``pytorch_model.bin`` (there is
    no ``model.safetensors`` there), so ``*.bin`` has to be included. The
    TensorFlow ``tf_model.h5`` is not used by transformers here.
    """
    return ["*.json", "*.safetensors", "*.bin", "*.txt"]


def is_transient_download_error(exc: BaseException) -> bool:
    """True when a download failure is worth retrying.

    Walks the ``__cause__``/``__context__`` chain by class *name*: the concrete
    exception types live in requests/urllib3/httpx, which this script does not
    import.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in TRANSIENT_ERROR_NAMES:
            return True
        current = current.__cause__ or current.__context__
    return False


def _run_with_retries(call, *, what: str, attempts: int, sleep=time.sleep):
    """Run ``call``, retrying transient failures with capped exponential backoff.

    ``attempts`` is the number of retries *after* the first try, so a call that
    keeps failing is made ``attempts + 1`` times before RuntimeError is raised.
    """
    tries = max(1, attempts)
    last_error: BaseException | None = None
    for index in range(tries + 1):
        try:
            return call()
        except Exception as exc:
            if not is_transient_download_error(exc):
                raise
            last_error = exc
            if index >= tries:
                break
            delay = min(2**index, 30)
            print(
                f"{what}: {type(exc).__name__} (attempt {index + 1}/{tries + 1});"
                f" retrying in {delay}s",
                file=sys.stderr,
            )
            sleep(delay)
    raise RuntimeError(
        f"{what} failed after {tries + 1} attempts: "
        f"{type(last_error).__name__}: {last_error}. "
        "The partial download is kept on disk and the next run resumes it with an HTTP "
        "Range request, so simply rerun this command. Installing hf_xet can make large "
        "transfers more reliable: pip install hf_xet"
    ) from last_error


# --------------------------------------------------------------------------- #
# Environment / formatting helpers
# --------------------------------------------------------------------------- #


def _load_env_file() -> None:
    """Load ``.env`` like wsgi.py does; python-dotenv is optional (and never overrides)."""
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv()


def _default_retries() -> int:
    raw = os.getenv("FLUX_DOWNLOAD_RETRIES", str(DEFAULT_RETRIES))
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_RETRIES


def _default_hf_home() -> Path:
    """Keep the weights next to the project, exactly like wsgi.py does."""
    return Path(__file__).resolve().parent.parent / ".hf-cache"


def _load_hub():
    """Import huggingface_hub lazily: --help and the tests never download."""
    import huggingface_hub

    return huggingface_hub


def _sleep(seconds: float) -> None:
    """Indirection so tests do not wait for the real backoff."""
    time.sleep(seconds)


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def _dir_size(directory: Path) -> int:
    total = 0
    for root, dirs, files in os.walk(directory):
        # The 0-byte .no_exist markers are bookkeeping, not weights.
        dirs[:] = [name for name in dirs if name != ".no_exist"]
        for name in files:
            if name.endswith(".no_exist"):
                continue
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue  # broken symlink left behind by a cache cleanup
    return total


def _print_summary(repo: str, snapshot: Path) -> None:
    print(f"  {repo}: {snapshot} ({_human_size(_dir_size(snapshot))} on disk)")


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #


def _download_seg(hub, repo: str, retries: int) -> str:
    print(f"downloading {repo} (resumable; safe to Ctrl-C and rerun)")
    return _run_with_retries(
        lambda: hub.snapshot_download(repo, allow_patterns=_seg_allow_patterns(), max_workers=1),
        what=f"downloading {repo}",
        attempts=retries,
        sleep=_sleep,
    )


def _download_gen(hub, repo: str, retries: int) -> str:
    print(f"downloading {repo} (resumable; safe to Ctrl-C and rerun)")
    index_snapshot = _run_with_retries(
        lambda: hub.snapshot_download(repo, allow_patterns=["model_index.json"], max_workers=1),
        what=f"downloading {repo}/model_index.json",
        attempts=retries,
        sleep=_sleep,
    )
    model_index = json.loads(
        (Path(index_snapshot) / "model_index.json").read_text(encoding="utf-8")
    )
    patterns = _gen_allow_patterns(model_index)
    return _run_with_retries(
        lambda: hub.snapshot_download(repo, allow_patterns=patterns, max_workers=1),
        what=f"downloading {repo}",
        attempts=retries,
        sleep=_sleep,
    )


# --------------------------------------------------------------------------- #
# Cache check (offline)
# --------------------------------------------------------------------------- #


def _repo_cache_dir(repo: str) -> Path:
    """``<hub cache>/models--<org>--<name>`` for ``repo``."""
    try:
        from huggingface_hub import constants

        cache_root = Path(constants.HF_HUB_CACHE)
    except (ImportError, AttributeError):
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_root = Path(hf_home) / "hub"
        else:
            cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    return cache_root / ("models--" + repo.replace("/", "--"))


def _has_incomplete_downloads(cache_dir: Path) -> bool:
    """True if an interrupted transfer left a ``blobs/*.incomplete`` behind."""
    blobs = cache_dir / "blobs"
    return blobs.is_dir() and any(blobs.rglob("*.incomplete"))


def _has_weight_file(directory: Path, suffixes: frozenset[str]) -> bool:
    return any(path.is_file() and path.suffix in suffixes for path in directory.rglob("*"))


def _missing_component_weights(snapshot: Path, model_index: dict) -> list[str]:
    """Component directories that have no weights yet (tokenizers/schedulers skip)."""
    missing = []
    for name in _component_names(model_index):
        value = model_index[name]
        class_name = value[1] if len(value) > 1 and isinstance(value[1], str) else ""
        if any(hint in class_name for hint in WEIGHTLESS_CLASS_HINTS):
            continue
        if not _has_weight_file(snapshot / name, WEIGHT_SUFFIXES):
            missing.append(name)
    return missing


def _seg_problems(repo: str, snapshot: Path) -> list[str]:
    problems = []
    if _has_incomplete_downloads(_repo_cache_dir(repo)):
        problems.append("an interrupted blob is still pending in the cache")
    if not (snapshot / "config.json").is_file():
        problems.append("config.json is missing")
    if not _has_weight_file(snapshot, frozenset({".safetensors", ".bin"})):
        problems.append("no weight file yet")
    return problems


def _check_seg(hub, repo: str) -> Path | None:
    try:
        snapshot = hub.snapshot_download(
            repo,
            allow_patterns=_seg_allow_patterns(),
            local_files_only=True,
            max_workers=1,
        )
    except Exception:
        return None
    return Path(snapshot)


def _check_gen(hub, repo: str) -> tuple[Path | None, list[str]]:
    try:
        index_snapshot = Path(
            hub.snapshot_download(
                repo,
                allow_patterns=["model_index.json"],
                local_files_only=True,
                max_workers=1,
            )
        )
        model_index = json.loads(
            (index_snapshot / "model_index.json").read_text(encoding="utf-8")
        )
    except Exception:
        return None, []
    try:
        snapshot = hub.snapshot_download(
            repo,
            allow_patterns=_gen_allow_patterns(model_index),
            local_files_only=True,
            max_workers=1,
        )
    except Exception:
        return None, []
    return Path(snapshot), _missing_component_weights(Path(snapshot), model_index)


def _check_models(hub, models: dict[str, str], selected: list[str]) -> int:
    failed = 0
    for kind in selected:
        repo = models[kind]
        if kind == "seg":
            snapshot = _check_seg(hub, repo)
            problems = _seg_problems(repo, snapshot) if snapshot is not None else []
        else:
            snapshot, missing = _check_gen(hub, repo)
            problems = []
            if snapshot is not None:
                if _has_incomplete_downloads(_repo_cache_dir(repo)):
                    problems.append("an interrupted blob is still pending in the cache")
                if missing:
                    problems.append("no weights for: " + ", ".join(missing))

        if snapshot is None:
            print(f"missing {repo}")
            failed += 1
        elif problems:
            print(f"incomplete {repo} ({'; '.join(problems)})")
            _print_summary(repo, snapshot)
            failed += 1
        else:
            print(f"ok {repo}")
            _print_summary(repo, snapshot)

    if failed:
        print(
            f"\n{failed} model(s) are missing or incomplete; rerun without --check "
            "to resume/complete the download.",
            file=sys.stderr,
        )
        return 1
    return 0


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    # Load .env first, exactly like wsgi.py: model ids and HF_HOME may live there,
    # and the --retries default is read from the environment below.
    _load_env_file()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only", choices=("seg", "gen"), help="download a single model")
    parser.add_argument(
        "--retries",
        type=int,
        default=_default_retries(),
        help="extra attempts on transient network errors (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="per-request download timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the cache and download nothing",
    )
    args = parser.parse_args(argv)

    models = {
        # Keep in sync with app/config.py.
        "seg": os.getenv("FLUX_SEG_MODEL", SEG_MODEL_DEFAULT),
        "gen": os.getenv("FLUX_GEN_MODEL", GEN_MODEL_DEFAULT),
    }
    selected = ["seg", "gen"] if args.only is None else [args.only]

    # huggingface_hub reads these at import time, so set them before the import.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(args.timeout))
    os.environ.setdefault("HF_HOME", str(_default_hf_home()))

    try:
        hub = _load_hub()
    except ModuleNotFoundError:
        print(
            "huggingface_hub is not installed. From the repository root run:\n"
            r"    .\.venv\Scripts\python.exe -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    if args.check:
        return _check_models(hub, models, selected)

    try:
        for kind in selected:
            repo = models[kind]
            if kind == "seg":
                snapshot = _download_seg(hub, repo, args.retries)
            else:
                snapshot = _download_gen(hub, repo, args.retries)
            _print_summary(repo, Path(snapshot))
    except KeyboardInterrupt:
        print(
            "\ninterrupted: the partial download is kept on disk; rerun the same "
            "command to resume.",
            file=sys.stderr,
        )
        return 1
    except RuntimeError as exc:
        print(f"\ndownload failed: {exc}", file=sys.stderr)
        print(
            "You can also run `python wsgi.py`: the server retries the download itself.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"\ndownload failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "Partial files are kept in the cache; rerun this command to resume "
            "the download.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
