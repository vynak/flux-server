"""Tests for scripts/download_models.py (no network, no real downloads)."""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_models.py"

SEG_REPO = "nvidia/segformer-b5-finetuned-ade-640-640"
GEN_REPO = "black-forest-labs/FLUX.2-klein-4B"

# The real FLUX model_index.json shape (is_distilled is a bool, not a list).
FLUX_MODEL_INDEX = {
    "_class_name": "Flux2KleinPipeline",
    "_diffusers_version": "0.37.0.dev0",
    "is_distilled": True,
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen3ForCausalLM"],
    "tokenizer": ["transformers", "Qwen2TokenizerFast"],
    "transformer": ["diffusers", "Flux2Transformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLFlux2"],
}


class ProtocolError(Exception):
    """Stands in for urllib3.exceptions.ProtocolError."""


class ChunkedEncodingError(Exception):
    """Stands in for requests.exceptions.ChunkedEncodingError."""


@pytest.fixture(scope="module")
def download_models():
    spec = importlib.util.spec_from_file_location("download_models", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_hub(monkeypatch, snapshot_download):
    """Inject a stand-in huggingface_hub so main() never touches the network."""
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return module


def _prepare_main(monkeypatch, tmp_path, download_models):
    """Isolate env vars and skip the real backoff sleeps.

    The model ids are *set* (not deleted) because main() loads .env with
    python-dotenv, which must not override values already in the environment.
    """
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    monkeypatch.setenv("FLUX_SEG_MODEL", SEG_REPO)
    monkeypatch.setenv("FLUX_GEN_MODEL", GEN_REPO)
    monkeypatch.setattr(download_models, "_sleep", lambda seconds: None)


def _write_complete_snapshot(root: Path) -> None:
    """A usable snapshot for both repos: JSONs + one weight file per component."""
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_text("", encoding="utf-8")
    (root / "model_index.json").write_text(json.dumps(FLUX_MODEL_INDEX), encoding="utf-8")
    for component in ("text_encoder", "transformer", "vae"):
        directory = root / component
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "diffusion_pytorch_model.safetensors").write_text("", encoding="utf-8")


def _fake_constant_snapshot(monkeypatch, snapshot_path: Path):
    def fake_snapshot_download(repo_id, **kwargs):
        assert kwargs["local_files_only"] is True
        assert kwargs["max_workers"] == 1
        return str(snapshot_path)

    _install_fake_hub(monkeypatch, fake_snapshot_download)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_component_names_ignores_metadata_keys(download_models):
    assert download_models._component_names(FLUX_MODEL_INDEX) == [
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
    ]


def test_gen_allow_patterns_skip_root_checkpoint_and_images(download_models):
    patterns = download_models._gen_allow_patterns(FLUX_MODEL_INDEX)

    assert patterns[0] == "model_index.json"
    assert set(patterns[1:]) == {
        "scheduler/*",
        "text_encoder/*",
        "tokenizer/*",
        "transformer/*",
        "vae/*",
    }
    assert all("/" in pattern for pattern in patterns[1:])
    assert "flux-2-klein-4b.safetensors" not in patterns
    assert not any(fnmatch.fnmatch("flux-2-klein-4b.safetensors", p) for p in patterns)
    assert not any(fnmatch.fnmatch("room.jpg", pattern) for pattern in patterns)


def test_seg_allow_patterns_include_the_bin_weights(download_models):
    patterns = download_models._seg_allow_patterns()

    assert patterns == ["*.json", "*.safetensors", "*.bin", "*.txt"]
    # The default revision of the nvidia repo only ships pytorch_model.bin.
    assert any(fnmatch.fnmatch("pytorch_model.bin", pattern) for pattern in patterns)


def test_is_transient_download_error_follows_cause_chain(download_models):
    try:
        try:
            raise ProtocolError("incomplete read")
        except ProtocolError as cause:
            raise ChunkedEncodingError("stream ended") from cause
    except ChunkedEncodingError as exc:
        assert download_models.is_transient_download_error(exc) is True


def test_is_transient_download_error_walks_wrapped_causes(download_models):
    try:
        try:
            raise ProtocolError("incomplete read")
        except ProtocolError as cause:
            raise RuntimeError("download aborted") from cause
    except RuntimeError as exc:
        assert download_models.is_transient_download_error(exc) is True


def test_is_transient_download_error_rejects_value_error(download_models):
    assert download_models.is_transient_download_error(ValueError("bad repo id")) is False


@pytest.mark.parametrize("name", ["RemoteProtocolError", "ConnectError", "ReadError"])
def test_is_transient_download_error_includes_httpx_names(download_models, name):
    error = type(name, (Exception,), {})("transport hiccup")
    assert download_models.is_transient_download_error(error) is True


# --------------------------------------------------------------------------- #
# Retry loop
# --------------------------------------------------------------------------- #


def test_run_with_retries_recovers_after_two_transient_failures(download_models):
    calls = []
    sleeps = []

    def call():
        calls.append(object())
        if len(calls) < 3:
            raise ChunkedEncodingError("stream ended")
        return "snapshot-path"

    result = download_models._run_with_retries(
        call, what="download", attempts=3, sleep=sleeps.append
    )

    assert result == "snapshot-path"
    assert len(calls) == 3
    assert sleeps == [1, 2]


def test_run_with_retries_raises_actionable_error_after_exhaustion(download_models):
    calls = []

    def call():
        calls.append(object())
        raise ChunkedEncodingError("stream ended")

    with pytest.raises(RuntimeError, match="resum") as excinfo:
        download_models._run_with_retries(
            call, what="download", attempts=2, sleep=lambda seconds: None
        )

    assert len(calls) == 3
    message = str(excinfo.value)
    assert "kept" in message
    assert "hf_xet" in message


def test_run_with_retries_does_not_retry_permanent_errors(download_models):
    calls = []

    def call():
        calls.append(object())
        raise ValueError("bad repository id")

    with pytest.raises(ValueError):
        download_models._run_with_retries(
            call, what="download", attempts=3, sleep=lambda seconds: None
        )

    assert len(calls) == 1


def test_default_retries_reads_env_and_clamps(monkeypatch, download_models):
    monkeypatch.delenv("FLUX_DOWNLOAD_RETRIES", raising=False)
    assert download_models._default_retries() == 5

    monkeypatch.setenv("FLUX_DOWNLOAD_RETRIES", "3")
    assert download_models._default_retries() == 3

    monkeypatch.setenv("FLUX_DOWNLOAD_RETRIES", "0")
    assert download_models._default_retries() == 1

    monkeypatch.setenv("FLUX_DOWNLOAD_RETRIES", "not-a-number")
    assert download_models._default_retries() == 5


# --------------------------------------------------------------------------- #
# main()
# --------------------------------------------------------------------------- #


def test_help_works_without_importing_huggingface_hub(download_models, capsys):
    with pytest.raises(SystemExit) as excinfo:
        download_models.main(["--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--only" in out
    assert "--check" in out


def test_load_env_file_is_optional(monkeypatch, download_models):
    # python-dotenv is a runtime requirement but must never be mandatory here.
    monkeypatch.setitem(sys.modules, "dotenv", None)
    assert download_models._load_env_file() is None


def test_main_gen_downloads_components_only(monkeypatch, tmp_path, download_models):
    _prepare_main(monkeypatch, tmp_path, download_models)
    (tmp_path / "model_index.json").write_text(
        json.dumps(FLUX_MODEL_INDEX), encoding="utf-8"
    )
    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    _install_fake_hub(monkeypatch, fake_snapshot_download)

    assert download_models.main(["--only", "gen", "--retries", "1"]) == 0

    assert [repo_id for repo_id, _ in calls] == [GEN_REPO, GEN_REPO]
    assert calls[0][1]["allow_patterns"] == ["model_index.json"]

    patterns = calls[1][1]["allow_patterns"]
    assert patterns == download_models._gen_allow_patterns(FLUX_MODEL_INDEX)
    assert "transformer/*" in patterns
    assert "flux-2-klein-4b.safetensors" not in patterns
    assert all(kwargs["max_workers"] == 1 for _, kwargs in calls)


def test_main_retries_transient_error_then_succeeds(monkeypatch, tmp_path, download_models):
    _prepare_main(monkeypatch, tmp_path, download_models)
    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ChunkedEncodingError("stream ended")
        return str(tmp_path)

    _install_fake_hub(monkeypatch, fake_snapshot_download)

    assert download_models.main(["--only", "seg", "--retries", "2"]) == 0
    assert len(calls) == 2


def test_main_returns_1_after_retries_are_exhausted(
    monkeypatch, tmp_path, download_models, capsys
):
    _prepare_main(monkeypatch, tmp_path, download_models)
    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append(kwargs)
        raise ChunkedEncodingError("connection reset by peer")

    _install_fake_hub(monkeypatch, fake_snapshot_download)

    assert download_models.main(["--only", "seg", "--retries", "2"]) == 1
    assert len(calls) == 3

    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "resum" in err


def test_main_check_reports_complete_cache(monkeypatch, tmp_path, download_models, capsys):
    _prepare_main(monkeypatch, tmp_path, download_models)
    _write_complete_snapshot(tmp_path)
    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append(kwargs)
        assert kwargs["local_files_only"] is True
        assert kwargs["max_workers"] == 1
        return str(tmp_path)

    _install_fake_hub(monkeypatch, fake_snapshot_download)

    assert download_models.main(["--check"]) == 0

    out = capsys.readouterr().out
    assert f"ok {SEG_REPO}" in out
    assert f"ok {GEN_REPO}" in out
    assert "incomplete" not in out
    # seg check + cached model_index.json lookup + gen component check
    assert len(calls) == 3


def test_main_check_flags_gen_component_without_weights(
    monkeypatch, tmp_path, download_models, capsys
):
    _prepare_main(monkeypatch, tmp_path, download_models)
    _write_complete_snapshot(tmp_path)
    transformer = tmp_path / "transformer"
    for weight in transformer.glob("*.safetensors"):
        weight.unlink()
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    _fake_constant_snapshot(monkeypatch, tmp_path)

    assert download_models.main(["--check", "--only", "gen"]) == 1

    captured = capsys.readouterr()
    assert f"incomplete {GEN_REPO}" in captured.out
    assert "no weights for: transformer" in captured.out
    assert "rerun without --check" in captured.err


def test_main_check_flags_pending_incomplete_blob(
    monkeypatch, tmp_path, download_models, capsys
):
    _prepare_main(monkeypatch, tmp_path, download_models)
    _write_complete_snapshot(tmp_path)
    blobs = tmp_path / "hf-home" / "hub" / "models--black-forest-labs--FLUX.2-klein-4B" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "9f29f9ed.incomplete").write_text("", encoding="utf-8")
    _fake_constant_snapshot(monkeypatch, tmp_path)

    assert download_models.main(["--check", "--only", "gen"]) == 1

    captured = capsys.readouterr()
    assert f"incomplete {GEN_REPO}" in captured.out
    assert "interrupted blob" in captured.out
    assert "rerun without --check" in captured.err


def test_main_check_passes_for_seg_snapshot(monkeypatch, tmp_path, download_models, capsys):
    _prepare_main(monkeypatch, tmp_path, download_models)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pytorch_model.bin").write_text("", encoding="utf-8")
    _fake_constant_snapshot(monkeypatch, tmp_path)

    assert download_models.main(["--check", "--only", "seg"]) == 0

    out = capsys.readouterr().out
    assert f"ok {SEG_REPO}" in out


def test_main_check_flags_seg_snapshot_without_weights(
    monkeypatch, tmp_path, download_models, capsys
):
    _prepare_main(monkeypatch, tmp_path, download_models)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    _fake_constant_snapshot(monkeypatch, tmp_path)

    assert download_models.main(["--check", "--only", "seg"]) == 1

    out = capsys.readouterr().out
    assert f"incomplete {SEG_REPO}" in out
    assert "no weight file" in out


def test_main_check_reports_missing_models(monkeypatch, tmp_path, download_models, capsys):
    _prepare_main(monkeypatch, tmp_path, download_models)

    def fake_snapshot_download(repo_id, **kwargs):
        raise FileNotFoundError("not in the cache")

    _install_fake_hub(monkeypatch, fake_snapshot_download)

    assert download_models.main(["--check"]) == 1

    out = capsys.readouterr().out
    assert out.count("missing ") == 2
