"""Resilient model loading: transient-download detection, retries, and wiring.

Every test here is offline: the fake torch/diffusers/transformers modules keep
the integration test from touching the real hub or the real weights.
"""

import sys
import types

import pytest
import requests
import urllib3

import app.pipeline as pipeline_module
from app.config import Settings
from app.pipeline import RepaintPipeline, _load_with_retries, is_transient_download_error

# --------------------------------------------------------------------------- #
# is_transient_download_error
# --------------------------------------------------------------------------- #


def test_is_transient_for_chunked_encoding_error_wrapping_protocol_error():
    try:
        try:
            raise urllib3.exceptions.ProtocolError("incomplete read")
        except urllib3.exceptions.ProtocolError as cause:
            raise requests.exceptions.ChunkedEncodingError("chunked body dropped") from cause
    except requests.exceptions.ChunkedEncodingError as exc:
        assert is_transient_download_error(exc)


def test_is_transient_for_bare_connection_reset():
    assert is_transient_download_error(ConnectionResetError("connection reset by peer"))


def test_is_transient_follows_the_cause_chain():
    wrapped = RuntimeError("could not fetch the weights")
    wrapped.__cause__ = requests.exceptions.ChunkedEncodingError("dropped mid-body")
    assert is_transient_download_error(wrapped)


def test_is_transient_follows_the_context_chain():
    wrapped = RuntimeError("could not fetch the weights")
    wrapped.__context__ = ConnectionResetError("reset")
    assert is_transient_download_error(wrapped)


@pytest.mark.parametrize("exc", [ValueError("bad"), KeyError("missing"), RuntimeError("boom")])
def test_is_not_transient_for_programming_errors(exc):
    assert is_transient_download_error(exc) is False


@pytest.mark.parametrize("name", ["RemoteProtocolError", "ConnectError", "ReadError"])
def test_is_transient_for_httpx_style_names(name):
    # huggingface_hub 2.x uses httpx; match by name so httpx need not be installed.
    exc_type = type(name, (Exception,), {})
    assert is_transient_download_error(exc_type("transfer aborted"))


def test_is_transient_is_loop_safe():
    exc = RuntimeError("self-referencing chain")
    exc.__cause__ = exc
    assert is_transient_download_error(exc) is False


# --------------------------------------------------------------------------- #
# _load_with_retries
# --------------------------------------------------------------------------- #


def test_retries_transient_errors_then_succeeds():
    calls = []
    sleeps = []

    def loader():
        calls.append(1)
        if len(calls) < 3:
            raise requests.exceptions.ChunkedEncodingError("dropped mid-body")
        return "loaded"

    result = _load_with_retries("segmentation model demo", loader, attempts=5, sleep=sleeps.append)

    assert result == "loaded"
    assert len(calls) == 3
    assert sleeps == [1, 2]  # exponential backoff, no real sleeping


def test_exhausted_retries_raise_an_actionable_error():
    errors = []
    sleeps = []

    def loader():
        error = requests.exceptions.ChunkedEncodingError("dropped mid-body")
        errors.append(error)
        raise error

    with pytest.raises(RuntimeError) as info:
        _load_with_retries("generative model demo", loader, attempts=5, sleep=sleeps.append)

    message = str(info.value)
    assert "generative model demo" in message
    assert "attempt 5 of 5" in message
    assert "the network connection was interrupted while downloading the model weights" in message
    assert "resumes from where it stopped" in message
    assert "you can also pre-download with: python scripts/download_models.py" in message
    assert "dropped mid-body" in message
    assert info.value.__cause__ is errors[-1]
    assert len(errors) == 5
    assert sleeps == [1, 2, 4, 8]


def test_non_transient_error_is_not_retried():
    calls = []
    sleeps = []

    def loader():
        calls.append(1)
        raise ValueError("bad configuration")

    with pytest.raises(RuntimeError) as info:
        _load_with_retries("demo model", loader, attempts=5, sleep=sleeps.append)

    assert len(calls) == 1
    assert sleeps == []
    assert isinstance(info.value.__cause__, ValueError)

    message = str(info.value)
    assert "Could not load demo model" in message
    assert "ValueError: bad configuration" in message
    assert "network" not in message  # a config error is not the connection's fault


def test_non_transient_failure_after_a_retry_is_not_blamed_on_the_network():
    calls = []
    sleeps = []

    def loader():
        calls.append(1)
        if len(calls) == 1:
            raise requests.exceptions.ChunkedEncodingError("dropped mid-body")
        raise KeyError("missing component")

    with pytest.raises(RuntimeError) as info:
        _load_with_retries("demo model", loader, attempts=5, sleep=sleeps.append)

    assert len(calls) == 2
    assert sleeps == [1]
    message = str(info.value)
    assert "Could not load demo model" in message
    assert "KeyError" in message
    assert "network" not in message


def test_attempts_are_clamped_to_at_least_one():
    calls = []

    def loader():
        calls.append(1)
        raise ValueError("nope")

    with pytest.raises(RuntimeError):
        _load_with_retries("demo model", loader, attempts=0, sleep=lambda _: None)

    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def test_settings_default_download_retries():
    assert Settings().download_retries == 5


@pytest.mark.parametrize(("raw", "expected"), [("3", 3), ("0", 1), ("-2", 1)])
def test_settings_reads_download_retries_from_env(monkeypatch, raw, expected):
    monkeypatch.setenv("FLUX_DOWNLOAD_RETRIES", raw)
    assert Settings.from_env().download_retries == expected


def test_settings_rejects_non_integer_download_retries(monkeypatch):
    monkeypatch.setenv("FLUX_DOWNLOAD_RETRIES", "abc")
    with pytest.raises(ValueError):
        Settings.from_env()


# --------------------------------------------------------------------------- #
# Integration: RepaintPipeline.load() retries one interrupted download
# --------------------------------------------------------------------------- #


def _fake_modules(counter):
    """Build stand-in torch/diffusers/transformers modules and the pipes they create."""
    torch = types.ModuleType("torch")
    torch.float16 = "sentinel-float16"
    torch.bfloat16 = "sentinel-bfloat16"
    torch.float32 = "sentinel-float32"

    class FakePipe:
        def __init__(self):
            self.devices = []
            self.attention_slicing = False

        def to(self, device):
            self.devices.append(device)
            return self

        def enable_attention_slicing(self):
            self.attention_slicing = True

    created = []

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model, torch_dtype=None):
            counter["from_pretrained_calls"] += 1
            if counter["from_pretrained_calls"] == 1:
                raise requests.exceptions.ChunkedEncodingError("dropped mid-body")
            pipe = FakePipe()
            created.append(pipe)
            return pipe

    diffusers = types.ModuleType("diffusers")
    diffusers.DiffusionPipeline = FakeDiffusionPipeline

    class FakeSegmentationModel:
        def __init__(self):
            self.devices = []
            self.evaluated = False

        def to(self, device):
            self.devices.append(device)
            return self

        def eval(self):
            self.evaluated = True
            return self

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model):
            counter["processor_calls"] += 1
            return cls()

    class FakeSegformer:
        @classmethod
        def from_pretrained(cls, model):
            counter["seg_model_calls"] += 1
            return FakeSegmentationModel()

    transformers = types.ModuleType("transformers")
    transformers.SegformerImageProcessor = FakeProcessor
    transformers.SegformerForSemanticSegmentation = FakeSegformer

    return {"torch": torch, "diffusers": diffusers, "transformers": transformers}, created


def test_load_retries_an_interrupted_diffusion_download(monkeypatch):
    counter = {"from_pretrained_calls": 0, "processor_calls": 0, "seg_model_calls": 0}
    modules, created = _fake_modules(counter)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    # Use the real retry logic but never sleep: one attempt must not cost a second.
    real_load_with_retries = pipeline_module._load_with_retries

    def no_sleep_load(what, loader, *, attempts):
        return real_load_with_retries(what, loader, attempts=attempts, sleep=lambda _: None)

    monkeypatch.setattr(pipeline_module, "_load_with_retries", no_sleep_load)

    pipeline = RepaintPipeline(Settings(download_retries=3, dtype="float32"))
    pipeline.load()

    assert pipeline.ready is True
    assert counter["from_pretrained_calls"] == 2  # interrupted, then resumed
    assert pipeline._gen_pipe is created[-1]
    assert pipeline._gen_pipe.devices == ["cpu"]
    assert pipeline._gen_pipe.attention_slicing is True
    assert pipeline._seg_model.devices == ["cpu"]
    assert pipeline._seg_model.evaluated is True
    assert counter["processor_calls"] == 1
    assert counter["seg_model_calls"] == 1
