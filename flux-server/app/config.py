"""Runtime configuration. Every value can be overridden by an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_origins() -> tuple[str, ...]:
    raw = os.getenv("FLUX_CORS_ORIGINS", "*")
    return tuple(item.strip() for item in raw.split(",") if item.strip()) or ("*",)


@dataclass(frozen=True)
class Settings:
    seg_model: str = "nvidia/segformer-b5-finetuned-ade-640-640"
    gen_model: str = "black-forest-labs/FLUX.2-klein-4B"
    device: str = "cpu"
    # "auto" picks float16 on CUDA and bfloat16 everywhere else: a 4B model in
    # float32 needs ~16 GB of RAM just for the weights.
    dtype: str = "auto"
    # Attempts when a model download is interrupted (env FLUX_DOWNLOAD_RETRIES).
    download_retries: int = 5

    # Working resolution: the input is downscaled so its longest side fits here.
    max_input_px: int = 1024
    # Diffusion resolution. Multiples of 8 keep the VAE happy; keeps CPU runs bounded.
    gen_size: int = 512
    steps: int = 4
    guidance: float = 0.0
    jpeg_quality: int = 92

    max_upload_mb: int = 12
    cors_origins: tuple[str, ...] = ("*",)
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            seg_model=os.getenv("FLUX_SEG_MODEL", cls.seg_model),
            gen_model=os.getenv("FLUX_GEN_MODEL", cls.gen_model),
            device=os.getenv("FLUX_DEVICE", cls.device),
            dtype=os.getenv("FLUX_DTYPE", cls.dtype),
            download_retries=max(1, _env_int("FLUX_DOWNLOAD_RETRIES", cls.download_retries)),
            max_input_px=_env_int("FLUX_MAX_INPUT_PX", cls.max_input_px),
            gen_size=_env_int("FLUX_GEN_SIZE", cls.gen_size),
            steps=_env_int("FLUX_STEPS", cls.steps),
            guidance=_env_float("FLUX_GUIDANCE", cls.guidance),
            jpeg_quality=_env_int("FLUX_JPEG_QUALITY", cls.jpeg_quality),
            max_upload_mb=_env_int("FLUX_MAX_UPLOAD_MB", cls.max_upload_mb),
            cors_origins=_env_origins(),
            log_level=os.getenv("FLUX_LOG_LEVEL", cls.log_level).upper(),
        )
