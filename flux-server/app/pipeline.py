"""Segment → regenerate → composite.

The heavy dependencies (torch, transformers, diffusers) are imported inside
:meth:`RepaintPipeline.load` so the rest of the package — and the test suite —
can be imported on a machine without a 2 GB download.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import Settings

log = logging.getLogger(__name__)

# ADE20K indices used by the SegFormer checkpoint.
WALL_CLASS = 0
CEILING_CLASS = 5

HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
COLOR_NAME_MAX = 48


# --------------------------------------------------------------------------- #
# Download retries
# --------------------------------------------------------------------------- #

# Exception type names that mean "the transfer broke, try again". They are matched
# by name so requests/urllib3/httpx are not imported at module scope: this module
# must stay importable on a machine without the heavy runtime dependencies.
_TRANSIENT_DOWNLOAD_ERRORS = frozenset(
    {
        # requests / urllib3 (huggingface_hub 0.x)
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
        # httpx (huggingface_hub 2.x)
        "RemoteProtocolError",
        "ConnectError",
        "ReadError",
    }
)


def is_transient_download_error(exc: BaseException) -> bool:
    """True when ``exc``, or an exception it wraps, is an interrupted network transfer."""
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending and len(seen) < 20:  # bounded: exception chains can be long or cyclic
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if type(node).__name__ in _TRANSIENT_DOWNLOAD_ERRORS:
            return True
        for linked in (node.__cause__, node.__context__):
            if linked is not None:
                pending.append(linked)
    return False


def _load_with_retries(
    what: str,
    loader: Callable[[], Any],
    *,
    attempts: int,
    sleep: Callable[[float], object] = time.sleep,
) -> Any:
    """Run ``loader``, retrying an interrupted download up to ``attempts`` times.

    huggingface_hub resumes the partial blob from the cache on the next attempt, so
    a retry normally continues the download instead of restarting it from zero.
    """
    attempts = max(1, int(attempts))
    for attempt in range(1, attempts + 1):
        try:
            return loader()
        except Exception as exc:
            if not is_transient_download_error(exc):
                raise RuntimeError(f"Could not load {what}: {type(exc).__name__}: {exc}") from exc
            if attempt == attempts:
                raise RuntimeError(
                    f"Could not load {what} (attempt {attempt} of {attempts}): the network "
                    "connection was interrupted while downloading the model weights. The "
                    "partial download is kept in the Hugging Face cache, so simply starting "
                    "the server again resumes from where it stopped; you can also "
                    "pre-download with: python scripts/download_models.py. Original error: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            log.warning(
                "download interrupted while loading %s (attempt %d/%d): %s — "
                "retrying, the partial file is resumed",
                what,
                attempt,
                attempts,
                exc,
            )
            sleep(min(2 ** (attempt - 1), 30))


class RepaintError(ValueError):
    """The request is well-formed but cannot be fulfilled (bad image, empty mask...)."""


class NotReady(RuntimeError):
    """Models are still loading."""


@dataclass(frozen=True)
class Color:
    name: str
    hex: str

    @classmethod
    def parse(cls, raw: object, field: str) -> Color:
        if not isinstance(raw, dict):
            raise RepaintError(f"{field} must be an object with 'name' and 'hex'")

        hex_value = str(raw.get("hex", "")).strip()
        if not HEX_RE.match(hex_value):
            raise RepaintError(f"{field}.hex must look like #RRGGBB")

        name = " ".join(str(raw.get("name", "")).split()).lower()[:COLOR_NAME_MAX] or hex_value
        return cls(name=name, hex=hex_value)


# --------------------------------------------------------------------------- #
# Image helpers (pure, no model)
# --------------------------------------------------------------------------- #


def fit_longest_side(image: Image.Image, limit: int) -> Image.Image:
    """Downscale so the longest side fits ``limit``. Never upscales."""
    if limit <= 0 or max(image.size) <= limit:
        return image
    scale = limit / max(image.size)
    size = (max(8, round(image.width * scale)), max(8, round(image.height * scale)))
    return image.resize(size, Image.LANCZOS)


def decode_image(payload: object, max_px: int) -> Image.Image:
    """Decode a base64 string or data URI into an RGB image capped at ``max_px``."""
    if not isinstance(payload, str) or not payload.strip():
        raise RepaintError("image must be a base64 string")

    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")

    try:
        raw = base64.b64decode("".join(payload.split()), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RepaintError("image is not valid base64") from exc

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:  # Pillow raises a family of errors here
        raise RepaintError("image could not be decoded") from exc

    # Phone photos carry an EXIF orientation that Pillow does not apply by default.
    return fit_longest_side(ImageOps.exif_transpose(image).convert("RGB"), max_px)


def encode_jpeg(image: Image.Image, quality: int) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def clean_mask(
    mask: np.ndarray,
    min_area_ratio: float = 0.01,
    close_ksize: int = 15,
    open_ksize: int = 11,
) -> np.ndarray:
    """Close holes, drop specks and keep components larger than ``min_area_ratio``."""
    uint8 = (mask.astype(np.uint8)) * 255

    closed = cv2.morphologyEx(
        uint8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize,) * 2)
    )
    opened = cv2.morphologyEx(
        closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize,) * 2)
    )

    min_area = opened.shape[0] * opened.shape[1] * min_area_ratio
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened)

    kept = np.zeros_like(opened)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= min_area:
            kept[labels == index] = 255

    return kept > 0


def soften_mask(mask: np.ndarray, ksize: int = 21, sigma: float = 7.0) -> np.ndarray:
    """Feather the edges so the composite does not show a cut-out line."""
    return cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), sigma)


def composite(original: np.ndarray, generated: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Blend ``generated`` over ``original`` using a per-pixel alpha in [0, 1]."""
    weight = alpha[..., None]
    blended = original.astype(np.float32) * (1.0 - weight) + generated.astype(np.float32) * weight
    return np.clip(blended, 0, 255).astype(np.uint8)


def build_prompt(
    walls: bool,
    ceiling: bool,
    wall_color: Color | None,
    ceiling_color: Color | None,
) -> str:
    """Compose the instruction sent to the diffusion model."""
    targets: list[str] = []
    if walls and wall_color:
        targets.append(f"the walls in {wall_color.name} ({wall_color.hex})")
    if ceiling and ceiling_color:
        targets.append(f"the ceiling in {ceiling_color.name} ({ceiling_color.hex})")

    if not targets:
        raise RepaintError("nothing to repaint")

    joined = " and ".join(targets)
    return (
        "Interior photograph of the same room, identical furniture, layout and lighting. "
        f"Repaint only {joined}. "
        "Keep textures, shadows, reflections, windows, doors, fixtures and trim unchanged. "
        "Photorealistic, high quality."
    )


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


class RepaintPipeline:
    """Holds the two models and serializes inference.

    Diffusion on CPU is memory hungry and the underlying pipelines are not
    thread-safe, so requests are executed one at a time behind a lock.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._seg_processor = None
        self._seg_model = None
        self._gen_pipe = None
        self._torch = None
        self._lock = threading.Lock()
        self._ready = threading.Event()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def load(self) -> None:
        try:
            import torch
            from diffusers import DiffusionPipeline
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Dependency '{exc.name}' is missing. Install the runtime requirements:\n"
                "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "    pip install -r requirements.txt\n"
                "On a machine without GPU support, CPU wheels are enough. See the README."
            ) from exc

        settings = self.settings
        dtype_name = settings.dtype
        if dtype_name == "auto":
            dtype_name = "float16" if "cuda" in settings.device else "bfloat16"

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype_name, torch.bfloat16)

        log.info("loading segmentation model %s", settings.seg_model)

        def _load_segmentation() -> None:
            self._seg_processor = SegformerImageProcessor.from_pretrained(settings.seg_model)
            self._seg_model = SegformerForSemanticSegmentation.from_pretrained(settings.seg_model)

        _load_with_retries(
            f"segmentation model {settings.seg_model}",
            _load_segmentation,
            attempts=settings.download_retries,
        )
        self._seg_model.to(settings.device).eval()

        log.info(
            "loading generative model %s (%s, %s)",
            settings.gen_model,
            settings.device,
            settings.dtype,
        )
        self._gen_pipe = _load_with_retries(
            f"generative model {settings.gen_model}",
            lambda: DiffusionPipeline.from_pretrained(settings.gen_model, torch_dtype=dtype),
            attempts=settings.download_retries,
        )
        self._gen_pipe.to(settings.device)
        self._gen_pipe.enable_attention_slicing()

        self._torch = torch
        self._ready.set()
        log.info("models ready")

    # -- internals ---------------------------------------------------------- #

    def _segment(self, image: Image.Image) -> np.ndarray:
        torch = self._torch
        inputs = self._seg_processor(images=image, return_tensors="pt").to(self.settings.device)
        with torch.inference_mode():
            logits = self._seg_model(**inputs).logits
            upsampled = torch.nn.functional.interpolate(
                logits,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )
        return upsampled.argmax(dim=1)[0].cpu().numpy()

    def _mask(
        self, seg_map: np.ndarray, walls: bool, ceiling: bool
    ) -> tuple[np.ndarray, float, float]:
        total = seg_map.size
        combined = np.zeros(seg_map.shape, dtype=bool)
        wall_pct = ceiling_pct = 0.0

        if walls:
            wall_mask = clean_mask(seg_map == WALL_CLASS)
            wall_pct = round(float(wall_mask.sum()) * 100.0 / total, 2)
            combined |= wall_mask
        if ceiling:
            ceiling_mask = clean_mask(seg_map == CEILING_CLASS)
            ceiling_pct = round(float(ceiling_mask.sum()) * 100.0 / total, 2)
            combined |= ceiling_mask

        return combined, wall_pct, ceiling_pct

    def _generate(self, image: Image.Image, prompt: str) -> Image.Image:
        side = max(8, self.settings.gen_size // 8 * 8)
        scale = min(side / max(image.size), 1.0)
        width = max(8, round(image.width * scale) // 8 * 8)
        height = max(8, round(image.height * scale) // 8 * 8)

        result = self._gen_pipe(
            prompt=prompt,
            image=image.resize((width, height), Image.LANCZOS),
            num_inference_steps=self.settings.steps,
            guidance_scale=self.settings.guidance,
        ).images[0]

        return result.resize(image.size, Image.LANCZOS)

    # -- public ------------------------------------------------------------- #

    def run(
        self,
        image: Image.Image,
        *,
        walls: bool,
        ceiling: bool,
        wall_color: Color | None,
        ceiling_color: Color | None,
    ) -> tuple[Image.Image, dict]:
        """Repaint the selected surfaces and return the image plus run stats."""
        if not self.ready:
            raise NotReady("models are still loading")

        prompt = build_prompt(walls, ceiling, wall_color, ceiling_color)
        started = time.perf_counter()

        with self._lock:
            seg_map = self._segment(image)
            mask, wall_pct, ceiling_pct = self._mask(seg_map, walls, ceiling)
            if not mask.any():
                raise RepaintError("no walls or ceiling were detected in this image")

            alpha = soften_mask(mask)
            generated = self._generate(image, prompt)
            result = composite(np.array(image), np.array(generated), alpha)

        stats = {
            "width": image.width,
            "height": image.height,
            "wall_pct": wall_pct,
            "ceiling_pct": ceiling_pct,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "prompt": prompt,
        }
        return Image.fromarray(result), stats
