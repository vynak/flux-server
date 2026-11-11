import base64

import numpy as np
import pytest
from PIL import Image

from app.pipeline import (
    Color,
    RepaintError,
    build_prompt,
    clean_mask,
    composite,
    decode_image,
    encode_jpeg,
    fit_longest_side,
    soften_mask,
)


def test_color_parse_normalizes_whitespace():
    color = Color.parse({"name": "  Dusty   pink ", "hex": "#C9848F"}, "wallColor")
    assert color.name == "dusty pink"
    assert color.hex == "#C9848F"


def test_color_parse_falls_back_to_hex_when_name_is_empty():
    assert Color.parse({"hex": "#123456"}, "wallColor").name == "#123456"


@pytest.mark.parametrize("bad", ["123456", "#12345", "#GGGGGG", ""])
def test_color_parse_rejects_bad_hex(bad):
    with pytest.raises(RepaintError):
        Color.parse({"name": "x", "hex": bad}, "wallColor")


def test_color_parse_rejects_non_object():
    with pytest.raises(RepaintError):
        Color.parse("blue", "wallColor")


def test_build_prompt_mentions_chosen_surface_only():
    prompt = build_prompt(True, False, Color("cream", "#E8E2D4"), None)
    assert "cream (#E8E2D4)" in prompt
    assert "ceiling" not in prompt


def test_build_prompt_requires_a_surface():
    with pytest.raises(RepaintError):
        build_prompt(False, False, None, None)


def test_decode_image_accepts_data_uri(png_base64):
    image = decode_image("data:image/png;base64," + png_base64, max_px=1024)
    assert image.mode == "RGB"
    assert image.size == (64, 48)


def test_decode_image_rejects_garbage():
    with pytest.raises(RepaintError):
        decode_image("not base64 at all!!", max_px=1024)


def test_decode_image_rejects_non_image_payload():
    with pytest.raises(RepaintError):
        decode_image(base64.b64encode(b"still not an image").decode(), max_px=1024)


def test_decode_image_downscales_large_input(wide_png_base64):
    image = decode_image(wide_png_base64, max_px=512)
    assert max(image.size) == 512


def test_fit_longest_side_never_upscales():
    image = Image.new("RGB", (100, 50))
    assert fit_longest_side(image, 1024) is image


def test_clean_mask_drops_specks_and_keeps_the_block():
    mask = np.zeros((200, 200), dtype=bool)
    mask[40:140, 40:140] = True
    mask[2:5, 2:5] = True

    cleaned = clean_mask(mask, min_area_ratio=0.01)

    assert cleaned[70, 70]
    assert not cleaned[3, 3]
    assert 9000 < cleaned.sum() <= 10000


def test_soften_mask_is_float_in_unit_range():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True

    alpha = soften_mask(mask)

    assert alpha.dtype == np.float32
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0


def test_composite_respects_alpha():
    original = np.zeros((2, 2, 3), dtype=np.uint8)
    generated = np.full((2, 2, 3), 200, dtype=np.uint8)
    alpha = np.full((2, 2), 0.5, dtype=np.float32)

    blended = composite(original, generated, alpha)

    assert np.allclose(blended, 100, atol=1)


def test_encode_jpeg_returns_data_uri():
    uri = encode_jpeg(Image.new("RGB", (8, 8), (10, 20, 30)), quality=80)
    assert uri.startswith("data:image/jpeg;base64,")
    assert len(base64.b64decode(uri.split(",", 1)[1])) > 0
