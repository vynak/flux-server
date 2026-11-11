import base64
import io

import pytest
from PIL import Image


def _encode(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.fixture
def png_base64() -> str:
    return _encode(Image.new("RGB", (64, 48), (120, 130, 140)))


@pytest.fixture
def wide_png_base64() -> str:
    return _encode(Image.new("RGB", (2000, 1000), (90, 20, 20)))
