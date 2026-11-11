
import pytest

from app import create_app
from app.config import Settings


class FakePipeline:
    """Stands in for the model pipeline: no torch, no 12 GB download."""

    def __init__(self, ready: bool = True) -> None:
        self.settings = Settings()
        self.ready = ready
        self.calls: list[dict] = []

    def run(self, image, *, walls, ceiling, wall_color, ceiling_color):
        self.calls.append(
            {"walls": walls, "ceiling": ceiling, "wall": wall_color, "ceiling_color": ceiling_color}
        )
        stats = {
            "width": image.width,
            "height": image.height,
            "wall_pct": 42.0,
            "ceiling_pct": 0.0,
            "elapsed_ms": 7,
            "prompt": "fake",
        }
        return image.copy(), stats


@pytest.fixture
def pipeline():
    return FakePipeline()


@pytest.fixture
def client(pipeline):
    return create_app(Settings(), pipeline=pipeline).test_client()


def _payload(image, **overrides):
    body = {
        "image": image,
        "paintWalls": True,
        "paintCeiling": False,
        "wallColor": {"name": "cream", "hex": "#E8E2D4"},
    }
    body.update(overrides)
    return body


def test_repaint_returns_jpeg_and_stats(client, png_base64):
    response = client.post("/repaint", json=_payload(png_base64))

    assert response.status_code == 200
    body = response.get_json()
    assert body["image"].startswith("data:image/jpeg;base64,")
    assert body["stats"]["wall_pct"] == 42.0


def test_repaint_passes_flags_to_pipeline(client, pipeline, png_base64):
    client.post("/repaint", json=_payload(png_base64))

    assert pipeline.calls[0]["walls"] is True
    assert pipeline.calls[0]["ceiling"] is False


def test_repaint_requires_a_surface(client, png_base64):
    response = client.post("/repaint", json={"image": png_base64})

    assert response.status_code == 400
    assert "paintWalls" in response.get_json()["error"]


def test_repaint_requires_a_colour_for_the_selected_surface(client, png_base64):
    body = _payload(png_base64)
    del body["wallColor"]

    response = client.post("/repaint", json=body)

    assert response.status_code == 400
    assert "wallColor" in response.get_json()["error"]


def test_repaint_rejects_bad_base64(client):
    response = client.post("/repaint", json=_payload("!!! not base64 !!!"))

    assert response.status_code == 400


def test_repaint_returns_503_while_loading(png_base64):
    client = create_app(Settings(), pipeline=FakePipeline(ready=False)).test_client()

    response = client.post("/repaint", json=_payload(png_base64))

    assert response.status_code == 503


def test_repaint_rejects_non_json(client):
    response = client.post("/repaint", data="hello", content_type="text/plain")

    assert response.status_code == 400


def test_healthz_and_readyz(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200


def test_readyz_is_503_while_loading():
    client = create_app(Settings(), pipeline=FakePipeline(ready=False)).test_client()

    with client.get("/readyz") as response:
        assert response.status_code == 503


def test_demo_page_is_served(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"<html" in response.data.lower()
