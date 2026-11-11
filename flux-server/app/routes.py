"""HTTP layer: validation, error mapping, JSON in / JSON out."""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, request

from .pipeline import Color, NotReady, RepaintError, decode_image, encode_jpeg

log = logging.getLogger(__name__)

bp = Blueprint("api", __name__)


def _error(message: str, status: int):
    return jsonify(error=message), status


@bp.post("/repaint")
def repaint():
    pipeline = current_app.extensions["repaint"]
    settings = pipeline.settings

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error("body must be a JSON object", 400)

    walls = payload.get("paintWalls") is True
    ceiling = payload.get("paintCeiling") is True
    if not walls and not ceiling:
        return _error("set paintWalls or paintCeiling to true", 400)

    try:
        wall_color = Color.parse(payload["wallColor"], "wallColor") if walls else None
        ceiling_color = Color.parse(payload["ceilingColor"], "ceilingColor") if ceiling else None
        image = decode_image(payload.get("image"), settings.max_input_px)
    except KeyError as exc:
        return _error(f"{exc.args[0]} is required", 400)
    except RepaintError as exc:
        return _error(str(exc), 400)

    if not pipeline.ready:
        return _error("models are still loading", 503)

    try:
        result, stats = pipeline.run(
            image,
            walls=walls,
            ceiling=ceiling,
            wall_color=wall_color,
            ceiling_color=ceiling_color,
        )
    except NotReady:
        return _error("models are still loading", 503)
    except RepaintError as exc:
        return _error(str(exc), 400)
    except Exception:
        log.exception("repaint failed")
        return _error("generation failed", 500)

    return jsonify(image=encode_jpeg(result, settings.jpeg_quality), stats=stats), 200


@bp.get("/healthz")
def healthz():
    """Liveness: the process answers."""
    return jsonify(status="ok"), 200


@bp.get("/readyz")
def readyz():
    """Readiness: the models are in memory and inference can start."""
    pipeline = current_app.extensions["repaint"]
    if pipeline.ready:
        return jsonify(status="ready"), 200
    return jsonify(status="loading"), 503
