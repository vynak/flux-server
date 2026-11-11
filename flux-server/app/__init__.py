"""HTTP service for interior repainting: segment walls/ceiling, regenerate, composite."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask
from flask_cors import CORS

from .config import Settings
from .pipeline import RepaintPipeline
from .routes import bp

__version__ = "0.1.0"

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"


def create_app(settings: Settings | None = None, pipeline: RepaintPipeline | None = None) -> Flask:
    """Build the application.

    Both arguments exist so tests can inject a fake pipeline and a fixed
    configuration without touching the environment or downloading models.
    """
    settings = settings or Settings.from_env()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_mb * 1024 * 1024

    CORS(app, resources={r"/*": {"origins": list(settings.cors_origins)}})

    app.extensions["repaint"] = pipeline or RepaintPipeline(settings)
    app.register_blueprint(bp)

    @app.errorhandler(413)
    def payload_too_large(_):
        return {"error": f"payload exceeds {settings.max_upload_mb} MB"}, 413

    @app.get("/")
    def demo() -> str:
        return (DEMO_DIR / "index.html").read_text(encoding="utf-8")

    return app
