"""Send an image to a running server and write the result to disk.

    python scripts/smoke.py room.jpg --walls --wall-hex "#E8E2D4" --wall-name cream
    python scripts/smoke.py room.jpg --ceiling --ceiling-hex "#F4F4F4" --out out/ceiling.jpg

Needs a server already running (`python wsgi.py`).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def build_payload(args: argparse.Namespace) -> dict:
    image = base64.b64encode(Path(args.image).read_bytes()).decode("ascii")
    payload: dict = {
        "image": image,
        "paintWalls": args.walls,
        "paintCeiling": args.ceiling,
    }
    if args.walls:
        payload["wallColor"] = {"name": args.wall_name, "hex": args.wall_hex}
    if args.ceiling:
        payload["ceilingColor"] = {"name": args.ceiling_name, "hex": args.ceiling_hex}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="path to a room photo")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--out", default="out/result.jpg")
    parser.add_argument("--walls", action="store_true")
    parser.add_argument("--ceiling", action="store_true")
    parser.add_argument("--wall-hex", default="#E8E2D4")
    parser.add_argument("--wall-name", default="cream")
    parser.add_argument("--ceiling-hex", default="#F4F4F4")
    parser.add_argument("--ceiling-name", default="white")
    args = parser.parse_args()

    if not args.walls and not args.ceiling:
        parser.error("pick --walls, --ceiling or both")

    request = urllib.request.Request(
        f"{args.url.rstrip('/')}/repaint",
        data=json.dumps(build_payload(args)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        print(f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')}", file=sys.stderr)
        return 1
    except urllib.error.URLError as error:
        print(f"cannot reach {args.url}: {error.reason}", file=sys.stderr)
        return 1

    raw = base64.b64decode(body["image"].split(",", 1)[1])
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)

    stats = body.get("stats", {})
    print(f"saved {destination} ({len(raw) // 1024} KB)")
    print(
        f"walls {stats.get('wall_pct')}% | ceiling {stats.get('ceiling_pct')}%"
        f" | {stats.get('elapsed_ms')} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
