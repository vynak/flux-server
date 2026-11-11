# flux-server

Change the colour of the walls or the ceiling in a photo of a room, and keep everything else exactly where it was.

Two models do the work:

- **SegFormer** finds the walls and the ceiling.
- **FLUX** redraws the room with the colour you picked.

The new pixels are pasted back over the original photo using the mask, with soft edges. Furniture, windows and shadows stay untouched. If you just ask an image model to "repaint this room", it also moves the furniture; here it can't.

```
upload ──► SegFormer ──► mask ──┐
   │                            ├──► blend ──► JPEG
   └────────► FLUX ─────────────┘
```

## How it works

1. Decode the upload and resize it so the long side is at most 1024 px (configurable).
2. Segment the image and keep the wall and/or ceiling areas. Small noise is removed and the border is blurred so the cut is not visible.
3. Generate a new version of the image with FLUX, using the original as a starting point.
4. Blend the generated image over the original only inside the mask, and return a JPEG.

## What you need

- Python 3.10 or newer.
- About 16 GB of free disk for the model weights.
- 16 GB of RAM minimum. 32 GB is comfortable.
- A CPU is enough. A CUDA GPU is optional.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

# CPU build of torch (~200 MB).
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
python wsgi.py
```

On Windows, if PowerShell blocks the activation script, skip it and call the venv's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe wsgi.py
```

Then open <http://localhost:8000>. The first start downloads the weights, so `/readyz` returns `503` until they are in memory; once it answers `{"status":"ready"}`, the demo works.

## Download the models first (optional)

The server downloads what it needs on the first run, but doing it before is easier to watch and easier to resume:

```bash
python scripts/download_models.py            # both models
python scripts/download_models.py --only gen # only FLUX (~16 GB)
python scripts/download_models.py --check    # verify what is cached, download nothing
```

Downloads are resumable. If the connection drops, run the same command again: it continues from the bytes already on disk. `Ctrl-C` is safe for the same reason. The weights live in `.hf-cache/` next to the code.

## Docker

```bash
docker compose up --build
```

The cache is kept in a named volume, so the big download happens only once.

## API

### `POST /repaint`

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "paintWalls": true,
  "paintCeiling": false,
  "wallColor": { "name": "cream", "hex": "#E8E2D4" }
}
```

| Field | Type | Notes |
|---|---|---|
| `image` | string | Base64 or data URI. Resized to the working size. |
| `paintWalls` | bool | At least one of `paintWalls` / `paintCeiling` must be `true`. |
| `paintCeiling` | bool | |
| `wallColor` | object | Required with `paintWalls`. `name` is used in the prompt, `hex` is `#RRGGBB`. |
| `ceilingColor` | object | Required with `paintCeiling`. |

Answer `200`:

```json
{
  "image": "data:image/jpeg;base64,...",
  "stats": { "width": 1024, "height": 768, "wall_pct": 23.4, "ceiling_pct": 0.0, "elapsed_ms": 41230 }
}
```

Errors return `{"error": "..."}` with `400` (bad input), `413` (too large), `503` (models not ready) or `500`.

`GET /healthz` says the process is alive. `GET /readyz` says the models are loaded.

There is also a small client, useful to test without the browser:

```bash
python scripts/smoke.py room.jpg --walls --wall-hex "#E8E2D4" --wall-name cream
```

## Settings

All settings are environment variables. Start from [`.env.example`](.env.example).

| Variable | Default | What it does |
|---|---|---|
| `FLUX_SEG_MODEL` | `nvidia/segformer-b5-finetuned-ade-640-640` | Segmentation model |
| `FLUX_GEN_MODEL` | `black-forest-labs/FLUX.2-klein-4B` | Image model |
| `FLUX_DEVICE` | `cpu` | `cpu`, `cuda` or `mps` |
| `FLUX_DTYPE` | `auto` | `auto` = bfloat16 on CPU, float16 on CUDA |
| `FLUX_DOWNLOAD_RETRIES` | `5` | Extra attempts when a download is cut |
| `FLUX_MAX_INPUT_PX` | `1024` | Long side of the working image |
| `FLUX_GEN_SIZE` | `512` | Size used for generation, multiple of 8 |
| `FLUX_STEPS` | `4` | Diffusion steps |
| `FLUX_MAX_UPLOAD_MB` | `12` | Maximum request body |
| `FLUX_CORS_ORIGINS` | `*` | Allowed origins, comma separated |

## Speed and memory

This runs on CPU. Segmentation takes about 2 seconds for a 700 px photo; generation takes **minutes** per image. That is the cost of a 4B model without a GPU. The API returns the result inline as base64, which keeps the client simple at the price of a bigger response.

Memory matters more than speed: the model and its text encoder have to fit in RAM at the same time. That is why `FLUX_DTYPE=auto` uses bfloat16 on CPU — in float32 the weights alone ask for about 16 GB.

Requests are handled one at a time, because one generation can hold several GB of memory. If you deploy with gunicorn, use **one** worker and more threads; every extra worker loads another copy of the model.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests replace the models with a fake pipeline, so they run in under a second, without torch and without downloading anything. Lint with `ruff check .`. CI runs both on every push.

## Project layout

```
app/
  __init__.py     builds the app, CORS, error handlers
  config.py       settings from the environment
  pipeline.py     decode, segment, mask, generate, blend
  routes.py       /repaint, /healthz, /readyz
demo/             one-page demo, no build step
scripts/
  smoke.py        command line client for a running server
  download_models.py  download or check the weights
tests/            tests for the image code and the HTTP layer
wsgi.py           entrypoint
```

## Common problems

**`No module named 'flask'`** — you are using the system Python. Call the venv's interpreter: `.\.venv\Scripts\python.exe wsgi.py`.

**`No module named 'torch'`** — torch is installed separately because the package depends on your hardware:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**The download stops every few minutes** — the network is dropping the transfer. It is resumable, so run the command again; the server also retries on its own (`FLUX_DOWNLOAD_RETRIES`). Do not enable `HF_HUB_ENABLE_HF_TRANSFER`: with the installed hub it disables resuming and every attempt starts from zero. If your connection is bad, `pip install hf_xet` uses a more reliable transfer.

**`PermissionError` in `~/.cache/huggingface`** — the folder cannot be written. `wsgi.py` already points the cache to `./.hf-cache`, so this usually means you started the app another way; add `HF_HOME=.hf-cache` to `.env`.

**I only want to see the page** — run without models. The UI loads, `/repaint` answers `503`:

```bash
flask --app app run --port 8000
```

## Limitations

- Only two categories are used: wall and ceiling. Floors, facades and roofs are not supported.
- The mask is not smart: anything standing in front of the wall but inside the mask gets repainted too (a picture frame, for example).
- Output size is capped by `FLUX_MAX_INPUT_PX`.
- One request at a time. There is no queue, so a second request waits for the first.

## License

MIT. See [LICENSE](LICENSE).
