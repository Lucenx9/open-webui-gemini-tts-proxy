#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${OPEN_WEBUI_CONTAINER:-open-webui}"
CACHE_DIR="${OPEN_WEBUI_SPEECH_CACHE_DIR:-/app/backend/data/cache/audio/speech}"
MAX_AGE_DAYS="${OPEN_WEBUI_SPEECH_CACHE_MAX_AGE_DAYS:-3}"
MAX_BYTES="${OPEN_WEBUI_SPEECH_CACHE_MAX_BYTES:-536870912}"

docker exec -i \
  -e CACHE_DIR="${CACHE_DIR}" \
  -e MAX_AGE_DAYS="${MAX_AGE_DAYS}" \
  -e MAX_BYTES="${MAX_BYTES}" \
  "${CONTAINER}" \
  python - <<'PY'
import os
import time
from pathlib import Path

cache_dir = Path(os.environ["CACHE_DIR"])
max_age_days = float(os.environ["MAX_AGE_DAYS"])
max_bytes = int(os.environ["MAX_BYTES"])

if not cache_dir.exists():
    print(f"speech cache not found: {cache_dir}")
    raise SystemExit(0)

now = time.time()
cutoff = now - (max_age_days * 86400)
deleted_files = 0
deleted_bytes = 0


def unlink(path: Path) -> None:
    global deleted_files, deleted_bytes
    try:
        size = path.stat().st_size
        path.unlink()
    except FileNotFoundError:
        return
    deleted_files += 1
    deleted_bytes += size


for path in cache_dir.iterdir():
    if path.is_file() and path.stat().st_mtime < cutoff:
        unlink(path)

groups: dict[str, dict[str, object]] = {}
for path in cache_dir.iterdir():
    if not path.is_file():
        continue
    entry = groups.setdefault(path.stem, {"files": [], "size": 0, "mtime": 0.0})
    stat = path.stat()
    entry["files"].append(path)
    entry["size"] += stat.st_size
    entry["mtime"] = max(entry["mtime"], stat.st_mtime)

total = sum(int(entry["size"]) for entry in groups.values())
if total > max_bytes:
    for entry in sorted(groups.values(), key=lambda item: float(item["mtime"])):
        if total <= max_bytes:
            break
        for path in entry["files"]:
            size = path.stat().st_size if path.exists() else 0
            unlink(path)
            total -= size

remaining = sum(path.stat().st_size for path in cache_dir.iterdir() if path.is_file())
print(
    f"speech cache pruned: deleted_files={deleted_files} "
    f"deleted_bytes={deleted_bytes} remaining_bytes={remaining}"
)
PY
