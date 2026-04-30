# Open WebUI Gemini TTS Proxy

Small OpenAI-compatible TTS proxy for using Gemini TTS through OpenRouter in Open WebUI.

OpenRouter's Gemini TTS endpoint returns raw PCM audio. Open WebUI's OpenAI TTS path expects an MP3 file it can cache and serve to the browser. This proxy requests PCM from OpenRouter, converts it to MP3 with `ffmpeg`, and returns `audio/mpeg` to Open WebUI.

It also splits long text into smaller Gemini TTS requests, retries transient upstream failures, recombines the audio in order, and exposes the 30 Gemini voice names to Open WebUI.

## Features

- OpenAI-compatible `/v1/audio/speech`
- `/v1/audio/models` and `/v1/audio/voices` endpoints
- PCM to MP3 conversion via `ffmpeg`
- Long-text splitting and ordered recombination
- Retry support for timeouts, 429, and 5xx responses
- Optional Open WebUI speech cache pruning script
- No API keys stored in the repository

## Requirements

- Python 3.11+
- `ffmpeg`
- OpenRouter API key
- Open WebUI configured with OpenAI-compatible TTS

## Install

```bash
git clone https://github.com/Lucenx9/open-webui-gemini-tts-proxy.git
cd open-webui-gemini-tts-proxy
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:

```bash
OPENROUTER_API_KEY=your-openrouter-key
```

Run:

```bash
./scripts/run.sh
```

By default the proxy listens on port `18792`.

## Open WebUI Setup

In Open WebUI, set TTS to the OpenAI-compatible engine:

- Engine: `openai`
- API Base URL, when Open WebUI runs in Docker: `http://host.docker.internal:18792/v1`
- API Base URL, when Open WebUI runs on the same host without Docker: `http://127.0.0.1:18792/v1`
- API Key: any non-empty placeholder, for example `unused`
- Model: `google/gemini-3.1-flash-tts-preview`
- Voice: one of the Gemini voices, for example `Charon`

If Open WebUI runs in Docker on Linux and `host.docker.internal` is unavailable, add a host mapping to the Open WebUI container or bind the proxy to an address reachable from the container.

## Environment

| Variable | Default | Description |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | empty | Required OpenRouter API key |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter-compatible base URL |
| `GEMINI_TTS_MODEL` | `google/gemini-3.1-flash-tts-preview` | Default TTS model |
| `GEMINI_TTS_VOICE` | `Charon` | Default voice |
| `GEMINI_TTS_PROXY_HOST` | `0.0.0.0` | Host passed to Uvicorn |
| `GEMINI_TTS_PROXY_PORT` | `18792` | Port passed to Uvicorn |
| `OPENROUTER_TIMEOUT_SECONDS` | `180` | Upstream read timeout |
| `OPENROUTER_RETRIES` | `2` | Retries for timeout, 429, and 5xx |
| `OPENROUTER_CONCURRENCY` | `2` | Max concurrent upstream requests |
| `MAX_CHARS_PER_UPSTREAM_REQUEST` | `320` | Long-text split size |
| `SILENCE_BETWEEN_CHUNKS_MS` | `140` | Silence inserted between recombined chunks |
| `OPENROUTER_HTTP_REFERER` | `http://localhost:3000` | Optional OpenRouter referer header |
| `OPENROUTER_X_TITLE` | `Open WebUI Gemini TTS Proxy` | Optional OpenRouter title header |

## Voices

Supported Gemini voices:

`Zephyr`, `Puck`, `Charon`, `Kore`, `Fenrir`, `Leda`, `Orus`, `Aoede`, `Callirrhoe`, `Autonoe`, `Enceladus`, `Iapetus`, `Umbriel`, `Algieba`, `Despina`, `Erinome`, `Algenib`, `Rasalgethi`, `Laomedeia`, `Achernar`, `Alnilam`, `Schedar`, `Gacrux`, `Pulcherrima`, `Achird`, `Zubenelgenubi`, `Vindemiatrix`, `Sadachbia`, `Sadaltager`, `Sulafat`.

## Optional Systemd User Service

Copy the example service:

```bash
mkdir -p ~/.config/systemd/user
cp examples/systemd/gemini-tts-proxy.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now gemini-tts-proxy.service
```

## Optional Open WebUI Speech Cache Pruning

Open WebUI caches generated speech files. The included pruning script removes old files and enforces a maximum cache size.

Defaults:

- cache path: `/app/backend/data/cache/audio/speech`
- max age: `3` days
- max size: `512 MiB`
- container name: `open-webui`

Run manually:

```bash
./scripts/prune-open-webui-speech-cache.sh
```

Or install the example timer:

```bash
cp examples/systemd/open-webui-speech-cache-prune.* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now open-webui-speech-cache-prune.timer
```

## Health Check

```bash
curl http://127.0.0.1:18792/healthz
```
