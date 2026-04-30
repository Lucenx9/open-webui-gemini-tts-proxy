import asyncio
import json
import logging
import os
import re
import subprocess
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DEFAULT_MODEL = os.getenv("GEMINI_TTS_MODEL", "google/gemini-3.1-flash-tts-preview")
DEFAULT_VOICE = os.getenv("GEMINI_TTS_VOICE", "Charon")
OPENROUTER_TIMEOUT_SECONDS = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "180"))
OPENROUTER_RETRIES = int(os.getenv("OPENROUTER_RETRIES", "2"))
OPENROUTER_CONCURRENCY = int(os.getenv("OPENROUTER_CONCURRENCY", "2"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "16000"))
MAX_CHUNKS_PER_REQUEST = int(os.getenv("MAX_CHUNKS_PER_REQUEST", "64"))
MAX_CHARS_PER_UPSTREAM_REQUEST = int(os.getenv("MAX_CHARS_PER_UPSTREAM_REQUEST", "320"))
SILENCE_BETWEEN_CHUNKS_MS = int(os.getenv("SILENCE_BETWEEN_CHUNKS_MS", "80"))
PCM_EDGE_FADE_MS = int(os.getenv("PCM_EDGE_FADE_MS", "12"))
MP3_BITRATE = os.getenv("MP3_BITRATE", "192k")
MP3_SAMPLE_RATE = int(os.getenv("MP3_SAMPLE_RATE", "44100"))
FFMPEG_AUDIO_FILTER = os.getenv(
    "FFMPEG_AUDIO_FILTER",
    "highpass=f=80,equalizer=f=3500:t=q:w=1.0:g=1.5",
)

SUPPORTED_MODELS = [
    "google/gemini-3.1-flash-tts-preview",
    "google/gemini-2.5-flash-preview-tts",
    "google/gemini-2.5-pro-preview-tts",
]

GEMINI_VOICES = {
    "Zephyr": "Bright",
    "Puck": "Upbeat",
    "Charon": "Informative",
    "Kore": "Firm",
    "Fenrir": "Excitable",
    "Leda": "Youthful",
    "Orus": "Firm",
    "Aoede": "Breezy",
    "Callirrhoe": "Easy-going",
    "Autonoe": "Bright",
    "Enceladus": "Breathy",
    "Iapetus": "Clear",
    "Umbriel": "Easy-going",
    "Algieba": "Smooth",
    "Despina": "Smooth",
    "Erinome": "Clear",
    "Algenib": "Gravelly",
    "Rasalgethi": "Informative",
    "Laomedeia": "Upbeat",
    "Achernar": "Soft",
    "Alnilam": "Firm",
    "Schedar": "Even",
    "Gacrux": "Mature",
    "Pulcherrima": "Forward",
    "Achird": "Friendly",
    "Zubenelgenubi": "Casual",
    "Vindemiatrix": "Gentle",
    "Sadachbia": "Lively",
    "Sadaltager": "Knowledgeable",
    "Sulafat": "Warm",
}

app = FastAPI(title="Open WebUI Gemini TTS Proxy", version="1.0.0")
log = logging.getLogger("gemini_tts_proxy")
log.setLevel(LOG_LEVEL)
openrouter_semaphore = asyncio.Semaphore(OPENROUTER_CONCURRENCY)
stats: dict[str, Any] = {
    "speech_requests": 0,
    "upstream_requests": 0,
    "upstream_timeouts": 0,
    "upstream_errors": 0,
    "retries": 0,
    "last_error": "",
}


def _voices_payload() -> dict[str, list[dict[str, str]]]:
    return {
        "voices": [
            {"id": voice, "name": f"{voice} - {style}"}
            for voice, style in GEMINI_VOICES.items()
        ]
    }


def _models_payload() -> dict[str, list[dict[str, Any]]]:
    return {
        "models": [{"id": model, "name": model} for model in SUPPORTED_MODELS],
        "data": [
            {
                "id": model,
                "object": "model",
                "created": 0,
                "owned_by": "openrouter",
            }
            for model in SUPPORTED_MODELS
        ],
    }


def _parse_audio_params(content_type: str) -> tuple[int, int]:
    rate_match = re.search(r"(?:rate|samplerate)=(\d+)", content_type)
    channels_match = re.search(r"channels=(\d+)", content_type)
    sample_rate = int(rate_match.group(1)) if rate_match else 24000
    channels = int(channels_match.group(1)) if channels_match else 1
    return sample_rate, channels


def _split_text(text: str, max_chars: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining.strip())
            break

        window = remaining[:max_chars]
        split_at = max(
            window.rfind(". "),
            window.rfind("! "),
            window.rfind("? "),
            window.rfind("; "),
            window.rfind(": "),
            window.rfind(", "),
        )
        if split_at < max_chars // 2:
            split_at = window.rfind(" ")
        if split_at < max_chars // 3:
            split_at = max_chars

        chunk = remaining[:split_at].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].strip()

    return chunks


def _silence_pcm(sample_rate: int, channels: int, duration_ms: int) -> bytes:
    frame_count = max(int(sample_rate * duration_ms / 1000), 0)
    return b"\x00" * frame_count * channels * 2


def _fade_pcm_edges(pcm: bytes, sample_rate: int, channels: int, fade_ms: int) -> bytes:
    if not pcm or fade_ms <= 0 or channels <= 0:
        return pcm

    sample_width = 2
    frame_size = channels * sample_width
    total_frames = len(pcm) // frame_size
    fade_frames = min(int(sample_rate * fade_ms / 1000), total_frames // 2)
    if fade_frames <= 1:
        return pcm

    aligned_length = total_frames * frame_size
    data = bytearray(pcm[:aligned_length])
    remainder = pcm[aligned_length:]

    def scale_sample(position: int, gain: float) -> None:
        sample = int.from_bytes(data[position : position + sample_width], "little", signed=True)
        faded = max(min(int(sample * gain), 32767), -32768)
        data[position : position + sample_width] = faded.to_bytes(sample_width, "little", signed=True)

    for frame in range(fade_frames):
        fade_in_gain = frame / fade_frames
        fade_out_gain = (fade_frames - frame - 1) / fade_frames

        start_base = frame * frame_size
        end_base = (total_frames - fade_frames + frame) * frame_size
        for channel in range(channels):
            offset = channel * sample_width
            scale_sample(start_base + offset, fade_in_gain)
            scale_sample(end_base + offset, fade_out_gain)

    return bytes(data) + remainder


def _convert_pcm_to_mp3(pcm: bytes, sample_rate: int, channels: int) -> bytes:
    ffmpeg_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
    ]
    if FFMPEG_AUDIO_FILTER:
        ffmpeg_command.extend(["-af", FFMPEG_AUDIO_FILTER])
    ffmpeg_command.extend(
        [
            "-ar",
            str(MP3_SAMPLE_RATE),
            "-f",
            "mp3",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            MP3_BITRATE,
            "pipe:1",
        ]
    )
    proc = subprocess.run(
        ffmpeg_command,
        input=pcm,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise HTTPException(status_code=502, detail=f"ffmpeg conversion failed: {detail}")
    return proc.stdout


def _error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error)
            return str(data)
    except Exception:
        pass
    return response.text[:1000]


async def _fetch_upstream_audio(upstream_payload: dict[str, Any], voice: str, model: str) -> tuple[bytes, str]:
    text_len = len(str(upstream_payload.get("input", "")))
    started = time.monotonic()
    response: httpx.Response | None = None
    timeout = httpx.Timeout(
        connect=20.0,
        read=OPENROUTER_TIMEOUT_SECONDS,
        write=30.0,
        pool=30.0,
    )

    async with openrouter_semaphore:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(OPENROUTER_RETRIES + 1):
                try:
                    stats["upstream_requests"] += 1
                    response = await client.post(
                        f"{OPENROUTER_BASE_URL}/audio/speech",
                        headers={
                            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:3000"),
                            "X-Title": os.getenv("OPENROUTER_X_TITLE", "Open WebUI Gemini TTS Proxy"),
                        },
                        json=upstream_payload,
                    )
                    if response.status_code < 500 and response.status_code not in (408, 429):
                        break
                    if attempt >= OPENROUTER_RETRIES:
                        break

                    stats["retries"] += 1
                    stats["upstream_errors"] += 1
                    stats["last_error"] = f"transient upstream {response.status_code}"
                    log.warning(
                        "retrying upstream status=%s voice=%s model=%s chars=%s attempt=%s",
                        response.status_code,
                        voice,
                        model,
                        text_len,
                        attempt + 1,
                    )
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
                except httpx.TimeoutException as exc:
                    stats["upstream_timeouts"] += 1
                    stats["last_error"] = f"timeout after {OPENROUTER_TIMEOUT_SECONDS:g}s"
                    if attempt >= OPENROUTER_RETRIES:
                        log.warning(
                            "openrouter timeout voice=%s model=%s chars=%s attempts=%s elapsed=%.1fs",
                            voice,
                            model,
                            text_len,
                            attempt + 1,
                            time.monotonic() - started,
                        )
                        raise HTTPException(
                            status_code=504,
                            detail=(
                                "OpenRouter TTS timed out after "
                                f"{attempt + 1} attempt(s), {OPENROUTER_TIMEOUT_SECONDS:g}s read timeout"
                            ),
                        ) from exc
                    stats["retries"] += 1
                    await asyncio.sleep(min(2**attempt, 8))

    if response is None:
        raise HTTPException(status_code=504, detail="OpenRouter TTS did not return a response")

    log.info(
        "tts upstream status=%s voice=%s model=%s chars=%s elapsed=%.1fs",
        response.status_code,
        voice,
        model,
        text_len,
        time.monotonic() - started,
    )

    if response.status_code != 200:
        stats["upstream_errors"] += 1
        stats["last_error"] = f"upstream {response.status_code}: {_error_detail(response)[:160]}"
        raise HTTPException(status_code=response.status_code, detail=_error_detail(response))

    return response.content, response.headers.get("content-type", "")


async def _create_speech(payload: dict[str, Any]) -> bytes:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not configured")

    voice = payload.get("voice") or DEFAULT_VOICE
    if voice not in GEMINI_VOICES:
        available = ", ".join(GEMINI_VOICES)
        raise HTTPException(status_code=400, detail=f"Unsupported Gemini voice '{voice}'. Available: {available}")

    model = payload.get("model") or DEFAULT_MODEL
    if model not in SUPPORTED_MODELS:
        model = DEFAULT_MODEL

    upstream_payload = {
        **payload,
        "model": model,
        "voice": voice,
        "response_format": "pcm",
    }

    stats["speech_requests"] += 1
    input_text = str(upstream_payload.get("input", ""))
    if len(input_text) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Input text is too long: {len(input_text)} chars, max {MAX_INPUT_CHARS}",
        )

    chunks = _split_text(input_text, MAX_CHARS_PER_UPSTREAM_REQUEST)
    if not chunks:
        raise HTTPException(status_code=400, detail="Input text is empty")
    if len(chunks) > MAX_CHUNKS_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Input text produces too many TTS chunks: {len(chunks)}, max {MAX_CHUNKS_PER_REQUEST}",
        )

    if len(chunks) == 1:
        audio, content_type = await _fetch_upstream_audio(upstream_payload, voice, model)
    else:
        log.info(
            "splitting long tts input voice=%s model=%s chars=%s chunks=%s",
            voice,
            model,
            len(input_text),
            len(chunks),
        )
        async def fetch_chunk(idx: int, chunk: str) -> tuple[int, bytes, str]:
            chunk_payload = {**upstream_payload, "input": chunk}
            chunk_audio, content_type = await _fetch_upstream_audio(chunk_payload, voice, model)
            return idx, chunk_audio, content_type

        chunk_results = await asyncio.gather(
            *(fetch_chunk(idx, chunk) for idx, chunk in enumerate(chunks, start=1))
        )

        pcm_parts: list[bytes] = []
        sample_rate = 24000
        channels = 1
        for idx, chunk_audio, content_type in sorted(chunk_results, key=lambda item: item[0]):
            if content_type.startswith("audio/mpeg"):
                raise HTTPException(
                    status_code=502,
                    detail="Unexpected MP3 chunk from upstream while recombining split TTS audio",
                )
            sample_rate, channels = _parse_audio_params(content_type)
            pcm_parts.append(_fade_pcm_edges(chunk_audio, sample_rate, channels, PCM_EDGE_FADE_MS))
            if idx < len(chunks):
                pcm_parts.append(_silence_pcm(sample_rate, channels, SILENCE_BETWEEN_CHUNKS_MS))
        audio = b"".join(pcm_parts)
        content_type = f"audio/pcm;rate={sample_rate};channels={channels}"

    if content_type.startswith("audio/mpeg"):
        return audio

    sample_rate, channels = _parse_audio_params(content_type)
    return await asyncio.to_thread(_convert_pcm_to_mp3, audio, sample_rate, channels)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "default_model": DEFAULT_MODEL,
        "default_voice": DEFAULT_VOICE,
        "voices": len(GEMINI_VOICES),
        "timeout_seconds": OPENROUTER_TIMEOUT_SECONDS,
        "retries": OPENROUTER_RETRIES,
        "concurrency": OPENROUTER_CONCURRENCY,
        "max_input_chars": MAX_INPUT_CHARS,
        "max_chunks_per_request": MAX_CHUNKS_PER_REQUEST,
        "max_chars_per_upstream_request": MAX_CHARS_PER_UPSTREAM_REQUEST,
        "silence_between_chunks_ms": SILENCE_BETWEEN_CHUNKS_MS,
        "pcm_edge_fade_ms": PCM_EDGE_FADE_MS,
        "mp3_bitrate": MP3_BITRATE,
        "mp3_sample_rate": MP3_SAMPLE_RATE,
        "ffmpeg_audio_filter": FFMPEG_AUDIO_FILTER,
        "stats": stats,
    }


@app.get("/v1/models")
@app.get("/models")
async def models() -> dict[str, list[dict[str, Any]]]:
    return _models_payload()


@app.get("/v1/audio/models")
@app.get("/audio/models")
async def audio_models() -> dict[str, list[dict[str, Any]]]:
    return _models_payload()


@app.get("/v1/audio/voices")
@app.get("/audio/voices")
async def audio_voices() -> dict[str, list[dict[str, str]]]:
    return _voices_payload()


@app.post("/v1/audio/speech")
@app.post("/audio/speech")
async def audio_speech(request: Request) -> Response:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

    mp3 = await _create_speech(payload)
    return Response(content=mp3, media_type="audio/mpeg")
