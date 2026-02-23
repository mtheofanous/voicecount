# asr_google.py
from __future__ import annotations
from pydub import AudioSegment
import io
import struct
from typing import List, Optional
import os
from core.config import ensure_google_credentials_file

# Module-level singleton — avoids a new gRPC connection on every voice call.
_speech_client = None

def _get_speech_client():
    global _speech_client
    if _speech_client is None:
        from google.cloud import speech  # type: ignore
        _speech_client = speech.SpeechClient()
    return _speech_client



def _looks_like_wav(b: bytes) -> bool:
    return len(b) >= 12 and b[0:4] == b"RIFF" and b[8:12] == b"WAVE"


def _parse_wav_header(b: bytes) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Returns (sample_rate, channels, bits_per_sample) if available.
    Minimal WAV parser: looks for 'fmt ' chunk.
    """
    if not _looks_like_wav(b):
        return None, None, None

    # WAV chunks start after 12 bytes
    i = 12
    sample_rate = channels = bits = None

    try:
        while i + 8 <= len(b):
            chunk_id = b[i : i + 4]
            chunk_size = struct.unpack("<I", b[i + 4 : i + 8])[0]
            i += 8

            if chunk_id == b"fmt " and chunk_size >= 16 and i + chunk_size <= len(b):
                fmt = b[i : i + chunk_size]
                # fmt layout (PCM):
                # 0:2 audio_format, 2:4 num_channels, 4:8 sample_rate,
                # 14:16 bits_per_sample
                channels = struct.unpack("<H", fmt[2:4])[0]
                sample_rate = struct.unpack("<I", fmt[4:8])[0]
                bits = struct.unpack("<H", fmt[14:16])[0]
                return sample_rate, channels, bits

            i += chunk_size
    except Exception:
        pass

    return sample_rate, channels, bits

def convert_to_wav_linear16(audio_bytes: bytes) -> bytes:
    """
    Convert audio bytes (wav/webm/ogg/m4a, any bit depth)
    to WAV LINEAR16 mono 16kHz (Google STT compatible).
    """
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

    audio = (
        audio
        .set_frame_rate(16000)
        .set_channels(1)
        .set_sample_width(2)  # 2 bytes = 16-bit
    )

    buf = io.BytesIO()
    audio.export(buf, format="wav")
    return buf.getvalue()

def _guess_google_language_code(language: str, vocab: List[str]) -> str:
    """
    Map your app language codes to Google language_code.
    If 'auto', guess based on presence of Greek characters in vocab.
    """
    lang = (language or "").strip().lower()

    if lang in ("el", "el-gr", "greek"):
        return "el-GR"
    if lang in ("es", "es-es", "spanish"):
        return "es-ES"
    if lang in ("en", "en-us", "english"):
        return "en-US"

    # auto / fallback
    has_greek = any(any("\u0370" <= ch <= "\u03FF" for ch in (w or "")) for w in (vocab or []))
    return "el-GR" if has_greek else "es-ES"


def asr_google(audio_bytes: bytes, vocab: List[str], language: str = "es") -> str:
    """
    Google Cloud Speech-to-Text transcription.

    Works locally (GOOGLE_APPLICATION_CREDENTIALS points to a JSON file)
    and on Streamlit Cloud (GOOGLE_CREDENTIALS_JSON stored in secrets).
    """

    # Credentials are set up once during bootstrap_once() in app.py.
    # Only re-check here if the env var is somehow missing (e.g. first call before bootstrap).
    credentials_path = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    if not credentials_path:
        ensure_google_credentials_file()
        credentials_path = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    if not credentials_path:
        raise RuntimeError(
            "Google Speech credentials missing.\n"
            "On Streamlit Cloud: set GOOGLE_CREDENTIALS_JSON in Secrets.\n"
            "Locally: set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON file path."
        )

    if not audio_bytes:
        return ""

    try:
        from google.cloud import speech  # type: ignore  # noqa: F401 (import check only)
    except Exception as e:
        raise RuntimeError("Missing dependency: pip install google-cloud-speech") from e

    # Google needs language_code
    language_code = _guess_google_language_code(language, vocab)

    # Speech contexts (catalog phrases)
    phrases = [p for p in (vocab or []) if isinstance(p, str) and p.strip()]
    phrases = phrases[:500]
    speech_contexts = [speech.SpeechContext(phrases=phrases, boost=15.0)] if phrases else []

    # Convert audio to Google-compatible WAV (16-bit PCM)
    audio_bytes = convert_to_wav_linear16(audio_bytes)

    # Parse WAV header
    sr, ch, bits = _parse_wav_header(audio_bytes)

    if not _looks_like_wav(audio_bytes):
        raise RuntimeError(
            "Google ASR backend expects WAV/LINEAR16 audio_bytes. "
            "Convert to 16kHz mono WAV before calling asr_google()."
        )

    sample_rate_hz = int(sr or 16000)
    channels = int(ch or 1)

    client = _get_speech_client()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate_hz,
        language_code=language_code,
        max_alternatives=1,
        enable_automatic_punctuation=False,
        audio_channel_count=channels,
        speech_contexts=speech_contexts,
    )

    audio = speech.RecognitionAudio(content=audio_bytes)
    resp = client.recognize(config=config, audio=audio)

    best = ""
    best_conf = -1.0
    for result in resp.results:
        if not result.alternatives:
            continue
        alt = result.alternatives[0]
        txt = (alt.transcript or "").strip()
        conf = float(getattr(alt, "confidence", 0.0) or 0.0)
        if txt and conf > best_conf:
            best, best_conf = txt, conf

    return best