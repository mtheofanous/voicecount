# asr_google.py
from __future__ import annotations
from pydub import AudioSegment
import io
import struct
from typing import List, Optional
import os



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
    Drop-in replacement interface:
        asr_google(audio_bytes: bytes, vocab: List[str], language: str = "es") -> str

    Requirements:
      pip install google-cloud-speech
      and Google credentials via Application Default Credentials:
        export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service_account.json"
      OR run in an environment with ADC set up (e.g., GCP).
    """


    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError(
            "Google Speech credentials missing.\n"
            "Set GOOGLE_APPLICATION_CREDENTIALS in your environment "
            "or .env file and restart Streamlit."
        )
        
    

    if not audio_bytes:
        return ""

    try:
        from google.cloud import speech  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: pip install google-cloud-speech") from e

    # Google needs language_code
    language_code = _guess_google_language_code(language, vocab)

    # Use catalog phrases as Speech Contexts (very effective for product names)
    # Keep it bounded; huge lists can slow / reduce quality.
    phrases = [p for p in (vocab or []) if isinstance(p, str) and p.strip()]
    phrases = phrases[:500]  # safe cap

    speech_contexts = []
    if phrases:
        speech_contexts = [speech.SpeechContext(phrases=phrases, boost=15.0)]

    # Detect WAV details if possible
    # Convert audio to Google-compatible WAV (16-bit PCM)
    audio_bytes = convert_to_wav_linear16(audio_bytes)

    # Now parse WAV header
    sr, ch, bits = _parse_wav_header(audio_bytes)


    # If not WAV, you have two options:
    #  1) Ensure Streamlit records WAV (recommended; your mic input already requests sample_rate=16000)
    #  2) Convert with ffmpeg/pydub before calling this function
    if not _looks_like_wav(audio_bytes):
        raise RuntimeError(
            "Google ASR backend expects WAV/LINEAR16 audio_bytes. "
            "Your audio does not look like WAV. Convert to 16kHz mono WAV before calling asr_google()."
        )

    # Reasonable defaults if header parsing fails
    sample_rate_hz = int(sr or 16000)
    channels = int(ch or 1)
    bits_per_sample = int(bits or 16)


    client = speech.SpeechClient()

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

    # Pick best transcript
    best = ""
    best_conf = -1.0
    for result in resp.results:
        if not result.alternatives:
            continue
        alt = result.alternatives[0]
        txt = (alt.transcript or "").strip()
        conf = float(getattr(alt, "confidence", 0.0) or 0.0)
        # Sometimes confidence is 0.0; still keep first non-empty
        if txt and (conf > best_conf):
            best, best_conf = txt, conf

    return best
