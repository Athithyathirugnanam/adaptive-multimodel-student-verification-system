"""WHISPER SPEECH-TO-TEXT SERVICE.

Pipeline responsibility (kept strictly separate from every other component):
    Recorded Video -> Extract Audio (ffmpeg) -> Whisper -> Transcript
                     (+ language + timestamped segments)

The student NEVER types their answer; this module turns the spoken
explanation into the primary textual input used by the verification LLM.
"""

import os
import shutil
import subprocess
import tempfile
import threading
import wave

import numpy as np

# Workaround: CTranslate2's bundled oneDNN/MKL fails to allocate memory on
# some Windows hosts when it spawns its default thread pool. Pin the relevant
# allocators to a single thread (must happen before ctranslate2 loads).
for _v in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "CT2_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


class SpeechRecognitionError(Exception):
    """Raised for microphone/audio/Whisper failures.

    Carries a USER-FRIENDLY message only - raw backend errors and stack
    traces are never exposed to students.
    """

    def __init__(self, user_message, detail=""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


# --------------------------------------------------------------------------- #
# FFMPEG AUDIO EXTRACTION                                                     #
# --------------------------------------------------------------------------- #
def _ffmpeg_binary():
    """Locate ffmpeg: system PATH first, then the bundled imageio binary."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:                                    # optional fallback dependency
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def extract_audio(video_path):
    """Extract a 16 kHz mono PCM WAV track from the recorded video."""
    if not os.path.exists(video_path):
        raise SpeechRecognitionError(
            "Recording not found. Please record and submit again.",
            detail="missing %s" % video_path)

    ffmpeg = _ffmpeg_binary()
    if not ffmpeg:
        raise SpeechRecognitionError(
            "Audio processing is temporarily unavailable. Please try again "
            "later.", detail="ffmpeg binary not found")

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
           "-acodec", "pcm_s16le", wav_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
        if proc.returncode != 0 or not os.path.exists(wav_path):
            raise SpeechRecognitionError(
                "We couldn't extract the audio from your recording. Please "
                "make sure your microphone was enabled and try again.",
                detail=proc.stderr.decode("utf-8", "ignore")[:500])
        return wav_path
    except SpeechRecognitionError:
        raise
    except subprocess.TimeoutExpired:
        _safe_remove_wav(wav_path)
        raise SpeechRecognitionError(
            "Audio processing took too long. Please try a shorter "
            "explanation.", detail="ffmpeg timeout")


# --------------------------------------------------------------------------- #
# AUDIO QUALITY GUARDS                                                        #
# --------------------------------------------------------------------------- #
def analyze_audio_quality(wav_path):
    """Cheap acoustic sanity check used for user-friendly warnings.

    Returns dict(duration, rms, peak, clipping_ratio, quality_label).
    """
    try:
        with wave.open(wav_path, "rb") as wf:
            n_frames = wf.getnframes()
            rate = wf.getframerate() or 16000
            raw = wf.readframes(n_frames)
    except (wave.Error, OSError):
        return {"duration": 0.0, "rms": 0.0, "peak": 0.0,
                "clipping_ratio": 0.0, "quality_label": "unreadable"}

    duration = n_frames / float(rate)
    if not raw:
        return {"duration": 0.0, "rms": 0.0, "peak": 0.0,
                "clipping_ratio": 0.0, "quality_label": "empty"}

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    clipping_ratio = (float(np.mean(np.abs(samples) > 0.985))
                      if samples.size else 0.0)

    if duration < 0.5:
        label = "empty"
    elif rms < 0.0005:
        label = "silent"
    elif rms < 0.003:
        label = "quiet"
    elif rms < 0.02 or clipping_ratio > 0.05:
        label = "fair"
    else:
        label = "good"

    return {
        "duration": round(duration, 2),
        "rms": round(rms, 5),
        "peak": round(peak, 4),
        "clipping_ratio": round(clipping_ratio, 5),
        "quality_label": label,
    }


def guard_audio_quality(quality):
    """Raise friendly errors for unusable recordings (no audio / corrupt track).
    For low-volume or quiet mics, we allow Whisper's built-in VAD to attempt
    transcription first instead of prematurely failing.
    """
    label = quality.get("quality_label")
    if label == "empty":
        raise SpeechRecognitionError(
            "No audio was captured. Please check that your microphone is "
            "enabled and record again.")
    if label == "unreadable":
        raise SpeechRecognitionError(
            "The audio track could not be read. Please record again.")


def _safe_remove_wav(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# WHISPER TRANSCRIPTION                                                       #
# --------------------------------------------------------------------------- #
_LOCAL_WHISPER_MODEL = None
_LOCAL_WHISPER_LOCK = threading.Lock()


def _load_whisper_model(model_size="base"):
    """Load faster-whisper. Tries CUDA first (validating actual runtime
    readiness to catch missing cublas64_*.dlls), falling back cleanly to
    fast int8 CPU with 4 threads."""
    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        # Validate that CUDA libraries (e.g. cublas64_12.dll) can actually be loaded
        import ctranslate2
        dummy = np.zeros((1, 80, 3000), dtype=np.float32)
        model.model.encode(ctranslate2.StorageView.from_array(dummy))
        return model
    except Exception:
        # Fall back to CPU with int8 quantization and 4 threads for sub-2s latency
        return WhisperModel(
            model_size, device="cpu", compute_type="int8", cpu_threads=4)


def warm_up_local_whisper(model_size="base"):
    """Pre-load the cached faster-whisper model so the first student recording
    does not have to pay the model-loading cost inside the request."""
    global _LOCAL_WHISPER_MODEL
    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError:
        return False

    if _LOCAL_WHISPER_MODEL is not None:
        return True

    with _LOCAL_WHISPER_LOCK:
        if _LOCAL_WHISPER_MODEL is None:
            _LOCAL_WHISPER_MODEL = _load_whisper_model(model_size)
    return True


def _transcribe_faster_whisper(wav_path, model_size):
    """Local Whisper via faster-whisper (CTranslate2). Returns None if the
    package is not installed."""
    global _LOCAL_WHISPER_MODEL
    if not warm_up_local_whisper(model_size):
        return None

    with _LOCAL_WHISPER_LOCK:
        model = _LOCAL_WHISPER_MODEL
    if model is None:
        return None

    try:
        segments_iter, info = model.transcribe(
            wav_path, beam_size=1, vad_filter=True)
        segments = [{"start": round(s.start, 2), "end": round(s.end, 2),
                     "text": s.text.strip()} for s in segments_iter]
    except Exception as exc:
        # If CUDA or memory failed during inference, fallback to CPU model
        with _LOCAL_WHISPER_LOCK:
            from faster_whisper import WhisperModel
            _LOCAL_WHISPER_MODEL = WhisperModel(
                model_size, device="cpu", compute_type="int8", cpu_threads=4)
            model = _LOCAL_WHISPER_MODEL
        segments_iter, info = model.transcribe(
            wav_path, beam_size=1, vad_filter=True)
        segments = [{"start": round(s.start, 2), "end": round(s.end, 2),
                     "text": s.text.strip()} for s in segments_iter]

    return {
        "transcript": " ".join(s["text"] for s in segments).strip(),
        "language": info.language or "",
        "segments": segments,
        "backend": "faster-whisper:%s" % model_size,
    }


def _transcribe_api(wav_path, llm_client, api_model):
    """Fallback: OpenAI-compatible Whisper endpoint."""
    if llm_client is None:
        return None
    try:
        with open(wav_path, "rb") as audio_file:
            response = llm_client.audio.transcriptions.create(
                model=api_model, file=audio_file,
                response_format="verbose_json")
    except Exception:
        # Some compatible servers ignore verbose_json; retry plain.
        try:
            with open(wav_path, "rb") as audio_file:
                response = llm_client.audio.transcriptions.create(
                    model=api_model, file=audio_file)
        except Exception as exc:
            raise SpeechRecognitionError(
                "Speech-to-text is temporarily unavailable. Please try "
                "again in a moment.", detail=str(exc)[:300])

    segments = []
    for seg in (getattr(response, "segments", None) or []):
        if isinstance(seg, dict):
            segments.append({"start": seg.get("start"), "end": seg.get("end"),
                             "text": (seg.get("text") or "").strip()})
        else:
            segments.append({"start": getattr(seg, "start", None),
                             "end": getattr(seg, "end", None),
                             "text": (getattr(seg, "text", "") or "").strip()})
    transcript = (getattr(response, "text", "") or "").strip()
    if not transcript and segments:
        transcript = " ".join(s["text"] for s in segments).strip()
    return {
        "transcript": transcript,
        "language": getattr(response, "language", "") or "",
        "segments": segments,
        "backend": "api:%s" % api_model,
    }


def transcribe(video_path, llm_client=None, model_size="base",
               api_model="whisper-1"):
    """Recorded Video -> Extract Audio -> Whisper -> transcript payload.

    Returns ``{"transcript", "language", "segments", "audio_quality",
    "backend"}``. Raises SpeechRecognitionError with a user-friendly message
    when no speech is detected, audio quality is too poor, or all backends
    fail.
    """
    wav_path = extract_audio(video_path)
    try:
        quality = analyze_audio_quality(wav_path)
        try:
            guard_audio_quality(quality)
        except SpeechRecognitionError as exc:
            # Attach the quality payload so callers (e.g. the live preview)
            # can show mic-level feedback even when the recording is silent
            # or too quiet - not just a generic "no speech" message.
            exc.audio_quality = quality
            raise

        # One retry: transient allocator/network hiccups (e.g. mkl_malloc
        # failures under memory pressure) frequently succeed on a second try
        # once the previous attempt's memory has been released.
        result = None
        for attempt in range(2):
            try:
                result = (_transcribe_faster_whisper(wav_path, model_size)
                          or _transcribe_api(wav_path, llm_client, api_model))
                break
            except SpeechRecognitionError:
                raise
            except Exception as exc:
                if attempt == 0:
                    import gc
                    import time
                    gc.collect()
                    time.sleep(1.0)
                    continue
                raise SpeechRecognitionError(
                    "Something went wrong while transcribing your "
                    "explanation. Please try again.",
                    detail=str(exc)[:300]) from exc
        if result is None:
            raise SpeechRecognitionError(
                "Speech-to-text is unavailable right now. Please try again "
                "later.", detail="no Whisper backend available")

        if not result.get("transcript"):
            if quality.get("quality_label") == "silent":
                raise SpeechRecognitionError(
                    "No speech was detected. Microphone captured silence (input level is near 0). "
                    "Please ensure your microphone is unmuted in Windows settings and the correct device is selected.")
            raise SpeechRecognitionError(
                "No speech was detected in your recording. Please record "
                "again and explain the concept out loud.")

        result["audio_quality"] = quality
        return result
    except SpeechRecognitionError:
        raise
    except Exception as exc:                 # never leak raw errors upward
        raise SpeechRecognitionError(
            "Something went wrong while transcribing your explanation. "
            "Please try again.", detail=str(exc)[:300])
    finally:
        _safe_remove_wav(wav_path)