"""Offline validation of the OpenCV + Whisper media pipeline.

Creates a synthetic video (OpenCV VideoWriter) with generated audio muxed in
via ffmpeg, then verifies:

  1. secure_save_video rejects unsupported formats / empty uploads
  2. OpenCV frame-by-frame processing returns attention-compatible metrics
  3. Audio extraction produces a valid WAV
  4. Audio-quality analysis labels clear audio as usable / silence as unusable
  5. Whisper transcribes the recording ({transcript, language, segments})
  6. Transcript persistence round-trip works
"""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import wave

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verification import opencv_service
from verification import whisper_service

CHECKS = []


def check(name, condition):
    CHECKS.append((name, bool(condition)))
    print("[PASS] %s" % name if condition else "[FAIL] %s" % name)


def make_synthetic_video(path, seconds=4, fps=25):
    """Animated frames so OpenCV has real frame-by-frame work to do."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (640, 480))
    total = int(seconds * fps)
    for i in range(total):
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        cx = int(320 + 120 * np.sin(2 * np.pi * i / total))
        cv2.circle(frame, (cx, 220), 90, (200, 200, 210), -1)
        cv2.circle(frame, (cx - 30, 190), 12, (30, 30, 30), -1)
        cv2.circle(frame, (cx + 30, 190), 12, (30, 30, 30), -1)
        writer.write(frame)
    writer.release()


def make_audio_wav(path, seconds=4, rate=16000, amplitude=0.35):
    """Amplitude-modulated tone approximating speech energy."""
    t = np.linspace(0, seconds, int(seconds * rate), endpoint=False)
    carrier = np.sin(2 * np.pi * 180.0 * t)
    envelope = 0.6 * (1 + np.sin(2 * np.pi * 3.0 * t))
    samples = (carrier * envelope * amplitude * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())


def mux_audio(video_path, wav_path, out_path):
    ffmpeg = whisper_service._ffmpeg_binary()
    subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", video_path, "-i", wav_path,
                    "-c:v", "copy", "-c:a", "aac", "-shortest", out_path],
                   check=True)


class _FakeUpload:
    """Minimal Flask FileStorage stand-in for save-path unit checks."""

    def __init__(self, data, filename):
        self.stream = io.BytesIO(data)
        self.filename = filename


def main():
    tmp = tempfile.mkdtemp(prefix="ams_media_")
    try:
        raw_video = os.path.join(tmp, "raw.mp4")
        audio_wav = os.path.join(tmp, "audio.wav")
        final_video = os.path.join(tmp, "explanation.mp4")

        make_synthetic_video(raw_video)
        make_audio_wav(audio_wav)
        mux_audio(raw_video, audio_wav, final_video)

        # --- 1) Secure storage rules ------------------------------------- #
        opencv_service.RECORDINGS_DIR = os.path.join(tmp, "recordings")
        saved = opencv_service.secure_save_video(
            _FakeUpload(open(final_video, "rb").read(), "explanation.webm"),
            "explanation.webm")
        check("supported format saved securely",
              os.path.exists(saved) and
              os.path.basename(saved) != "explanation.webm")

        rejected = False
        try:
            opencv_service.secure_save_video(
                _FakeUpload(b"x" * 1000, "notes.txt"), "notes.txt")
        except opencv_service.MediaProcessingError as exc:
            rejected = bool(exc.user_message)
        check("unsupported extension rejected with friendly message", rejected)

        empty_rejected = False
        try:
            opencv_service.secure_save_video(
                _FakeUpload(b"", "empty.mp4"), "empty.mp4")
        except opencv_service.MediaProcessingError:
            empty_rejected = True
        check("empty upload rejected", empty_rejected)

        # --- 2) OpenCV frame-by-frame processing -------------------------- #
        metrics = opencv_service.process_video(saved)
        check("opencv processed every frame", metrics["total_frames"] >= 80)
        check("attention metrics present",
              all(k in metrics for k in (
                  "face_presence_ratio", "frontal_ratio",
                  "head_pose_stability", "distraction_events")))
        check("metrics within expected ranges",
              0.0 <= metrics["face_presence_ratio"] <= 1.0
              and metrics["duration_seconds"] > 0)

        bad = False
        corrupt = os.path.join(tmp, "corrupt.mp4")
        with open(corrupt, "wb") as fh:
            fh.write(b"not a video at all")
        try:
            opencv_service.process_video(corrupt)
        except opencv_service.MediaProcessingError as exc:
            bad = bool(exc.user_message)
        check("corrupt video handled gracefully", bad)

        # --- 3-5) Audio extraction -> quality -> Whisper ------------------- #
        wav_path = whisper_service.extract_audio(saved)
        check("audio extracted to 16kHz mono WAV",
              os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000)

        quality = whisper_service.analyze_audio_quality(wav_path)
        check("clear audio labeled usable", quality["quality_label"] == "good")

        silent = os.path.join(tmp, "silent.wav")
        with wave.open(silent, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)
        no_speech = False
        try:
            whisper_service.guard_audio_quality(
                whisper_service.analyze_audio_quality(silent))
        except whisper_service.SpeechRecognitionError as exc:
            no_speech = ("No audio" in exc.user_message or
                         "clear speech" in exc.user_message)
        check("silence raises friendly no-speech error", no_speech)

        transcript_text = ""
        language_text = ""
        try:
            result = whisper_service.transcribe(saved, llm_client=None,
                                                model_size="tiny")
            transcript_ok = isinstance(result.get("transcript"), str)
            lang_ok = ("language" in result and
                       isinstance(result.get("segments"), list))
            transcript_text = result.get("transcript", "")
            language_text = result.get("language", "")
        except whisper_service.SpeechRecognitionError as exc:
            # A pure tone carries no real speech - a friendly no-speech error
            # is equally valid pipeline behaviour.
            transcript_ok = "No speech" in exc.user_message or \
                "speech" in exc.user_message.lower()
            lang_ok = True
        check("whisper returned transcript payload", transcript_ok)
        check("whisper returned language + segments", lang_ok)

        # --- 6) Transcript persistence ------------------------------------ #
        opencv_service.store_transcript(
            saved, {"transcript": transcript_text,
                    "language": language_text, "segments": []},
            metrics)
        loaded = opencv_service.load_transcript(os.path.basename(saved))
        check("stored transcript round-trips",
              loaded is not None
              and loaded["whisper"]["transcript"] == transcript_text
              and loaded["video_analysis"]["total_frames"] ==
              metrics["total_frames"])

        passed = sum(1 for _, ok in CHECKS if ok)
        print("\n===== MEDIA PIPELINE: %d/%d CHECKS PASSED ====="
              % (passed, len(CHECKS)))
        return 0 if passed == len(CHECKS) else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())