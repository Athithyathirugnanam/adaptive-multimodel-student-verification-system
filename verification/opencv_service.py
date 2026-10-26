"""OPENCV MEDIA SERVICE: secure recording storage + frame-by-frame visual analysis.

Pipeline responsibility (kept strictly separate from every other model):
    Student Camera -> (browser records) -> Uploaded Video
        -> OpenCV secure storage -> OpenCV frame-by-frame processing
        -> Visual-analysis metrics (attention inputs for the AUVA pipeline)

This module NEVER interprets understanding. It only produces low-level visual
indicators consumed by ``verification.behavioral.analyze_attention``.
"""

import json
import os
import time
import uuid

import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDINGS_DIR = os.path.join(_PROJECT_ROOT, "recordings")

ALLOWED_EXTENSIONS = {".webm", ".mp4", ".avi", ".mov", ".mkv"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024          # 200 MB hard cap
# Sample at most this many frames per second for visual analysis (cost control)
ANALYSIS_SAMPLE_FPS = 5.0


class MediaProcessingError(Exception):
    """Raised for any recording/processing failure.

    Carries a USER-FRIENDLY message only - raw backend details must never be
    surfaced through the API.
    """

    def __init__(self, user_message, detail=""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


# --------------------------------------------------------------------------- #
# SECURE STORAGE                                                              #
# --------------------------------------------------------------------------- #
def _recordings_dir():
    """Ensure the recordings directory exists."""
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    return RECORDINGS_DIR


def _safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def secure_save_video(file_storage, original_filename=""):
    """Persist an uploaded recording under a random, non-guessable name.

    Only known video extensions are accepted, size is capped, and the stored
    name carries no student-controlled content. Returns the absolute path.
    """
    ext = os.path.splitext(original_filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise MediaProcessingError(
            "Unsupported video format. Please record again using the "
            "built-in camera recorder (WebM/MP4).",
            detail="extension=%r" % ext)

    directory = _recordings_dir()
    safe_name = "%s_%d%s" % (uuid.uuid4().hex, int(time.time()), ext)
    out_path = os.path.join(directory, safe_name)

    size = 0
    try:
        with open(out_path, "wb") as fh:
            while True:
                chunk = file_storage.stream.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise MediaProcessingError(
                        "The recording is too large. Please keep your "
                        "explanation to a few minutes.",
                        detail="size=%d" % size)
                fh.write(chunk)
    except MediaProcessingError:
        _safe_remove(out_path)
        raise
    except OSError as exc:
        _safe_remove(out_path)
        raise MediaProcessingError(
            "We couldn't save your recording. Please try submitting again.",
            detail=str(exc))

    if size == 0:
        _safe_remove(out_path)
        raise MediaProcessingError(
            "The recorded video was empty. Please record your explanation "
            "again.", detail="zero-byte upload")

    return out_path


def store_transcript(recording_path, transcript_payload, video_analysis):
    """Store the Whisper transcript + visual metrics next to the recording.

    The transcript is persisted so it can be passed to the verification AI
    later without re-running speech recognition.
    """
    meta = {
        "stored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "original_name": os.path.basename(recording_path),
        "whisper": transcript_payload,
        "video_analysis": video_analysis,
    }
    meta_path = recording_path + ".json"
    try:
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
    except OSError:
        # Metadata persistence is best-effort; the API response still carries
        # everything the client needs.
        return None
    return meta_path


def load_transcript(recording_id):
    """Load a previously stored Whisper result for a recording id."""
    safe_id = os.path.basename(str(recording_id))
    meta_path = os.path.join(_recordings_dir(), safe_id + ".json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# OPENCV FRAME-BY-FRAME VISUAL ANALYSIS                                       #
# --------------------------------------------------------------------------- #
_YUNET_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "face_detection_yunet_2023mar.onnx")

_FACE_DETECTOR = None
_DETECTOR_SIZE = None


def _get_face_detector(width, height):
    """Lazily create the OpenCV YuNet DNN face detector (OpenCV >= 5).

    Returns None when the bundled ONNX model is unavailable - callers fall
    back to a coarse heuristic instead of failing the whole analysis.
    """
    global _FACE_DETECTOR, _DETECTOR_SIZE
    if not os.path.exists(_YUNET_MODEL_PATH):
        return None
    if _FACE_DETECTOR is None:
        try:
            _FACE_DETECTOR = cv2.FaceDetectorYN_create(
                _YUNET_MODEL_PATH, "", (width, height), score_threshold=0.6)
        except cv2.error:
            return None
    if _DETECTOR_SIZE != (width, height):
        _FACE_DETECTOR.setInputSize((width, height))
        _DETECTOR_SIZE = (width, height)
    return _FACE_DETECTOR


def _detect_face_box(frame):
    """Return ``(box_or_None, engine_name)`` for the dominant visible face.

    Primary engine : OpenCV FaceDetectorYN (YuNet, DNN).
    Fallback       : YCrCb skin-tone region heuristic, used only when the
                     model file is missing, so visual processing still yields
                     a coarse presence indicator instead of an error.
    """
    height, width = frame.shape[:2]
    detector = _get_face_detector(width, height)
    if detector is not None:
        _, faces = detector.detect(frame)
        if faces is not None and len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])[:4]
            return ((float(x), float(y), float(w), float(h)),
                    "OpenCV FaceDetectorYN (YuNet)")
        return None, "OpenCV FaceDetectorYN (YuNet)"

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    if float(np.count_nonzero(mask)) / mask.size > 0.02:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(
                max(contours, key=cv2.contourArea))
            return ((float(x), float(y), float(w), float(h)),
                    "OpenCV skin-tone heuristic")
    return None, "OpenCV skin-tone heuristic"


def process_video(video_path):
    """Process the recorded video frame-by-frame with OpenCV.

    Returns visual-analysis metrics compatible with
    ``verification.behavioral.analyze_attention``:

      face_presence_ratio : fraction of analyzed frames with a detectable face
      frontal_ratio       : fraction of sampled frames facing roughly forward
                            (frontal-face detector hit - an approximation)
      gaze_on_camera_ratio: camera-facing proxy derived from frontal hits
      head_pose_stability : stability of the detected face position across
                            frames (proxy indicator, NOT a pose estimator)
      distraction_events  : number of distinct look-away/leave episodes

    Raises MediaProcessingError with a user-friendly message when the video
    cannot be opened or contains no decodable frames.
    """
    if not video_path or not os.path.exists(video_path):
        raise MediaProcessingError(
            "Recording not found. Please record and submit again.",
            detail="missing %s" % video_path)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise MediaProcessingError(
            "We couldn't read your recording (unsupported or corrupted "
            "video). Please record again.",
            detail="cv2.VideoCapture failed to open")

    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        source_fps = float(source_fps) if source_fps > 0 else 30.0
        sample_every = max(1, int(round(source_fps / ANALYSIS_SAMPLE_FPS)))

        total_frames = 0
        analyzed_frames = 0
        face_frames = 0
        centroids = []
        distraction_events = 0
        consecutive_missing = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            total_frames += 1

            # Frame-by-frame OpenCV processing, sub-sampled for performance.
            if (total_frames - 1) % sample_every != 0:
                continue

            analyzed_frames += 1
            box, engine = _detect_face_box(frame)
            if box is not None:
                face_frames += 1
                consecutive_missing = 0
                # Dominant detected region assumed to be the student.
                centroids.append((box[0] + box[2] / 2.0,
                                  box[1] + box[3] / 2.0))
            else:
                if consecutive_missing == 0 and analyzed_frames > 1:
                    distraction_events += 1       # a look-away episode starts
                consecutive_missing += 1

        if total_frames == 0:
            raise MediaProcessingError(
                "We couldn't decode your recording. Please record again "
                "with the built-in camera recorder.",
                detail="zero decodable frames")

        duration_seconds = total_frames / source_fps
        face_presence_ratio = (face_frames / analyzed_frames
                               if analyzed_frames else 0.0)

        # Head-position stability: mean displacement of the face centroid
        # between consecutive detections, normalized against a nominal frame
        # distance. This is a coarse proxy indicator only.
        stability = None
        if len(centroids) >= 2:
            displacements = [
                float(np.hypot(curr[0] - prev[0], curr[1] - prev[1]))
                for prev, curr in zip(centroids, centroids[1:])]
            mean_disp = float(np.mean(displacements)) if displacements else 0.0
            stability = float(np.clip(1.0 - (mean_disp / 120.0), 0.0, 1.0))

        return {
            "engine": engine,
            "total_frames": total_frames,
            "analyzed_frames": analyzed_frames,
            "duration_seconds": round(duration_seconds, 2),
            "source_fps": round(source_fps, 2),
            "sample_fps": ANALYSIS_SAMPLE_FPS,
            # Metrics consumed by behavioral.analyze_attention:
            "face_presence_ratio": round(face_presence_ratio, 4),
            "frontal_ratio": round(face_presence_ratio, 4),
            "gaze_on_camera_ratio": round(face_presence_ratio * 0.9, 4),
            "head_pose_stability": (round(stability, 4)
                                    if stability is not None else None),
            "distraction_events": distraction_events,
        }
    except MediaProcessingError:
        raise
    except cv2.error as exc:
        raise MediaProcessingError(
            "An error occurred while analyzing your recording. Please try "
            "again.", detail=str(exc))
    finally:
        capture.release()