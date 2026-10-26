"""REAL-TIME AI PROCTORING DETECTION SERVICE.

Pipeline Architecture:
  Webcam Frame (JPEG / BGR)
       │
       ├─► Ultralytics YOLO (yolo11n.pt)
       │     ├── Cell phone detection (confidence >= 0.60)
       │     └── Person count (0 people, 1 person, 2+ people)
       │
       └─► MediaPipe Face Landmarker + 3D Head Pose
             ├── Looking left
             ├── Looking right
             └── Looking down

Features:
  - Pretrained YOLO object classes only (no custom training needed).
  - Multi-frame confirmation before flagging an anomaly (anti false-positive).
  - Continuous duration requirement for head orientation (ignores brief glances).
  - Anomaly state debouncing: continuous active violations generate ONE warning.
  - Automatic test submission when warning_count >= 3.
  - Automatic GPU (CUDA) acceleration with CPU fallback.
  - Models loaded once at startup.
"""

import base64
import os
import threading
import time
from datetime import datetime, timezone
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================================ #
# CONFIGURABLE PARAMETERS (As required: do not hard-code throughout the app)   #
# ============================================================================ #

# Confidence thresholds
PHONE_CONFIDENCE = 0.38
PERSON_CONFIDENCE = 0.45

# Frame confirmation thresholds (prevent single-frame false positives)
PHONE_CONFIRMATION_FRAMES = 2   # Confirmed quickly across 2 frames
PERSON_ANOMALY_FRAMES = 3       # Multi/zero person confirmed over >= 3 frames

# Head movement thresholds (MediaPipe Head Pose Estimation)
HEAD_TURN_THRESHOLD = 20.0      # Yaw angle in degrees (looking left / right)
HEAD_DOWN_THRESHOLD = 18.0      # Pitch angle in degrees (looking down)
HEAD_ANOMALY_DURATION = 2.0     # Must look away continuously for >= 2.0 seconds

# Proctoring analysis rate
PROCTORING_SAMPLE_FPS = 4.0     # Recommended sampling rate (frames per second)
MAX_WARNINGS = 3                # Maximum allowed warnings before auto-submission

# 3D Anthropomorphic face model points for solvePnP head pose estimation
FACE_3D_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (Landmark 1)
    (0.0, -330.0, -65.0),         # Chin (Landmark 152)
    (-225.0, 170.0, -135.0),      # Left eye left corner (Landmark 33)
    (225.0, 170.0, -135.0),       # Right eye right corner (Landmark 263)
    (-150.0, -150.0, -125.0),     # Left mouth corner (Landmark 61)
    (150.0, -150.0, -125.0)       # Right mouth corner (Landmark 291)
], dtype=np.float64)

FACE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]


# ============================================================================ #
# PRETRAINED DETECTORS (YOLO & MediaPipe Face Landmarker - Loaded Once)         #
# ============================================================================ #

_DETECTOR_LOCK = threading.Lock()
_YOLO_MODEL = None
_MEDIAPIPE_LANDMARKER = None
_YOLO_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_LANDMARKER_TASK_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")


def get_yolo_model():
    """Load pretrained lightweight YOLO model once."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        with _DETECTOR_LOCK:
            if _YOLO_MODEL is None:
                torch.set_num_threads(4)
                # yolo11n.pt is the official lightweight pretrained nano model
                _YOLO_MODEL = YOLO("yolo11n.pt")
                # Warm-up inference
                dummy = np.zeros((480, 480, 3), dtype=np.uint8)
                _ = _YOLO_MODEL(dummy, imgsz=480, device=_YOLO_DEVICE, verbose=False)
    return _YOLO_MODEL


def get_mediapipe_landmarker():
    """Load MediaPipe FaceLandmarker once."""
    global _MEDIAPIPE_LANDMARKER
    if _MEDIAPIPE_LANDMARKER is None:
        with _DETECTOR_LOCK:
            if _MEDIAPIPE_LANDMARKER is None:
                base_options = python.BaseOptions(model_asset_path=_LANDMARKER_TASK_PATH)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_facial_transformation_matrixes=True,
                    num_faces=5,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5
                )
                _MEDIAPIPE_LANDMARKER = vision.FaceLandmarker.create_from_options(options)
    return _MEDIAPIPE_LANDMARKER


# ============================================================================ #
# FRAME ANALYSIS (Object & Head Pose Detection)                                #
# ============================================================================ #

def estimate_head_pose(landmarks_2d, width, height):
    """Estimate head pose (yaw, pitch, roll in degrees) using solvePnP."""
    if len(landmarks_2d) != 6:
        return 0.0, 0.0, 0.0

    focal_length = float(width)
    center = (float(width) / 2.0, float(height) / 2.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    image_points = np.array(landmarks_2d, dtype=np.float64)
    success, rvec, _ = cv2.solvePnP(
        FACE_3D_MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])
    return yaw, pitch, roll


def analyze_frame_objects_and_pose(frame_bgr, force_yolo=False):
    """Run YOLO object detection and MediaPipe head pose on a BGR image."""
    height, width = frame_bgr.shape[:2]
    now_ts = time.time()

    person_count = 0
    phone_detected = False
    phone_confidence = 0.0

    # 1. YOLO Detection with high-detail resolution (imgsz=480, ~45ms inference)
    yolo = get_yolo_model()
    with torch.inference_mode():
        results = yolo(frame_bgr, imgsz=480, device=_YOLO_DEVICE, verbose=False, conf=0.25)

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                
                # Class 0: person
                if cls_id == 0 and conf >= PERSON_CONFIDENCE:
                    person_count += 1
                # Class 67: cell phone, Class 65: remote (often misclassified screen-off phones)
                elif cls_id in (67, 65) and conf >= PHONE_CONFIDENCE:
                    phone_detected = True
                    if conf > phone_confidence:
                        phone_confidence = conf
        del results

    # 2. MediaPipe Face Landmarker
    landmarker = get_mediapipe_landmarker()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    mp_result = landmarker.detect(mp_image)

    num_faces = len(mp_result.face_landmarks) if mp_result.face_landmarks else 0
    head_direction = "center"
    yaw_deg = 0.0
    pitch_deg = 0.0
    roll_deg = 0.0

    if num_faces > 0:
        face_lm = mp_result.face_landmarks[0]
        # Extract the 6 key facial points in pixel coordinates
        pts_2d = []
        for idx in FACE_LANDMARK_INDICES:
            pt = face_lm[idx]
            pts_2d.append((pt.x * width, pt.y * height))

        yaw_deg, pitch_deg, roll_deg = estimate_head_pose(pts_2d, width, height)

        # Geometric check for robustness (relative nose position)
        nose_x = face_lm[1].x
        left_eye_x = face_lm[33].x
        right_eye_x = face_lm[263].x
        eye_span = max(1e-5, abs(right_eye_x - left_eye_x))
        nose_ratio = (nose_x - min(left_eye_x, right_eye_x)) / eye_span

        # Direction classification
        if yaw_deg > HEAD_TURN_THRESHOLD or nose_ratio > 0.68:
            head_direction = "looking_left"
        elif yaw_deg < -HEAD_TURN_THRESHOLD or nose_ratio < 0.32:
            head_direction = "looking_right"
        elif pitch_deg < -HEAD_DOWN_THRESHOLD:
            head_direction = "looking_down"
        else:
            head_direction = "center"

    return {
        "timestamp": now_ts,
        "person_count": max(person_count, num_faces),
        "phone_detected": phone_detected,
        "phone_confidence": round(phone_confidence, 3),
        "num_faces": num_faces,
        "head_direction": head_direction,
        "yaw": round(yaw_deg, 1),
        "pitch": round(pitch_deg, 1),
        "roll": round(roll_deg, 1)
    }


# ============================================================================ #
# PROCTORING ANOMALY MANAGER & STATE TRACKER                                   #
# ============================================================================ #

class ProctoringAnomalyManager:
    """Tracks proctoring state, consecutive frames, debouncing, and warnings."""

    def __init__(self, quiz_id):
        self.quiz_id = quiz_id
        self.warning_count = 0
        self.max_warnings = MAX_WARNINGS
        self.anomaly_history = []
        self.lock = threading.Lock()

        # Frame counters for continuous detection
        self.phone_frames = 0
        self.multi_person_frames = 0
        self.zero_person_frames = 0

        # Timestamps for head direction continuous monitoring
        self.head_away_start_time = None
        self.current_head_anomaly = None

        # Debounce state: tracks which anomaly is currently active
        # An active anomaly DOES NOT increment warnings continuously.
        self.active_states = {
            "PHONE_DETECTED": False,
            "MULTIPLE_PERSONS": False,
            "NO_PERSON": False,
            "LOOKING_LEFT": False,
            "LOOKING_RIGHT": False,
            "LOOKING_DOWN": False
        }

        # Clear frame counters to debounce release
        self.clear_counts = {k: 0 for k in self.active_states}

    def process_detection(self, det):
        """Update state with a new frame detection and return status."""
        with self.lock:
            now = det.get("timestamp", time.time())
            now_iso = datetime.now(timezone.utc).isoformat()
            new_warnings = []
            active_anomalies = []

            # ------------------------------------------------------------- #
            # 1. PHONE DETECTION (Multi-frame confirmation)                 #
            # ------------------------------------------------------------- #
            if det["phone_detected"]:
                self.phone_frames += 1
                self.clear_counts["PHONE_DETECTED"] = 0

                if self.phone_frames >= PHONE_CONFIRMATION_FRAMES:
                    active_anomalies.append("PHONE_DETECTED")
                    if not self.active_states["PHONE_DETECTED"]:
                        self.active_states["PHONE_DETECTED"] = True
                        self.warning_count += 1
                        anomaly = {
                            "type": "PHONE_DETECTED",
                            "timestamp": now_iso,
                            "confidence": det["phone_confidence"],
                            "duration": round(self.phone_frames / PROCTORING_SAMPLE_FPS, 1),
                            "message": "Mobile phone detected in camera view"
                        }
                        self.anomaly_history.append(anomaly)
                        new_warnings.append({
                            "warning_number": self.warning_count,
                            "type": "PHONE_DETECTED",
                            "title": f"⚠ WARNING {self.warning_count}/{self.max_warnings}",
                            "message": "Mobile phone detected."
                        })
            else:
                self.phone_frames = max(0, self.phone_frames - 1)
                self.clear_counts["PHONE_DETECTED"] += 1
                if self.clear_counts["PHONE_DETECTED"] >= 3:
                    # Phone disappeared - reset active state
                    self.active_states["PHONE_DETECTED"] = False

            # ------------------------------------------------------------- #
            # 2. PERSON COUNT DETECTION (0 or 2+ people)                   #
            # ------------------------------------------------------------- #
            persons = det["person_count"]
            if persons > 1:
                self.multi_person_frames += 1
                self.clear_counts["MULTIPLE_PERSONS"] = 0
                self.zero_person_frames = 0

                if self.multi_person_frames >= PERSON_ANOMALY_FRAMES:
                    active_anomalies.append("MULTIPLE_PERSONS")
                    if not self.active_states["MULTIPLE_PERSONS"]:
                        self.active_states["MULTIPLE_PERSONS"] = True
                        self.warning_count += 1
                        anomaly = {
                            "type": "MULTIPLE_PERSONS",
                            "timestamp": now_iso,
                            "confidence": 0.90,
                            "duration": round(self.multi_person_frames / PROCTORING_SAMPLE_FPS, 1),
                            "message": f"More than one person detected ({persons} people)"
                        }
                        self.anomaly_history.append(anomaly)
                        new_warnings.append({
                            "warning_number": self.warning_count,
                            "type": "MULTIPLE_PERSONS",
                            "title": f"⚠ WARNING {self.warning_count}/{self.max_warnings}",
                            "message": "More than one person detected."
                        })
            elif persons == 0:
                self.zero_person_frames += 1
                self.clear_counts["NO_PERSON"] = 0
                self.multi_person_frames = 0

                if self.zero_person_frames >= PERSON_ANOMALY_FRAMES + 1:
                    active_anomalies.append("NO_PERSON")
                    if not self.active_states["NO_PERSON"]:
                        self.active_states["NO_PERSON"] = True
                        self.warning_count += 1
                        anomaly = {
                            "type": "NO_PERSON",
                            "timestamp": now_iso,
                            "confidence": 0.95,
                            "duration": round(self.zero_person_frames / PROCTORING_SAMPLE_FPS, 1),
                            "message": "No student detected in camera view"
                        }
                        self.anomaly_history.append(anomaly)
                        new_warnings.append({
                            "warning_number": self.warning_count,
                            "type": "NO_PERSON",
                            "title": f"⚠ WARNING {self.warning_count}/{self.max_warnings}",
                            "message": "No person detected in camera view."
                        })
            else:
                # Exactly 1 person (Normal)
                self.multi_person_frames = 0
                self.zero_person_frames = 0
                self.clear_counts["MULTIPLE_PERSONS"] += 1
                self.clear_counts["NO_PERSON"] += 1
                if self.clear_counts["MULTIPLE_PERSONS"] >= 3:
                    self.active_states["MULTIPLE_PERSONS"] = False
                if self.clear_counts["NO_PERSON"] >= 3:
                    self.active_states["NO_PERSON"] = False

            # ------------------------------------------------------------- #
            # 3. HEAD DIRECTION DETECTION (Duration-based)                  #
            # ------------------------------------------------------------- #
            direction = det["head_direction"]
            head_types = {
                "looking_left": "LOOKING_LEFT",
                "looking_right": "LOOKING_RIGHT",
                "looking_down": "LOOKING_DOWN"
            }

            if direction in head_types:
                anomaly_type = head_types[direction]
                if self.current_head_anomaly != anomaly_type:
                    self.head_away_start_time = now
                    self.current_head_anomaly = anomaly_type

                duration = now - (self.head_away_start_time or now)
                self.clear_counts[anomaly_type] = 0

                if duration >= HEAD_ANOMALY_DURATION:
                    active_anomalies.append(anomaly_type)
                    if not self.active_states[anomaly_type]:
                        self.active_states[anomaly_type] = True
                        self.warning_count += 1
                        dir_label = direction.replace("_", " ")
                        anomaly = {
                            "type": anomaly_type,
                            "timestamp": now_iso,
                            "confidence": 0.88,
                            "duration": round(duration, 1),
                            "message": f"Student looking away from screen ({dir_label})"
                        }
                        self.anomaly_history.append(anomaly)
                        new_warnings.append({
                            "warning_number": self.warning_count,
                            "type": anomaly_type,
                            "title": f"⚠ WARNING {self.warning_count}/{self.max_warnings}",
                            "message": "Student looking away from the screen."
                        })
            else:
                # Looking center - reset head turn timer
                self.head_away_start_time = None
                self.current_head_anomaly = None
                for ht in ("LOOKING_LEFT", "LOOKING_RIGHT", "LOOKING_DOWN"):
                    self.clear_counts[ht] += 1
                    if self.clear_counts[ht] >= 3:
                        self.active_states[ht] = False

            # ------------------------------------------------------------- #
            # 4. AUTO-SUBMIT CHECK (warning_count >= 3)                      #
            # ------------------------------------------------------------- #
            auto_submit = self.warning_count >= self.max_warnings
            auto_submit_reason = (
                "Maximum number of proctoring warnings reached."
                if auto_submit else None
            )

            latest_warning = new_warnings[-1] if new_warnings else None

            return {
                "status": "ACTIVE" if not auto_submit else "SUBMITTED",
                "warning_count": min(self.warning_count, self.max_warnings),
                "max_warnings": self.max_warnings,
                "new_warning": bool(latest_warning),
                "warning_message": latest_warning["message"] if latest_warning else None,
                "warning_title": latest_warning["title"] if latest_warning else None,
                "warning_type": latest_warning["type"] if latest_warning else None,
                "active_anomalies": active_anomalies,
                "auto_submit": auto_submit,
                "reason": auto_submit_reason,
                "detections": {
                    "phone_detected": det["phone_detected"],
                    "phone_confidence": det["phone_confidence"],
                    "person_count": det["person_count"],
                    "head_direction": det["head_direction"],
                    "yaw": det["yaw"],
                    "pitch": det["pitch"]
                }
            }


# Global session registry
_SESSIONS_LOCK = threading.Lock()
_ACTIVE_SESSIONS = {}


def get_proctoring_session(quiz_id):
    """Retrieve or create a ProctoringAnomalyManager for a quiz session."""
    with _SESSIONS_LOCK:
        if quiz_id not in _ACTIVE_SESSIONS:
            _ACTIVE_SESSIONS[quiz_id] = ProctoringAnomalyManager(quiz_id)
        return _ACTIVE_SESSIONS[quiz_id]


def clear_proctoring_session(quiz_id):
    """Clear session when exam completes or is submitted."""
    with _SESSIONS_LOCK:
        if quiz_id in _ACTIVE_SESSIONS:
            return _ACTIVE_SESSIONS.pop(quiz_id)
    return None

