# AMS-LAS — Student Understanding Verification & Analysis Module

Determines how well a student understands a topic after the student explains it
in their own words through a recorded explanation. The result is an
**explainable verification report**, not merely a numerical score.

The student speaks (and may optionally show their face). **The student never
types their answer.** Their spoken explanation is transcribed by **Whisper**
and verified by the AI; the recorded video is processed by **OpenCV**.

## Student flow

```
Student sees topic
      ↓
Student opens camera + microphone            (browser getUserMedia)
      ↓
Student explains the topic verbally
      ↓
OpenCV captures/processes video              (browser MediaRecorder → backend)
      ↓
Whisper converts speech → transcript         (audio extracted via ffmpeg)
      ↓
AI verifies conceptual understanding
      ↓
Knowledge gaps + misconceptions identified
      ↓
AUVA calculates understanding
      ↓
Student receives explainable verification report
```

The browser/frontend owns camera & microphone **permission handling**; the
backend OpenCV service processes the received video frames.

## Pipeline

```
Student Camera → (browser records) → Uploaded Video
    → OpenCV secure storage
    → OpenCV frame-by-frame visual processing        → visual analysis (attention inputs)
    → Audio extraction (ffmpeg)
    → Whisper speech-to-text                         → {transcript, language, segments}
    → Embedding model (semantic similarity)
    → LLM (concept coverage, knowledge gaps, misconceptions)
    → MediaPipe/visual model (attention indicators)
    → Audio processing (speech characteristics)
    → Emotion model (facial-expression info)
    → AUVA                                           → final understanding score
    → EXPLAINABLE VERIFICATION REPORT
```

The final AUVA score is computed with the **existing AUVA algorithm** — the LLM
is never allowed to assign an arbitrary understanding percentage. Each model has
a single responsibility (see "Responsibility separation" below).

## Layout

| Path | Purpose |
|---|---|
| `verification/opencv_service.py` | OpenCV: secure recording storage + frame-by-frame visual analysis (face/stability/distraction metrics) — inputs to the attention component |
| `verification/whisper_service.py` | Whisper: ffmpeg audio extraction, audio-quality guards (no-speech / poor-audio), speech-to-text → `{transcript, language, segments}` (local faster-whisper → OpenAI-compatible API fallback) |
| `verification/models/face_detection_yunet_2023mar.onnx` | Bundled OpenCV YuNet face-detection model |
| `verification/config.py` + `verification_config.json` | Configurable AUVA weights, understanding-level thresholds (90/75/60/40), severity weights, confidence weights, Whisper backend selection |
| `verification/semantic.py` | Semantic similarity (Sentence-BERT → embedding API → lexical fallback), cosine normalized 0–100 |
| `verification/concepts.py` | Concept extraction, coverage (clearly / partially / not explained — name-dropping never counts as covered), knowledge gaps, LLM misconception detection (genuine conceptual errors only) |
| `verification/behavioral.py` | Attention (visual-analysis inputs), speech confidence (numpy acoustic analysis of WAV: speech activity, pauses, energy, pitch variation, stability), emotion normalization |
| `verification/auva.py` | AUVA formula, gap/misconception penalties, understanding levels, separate verification-confidence calculation |
| `verification/pipeline.py` | Orchestrator producing the final report |
| `app.py` → `POST /api/recording/upload`, `POST /api/verify-understanding`, `POST /api/verify-quiz-answer` | REST endpoints |

## API

### POST `/api/recording/upload`  (multipart `video`)

Receives the student's recorded explanation and runs the full capture →
analysis chain:

```
uploaded video
  → secure_save_video  (OpenCV service; random UUID name, format/size guarded)
  → process_video      (OpenCV frame-by-frame face/stability/distraction metrics)
  → extract_audio      (ffmpeg → 16 kHz mono WAV)
  → transcribe         (Whisper → {transcript, language, segments})
  → store_transcript   (persisted next to the recording so the AI reuses it)
```

Returns

```json
{
  "recording_id": "<basename>.mp4",
  "transcript": "...",
  "language": "en",
  "segments": [{ "start": 0.0, "end": 8.0, "text": "..." }],
  "audio_quality": { "duration": 14.95, "rms": 0.106, "quality_label": "good" },
  "video_analysis": { "face_presence_ratio": 0.9, "distraction_events": 1, ... }
}
```

### POST `/api/verify-understanding`

```json
POST /api/verify-understanding
{
  "subject": "Computer Science",
  "topic": "Object-Oriented Programming",
  "expected_representation": "Classes are blueprints... Inheritance reuses...",
  "transcript": "<Whisper transcript of the student's explanation>",   // optional if recording_id supplied
  "recording_id": "<id returned by /api/recording/upload>",            // optional; uses stored transcript + visual metrics
  "video_analysis": { "face_presence_ratio": 0.95, ... },              // optional
  "audio_analysis": { "speech_activity_ratio": 0.72, ... },           // optional
  "emotion_analysis": { "emotion_summary": {...} },                   // optional
  "audio": "<base64 WAV>"                                              // optional, live acoustic analysis
}
```

### POST `/api/verify-quiz-answer`

```json
{
  "topic": "Computer Science",
  "question": "Explain object-oriented programming in your own words.",
  "recording_id": "<id returned by /api/recording/upload>"
}
```

When `recording_id` is supplied, the **stored Whisper transcript** is used as
the primary textual input for verification and the **OpenCV visual metrics**
feed the (supplementary) attention analysis — no manual answer text is required.


## Design guarantees

- **Primary conceptual evidence** (semantic similarity, concept coverage,
  knowledge gaps, misconceptions) is strictly separated from **supplementary
  behavioral evidence** (attention, speech characteristics, emotion).
- Attention/speech/emotion NEVER conclude understanding or lack of it.
- Semantic similarity alone is never treated as proof of understanding.
- Verification **confidence ≠ understanding** (AUVA score vs. evidence reliability).
- The default AUVA weights (0.35 / 0.25 / 0.15 / 0.10 − 0.10 − 0.05) are an
  **initial proposed configuration** — NOT scientifically validated universal
  weights — and must be experimentally evaluated.

## Run

```
d:\tsa\.venv\Scripts\python.exe app.py            # Flask app on :5000
d:\tsa\.venv\Scripts\python.exe test_verification.py  # offline validation (22 checks)
```

## Quiz integration

The Quiz section is a single integrated **Quiz & Understanding Verification** flow:
students type each answer (no mic/camera), every submitted answer is automatically
run through the AUVA pipeline (`POST /api/verify-quiz-answer`) against an LLM-generated expected concept representation, and a per-answer explainable feedback
card plus an end-of-quiz aggregate understanding summary are shown inline.

## Responsibility separation

Each component is responsible for exactly one job:

| Component | Responsibility |
|---|---|
| **OpenCV** (`opencv_service.py`) | Camera/video capture and visual frame processing only |
| **Whisper** (`whisper_service.py`) | Speech-to-text only |
| **Embedding model** (`semantic.py`) | Semantic similarity only |
| **LLM** (`concepts.py`) | Concept coverage, knowledge gaps, misconception detection |
| **OpenCV / MediaPipe visual model** (`behavioral.analyze_attention`) | Attention-related visual indicators |
| **Audio processing** (`behavioral.analyze_speech_confidence`) | Speech characteristics |
| **Emotion model** (`behavioral.analyze_emotion`) | Supplementary facial-expression information |
| **AUVA** (`auva.py`) | Final mathematical understanding calculation |

No single model is responsible for every task.

## Error handling

All user-facing failures are translated into clear, friendly messages — raw
backend errors and stack traces are never exposed to students:

- Camera/microphone permission denied or device unavailable (handled in the
  browser via `getUserMedia`, surfaced as a friendly message)
- Unsupported / corrupted / empty video (OpenCV service)
- No speech detected or poor audio quality (Whisper service guards)
- Whisper backend unavailable or transcription failure
- LLM / API failure and analysis timeouts (mapped to retry-friendly messages)

