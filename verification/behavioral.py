"""SUPPLEMENTARY BEHAVIORAL EVIDENCE: attention, speech confidence, emotion.

Nothing in this module may be used to conclude that a student understands or
does not understand the concept. Attention is derived from visual-analysis
inputs, speech confidence from real acoustic analysis of the recorded audio
(speech activity, pauses, energy, pitch variation, stability), and emotion is
only normalized/passed through when facial-expression analysis is available.
When an input modality is missing, the component reports `available: False`
and the AUVA pipeline substitutes a neutral score with an explicit limitation.
"""

import io
import wave

import numpy as np


def _clamp(value, low=0.0, high=100.0):
    return max(low, min(high, float(value)))


def _get(value, default=None):
    return default if value is None else value


# --------------------------------------------------------------------------- #
# 5. ATTENTION ANALYSIS
# --------------------------------------------------------------------------- #
def analyze_attention(video_analysis, neutral=50):
    """Compute attention evidence from visual-analysis results.

    Expected optional fields inside `video_analysis`:
      face_presence_ratio   : 0-1 fraction of frames with a detectable face
      frontal_ratio         : 0-1 fraction of frames facing roughly forward
      gaze_on_camera_ratio  : 0-1 fraction where gaze is toward camera/screen
      head_pose_stability   : 0-1 stability indicator
      distraction_events    : int count of look-aways / exits
    """
    if not video_analysis or not isinstance(video_analysis, dict):
        return {
            "score": neutral,
            "indicators": ["No video/visual analysis was provided; attention could "
                           "not be measured. A neutral value was used."],
            "available": False,
        }

    indicators = []
    face = _get(video_analysis.get("face_presence_ratio"))
    frontal = _get(video_analysis.get("frontal_ratio"),
                   video_analysis.get("head_orientation_frontal_ratio"))
    gaze = _get(video_analysis.get("gaze_on_camera_ratio"),
                video_analysis.get("gaze_related_indicator"))
    stability = _get(video_analysis.get("head_pose_stability"))
    distractions = _get(video_analysis.get("distraction_events"), 0)

    parts, weights = [], []
    if face is not None:
        parts.append(_clamp(face, 0, 1) * 100)
        weights.append(0.40)
        indicators.append("Face present in %.0f%% of analyzed frames."
                          % (_clamp(face, 0, 1) * 100))
    if frontal is not None:
        parts.append(_clamp(frontal, 0, 1) * 100)
        weights.append(0.25)
        indicators.append("Head oriented toward the camera/screen in %.0f%% of frames."
                          % (_clamp(frontal, 0, 1) * 100))
    if gaze is not None:
        parts.append(_clamp(gaze, 0, 1) * 100)
        weights.append(0.25)
        indicators.append("Gaze-related indicator suggests focus in %.0f%% of samples."
                          % (_clamp(gaze, 0, 1) * 100))
    if stability is not None:
        parts.append(_clamp(stability, 0, 1) * 100)
        weights.append(0.10)
        indicators.append("Head-pose stability: %.0f%%." % (_clamp(stability, 0, 1) * 100))

    if not parts:
        return {
            "score": neutral,
            "indicators": ["Visual-analysis data was provided but contained no usable "
                           "attention indicators. A neutral value was used."],
            "available": False,
        }

    score = sum(p * w for p, w in zip(parts, weights)) / sum(weights)
    if distractions:
        penalty = min(15.0, float(distractions) * 3.0)
        score -= penalty
        indicators.append("%d potential distraction event(s) detected (-%.0f points)."
                          % (int(distractions), penalty))

    indicators.append(
        "NOTE: attention is supplementary behavioral evidence only - it never "
        "concludes that the student understood (or did not understand) the topic.")
    return {
        "score": round(_clamp(score), 1),
        "indicators": indicators,
        "available": True,
    }

# --------------------------------------------------------------------------- #
# 6. SPEECH CONFIDENCE ANALYSIS - acoustic feature extraction
# --------------------------------------------------------------------------- #
def _decode_audio(audio_bytes):
    """Decode WAV bytes to a float mono signal.
    Returns (signal, sample_rate) or (None, error_reason)."""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        if sampwidth == 2:
            sig = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            sig = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sampwidth == 1:
            sig = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            return None, "unsupported sample width %d" % sampwidth
        if n_channels > 1:
            trim = len(sig) // n_channels * n_channels
            sig = sig[:trim].reshape(-1, n_channels).mean(axis=1)
        return sig.astype(np.float64), int(rate)
    except Exception as exc:
        return None, str(exc)


def _frame_signal(sig, rate, frame_len=0.025, hop=0.010):
    fl = max(1, int(frame_len * rate))
    hp = max(1, int(hop * rate))
    frames = [sig[s:s + fl] for s in range(0, max(1, len(sig) - fl + 1), hp)]
    return np.array(frames), hp


def _estimate_f0(frame, rate):
    """Estimate the fundamental frequency of one voiced frame via autocorrelation."""
    frame = frame - frame.mean()
    if np.dot(frame, frame) <= 1e-9 or np.max(np.abs(frame)) < 1e-4:
        return None
    corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    corr = corr / (corr[0] + 1e-12)
    lo, hi = int(rate / 400), min(int(rate / 70), len(corr) - 1)  # 70-400 Hz
    if hi <= lo:
        return None
    peak_idx = lo + int(np.argmax(corr[lo:hi]))
    if corr[peak_idx] < 0.3:  # unvoiced frame
        return None
    return float(rate / peak_idx)


def analyze_speech_confidence(audio_bytes=None, audio_analysis=None, neutral=50):
    """Analyze acoustic characteristics of the student's explanation audio.

    Accepts raw WAV `audio_bytes` (analyzed here with numpy: speech activity,
    pauses, energy, pitch variation, stability) and/or a pre-computed
    `audio_analysis` dict. Speech confidence must NOT be treated as a direct
    measurement of knowledge.
    """
    metrics = {}
    available = False

    # Merge any pre-computed metrics first.
    if isinstance(audio_analysis, dict):
        for src_key in ("speech_activity_ratio", "pause_count",
                        "average_pause_seconds", "mean_energy",
                        "pitch_variation_hz", "duration_seconds"):
            val = audio_analysis.get(src_key)
            if val is not None:
                try:
                    key = {"pitch_variation_hz": "pitch_std_hz",
                           "duration_seconds": "duration"}.get(src_key, src_key)
                    metrics[key] = float(val)
                    available = True
                except (TypeError, ValueError):
                    pass

    # Real acoustic analysis on raw audio.
    if audio_bytes:
        sig, err = _decode_audio(audio_bytes)
        if sig is not None and len(sig) > 0:
            available = True
            frames, hop = _frame_signal(sig, err)
            fps = 1000.0 / hop  # frames per second
            rms = np.sqrt((frames ** 2).mean(axis=1))
            noise_floor = float(np.percentile(rms, 10))
            active = rms > max(0.01, noise_floor * 3.0)
            metrics["speech_activity_ratio"] = round(float(active.mean()), 3)

            # Pauses: inactive runs >= 300 ms located between active regions.
            pauses, run = [], 0
            for is_active in active:
                if not is_active:
                    run += 1
                else:
                    if run >= int(0.3 * fps):
                        pauses.append(run / fps)
                    run = 0
            if run >= int(0.3 * fps):
                pauses.append(run / fps)
            interior = pauses[:-1] if len(pauses) > 1 else []
            metrics["pause_count"] = len(interior)
            metrics["average_pause_seconds"] = (round(float(np.mean(interior)), 3)
                                                if interior else 0.0)

            active_rms = rms[active]
            if len(active_rms):
                med = float(np.median(active_rms))
                adequacy = _clamp(50.0 + 33.0 * np.log10(max(med, 1e-6) / 0.05))
                metrics["energy_adequacy"] = round(adequacy, 1)

            f0s = [f for f in (_estimate_f0(fr, err) for fr in frames[::5]) if f]
            if len(f0s) >= 5:
                f0_arr = np.asarray(f0s)
                cv = float(f0_arr.std() / (f0_arr.mean() + 1e-9))
                # Moderate variation (~0.15-0.35 CV) reads as engaged, stable speech.
                metrics["pitch_mean_hz"] = round(float(f0_arr.mean()), 1)
                metrics["pitch_std_hz"] = round(float(f0_arr.std()), 1)
                metrics["pitch_variation_score"] = round(
                    _clamp(100.0 - abs(cv - 0.22) * 250.0), 1)
                if len(f0s) >= 10:
                    jitter = float(np.abs(np.diff(f0_arr)).mean()
                                   / (f0_arr.mean() + 1e-9))
                    metrics["stability_score"] = round(_clamp(100.0 - jitter * 200.0), 1)

    if not available:
        return {
            "score": neutral,
            "indicators": ["No audio data or pre-computed acoustic analysis was "
                           "provided; speech confidence could not be measured. A "
                           "neutral value was used."],
            "available": False,
            "metrics": {},
        }

    # --- score composition (heuristic weights, documented as such) -------------
    def component(value, good_low, good_high, hard_floor=20.0):
        if value is None:
            return 60.0
        if good_low <= value <= good_high:
            return 100.0
        dist = (good_low - value) if value < good_low else (value - good_high)
        span = max(good_high - good_low, 1e-6)
        return _clamp(100.0 - (dist / span) * 80.0, hard_floor, 100.0)

    s_activity = component(metrics.get("speech_activity_ratio"), 0.55, 0.90)
    avg_pause = metrics.get("average_pause_seconds")
    n_pauses = metrics.get("pause_count")
    duration = metrics.get("duration")
    pause_rate = ((n_pauses / duration) * 60.0
                  if n_pauses is not None and duration else n_pauses)
    s_pause = component(pause_rate, 0.0, 15.0, hard_floor=10.0)
    if avg_pause is not None and avg_pause > 1.5:
        s_pause = max(10.0, s_pause - 20.0)
    s_energy = float(metrics.get("energy_adequacy", 60.0))
    s_pitch = float(metrics.get("pitch_variation_score", 60.0))
    s_stab = float(metrics.get("stability_score", 60.0))

    score = (0.30 * s_activity + 0.25 * s_pause + 0.20 * s_energy +
             0.15 * s_pitch + 0.10 * s_stab)

    indicators = [
        "Speech activity ratio: %.0f%% of frames contained speech."
        % (metrics.get("speech_activity_ratio", 0) * 100),
        "%d mid-speech pause(s) detected (avg %.2fs)."
        % (metrics.get("pause_count", 0), metrics.get("average_pause_seconds", 0)),
    ]
    if "energy_adequacy" in metrics:
        indicators.append("Speech-energy adequacy: %.0f/100." % metrics["energy_adequacy"])
    if "pitch_std_hz" in metrics:
        indicators.append(
            "Pitch variation: %.1f Hz std-dev around %.1f Hz mean."
            % (metrics["pitch_std_hz"], metrics.get("pitch_mean_hz", 0)))
    indicators.append(
        "NOTE: speech confidence is NOT a direct measurement of knowledge - a "
        "hesitant speaker can still understand the topic well.")

    return {
        "score": round(score, 1),
        "indicators": indicators,
        "available": True,
        "metrics": metrics,
    }


# --------------------------------------------------------------------------- #
# 7. EMOTION ANALYSIS
# --------------------------------------------------------------------------- #
_EMOTIONS = ("neutral", "happy", "sad", "fear", "anger", "surprise", "disgust")


def analyze_emotion(emotion_analysis):
    """Normalize facial-expression results. Emotion is ONLY supplementary
    information and is never evidence about conceptual understanding."""
    summary = {}
    if isinstance(emotion_analysis, dict):
        source = emotion_analysis.get(
            "emotion_summary",
            emotion_analysis.get("emotions", emotion_analysis))
        if isinstance(source, dict):
            total = 0.0
            for emo in _EMOTIONS:
                val = source.get(emo)
                if val is not None:
                    try:
                        summary[emo] = max(0.0, float(val))
                        total += summary[emo]
                    except (TypeError, ValueError):
                        pass
            if total > 0:
                summary = {emo: round(summary.get(emo, 0.0) / total * 100.0, 1)
                           for emo in _EMOTIONS}
                return {
                    "emotion_summary": summary,
                    "available": True,
                    "note": ("Emotion distribution across the recording. Emotion is "
                             "supplementary information ONLY and is never used as "
                             "evidence that the student understands or does not "
                             "understand the concept."),
                }
    return {
        "emotion_summary": {emo: 0.0 for emo in _EMOTIONS},
        "available": False,
        "note": ("Facial-expression analysis was not available; all emotion values "
                 "are reported as zero/unavailable. No inference about understanding "
                 "is drawn from emotion."),
    }

