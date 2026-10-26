"""AUVA - Adaptive Understanding Verification Algorithm.

AUVA combines PRIMARY CONCEPTUAL EVIDENCE (semantic similarity, concept
coverage) with SUPPLEMENTARY BEHAVIORAL EVIDENCE (attention, speech
confidence), minus penalties for knowledge gaps and misconceptions.

IMPORTANT: the default weights are an INITIAL PROPOSED CONFIGURATION loaded
from verification_config.json. They are NOT scientifically validated universal
weights and must be experimentally evaluated before being relied upon.
"""

import math

from .behavioral import analyze_emotion  # noqa: F401  (re-export convenience)


def _clamp(value, low=0.0, high=100.0):
    return max(low, min(high, float(value)))


def knowledge_gap_penalty(gaps, severity_weights):
    """Penalty 0-100 from knowledge-gap severities.

    Each gap contributes its severity weight scaled by a per-gap cap; the total
    is capped at 100 so the AUVA subtraction stays well-defined.
    """
    if not gaps:
        return 0.0
    total = 0.0
    for gap in gaps:
        sev = str(gap.get("severity", "medium")).lower()
        total += float(severity_weights.get(sev, 0.6)) * 40.0
    return round(_clamp(total), 1)


def misconception_penalty(misconceptions, severity_weights):
    """Penalty 0-100 from genuine misconceptions (heavier than gaps because a
    false belief actively contradicts correct understanding)."""
    if not misconceptions:
        return 0.0
    total = 0.0
    for mis in misconceptions:
        sev = str(mis.get("severity", "medium")).lower()
        total += float(severity_weights.get(sev, 0.6)) * 60.0
    return round(_clamp(total), 1)


def compute_auva(component_scores, weights):
    """Apply the configurable AUVA formula. All components must be 0-100."""
    w = {
        "semantic_similarity": weights.get("semantic_similarity", 0.35),
        "concept_coverage": weights.get("concept_coverage", 0.25),
        "attention": weights.get("attention", 0.15),
        "speech_confidence": weights.get("speech_confidence", 0.10),
        "knowledge_gap_penalty": weights.get("knowledge_gap_penalty", 0.10),
        "misconception_penalty": weights.get("misconception_penalty", 0.05),
    }
    c = component_scores
    raw = (
        w["semantic_similarity"] * c["semantic_similarity"]
        + w["concept_coverage"] * c["concept_coverage"]
        + w["attention"] * c["attention"]
        + w["speech_confidence"] * c["speech_confidence"]
        - w["knowledge_gap_penalty"] * c["knowledge_gap_penalty"]
        - w["misconception_penalty"] * c["misconception_penalty"]
    )
    return round(_clamp(raw), 1)


def understanding_level(score, levels_config):
    """Map an AUVA score to a level label using configurable thresholds."""
    for entry in levels_config:  # already sorted highest-first by config loader
        if score >= entry["min"]:
            return entry["label"]
    return levels_config[-1]["label"] if levels_config else "Needs Improvement"


def verification_confidence(transcript, num_concepts, analyses_available,
                            transcript_quality=None, audio_quality=None,
                            confidence_weights=None):
    """Compute how RELIABLE the verification result is (0-100).

    This is deliberately separate from the AUVA score: confidence describes
    evidence sufficiency/quality, NOT the student's level of understanding.
      - explanation_amount   : word count of the student's explanation
      - transcript_quality   : ASR quality signal (e.g. Whisper avg_logprob)
      - audio_quality        : acoustic recording quality when measurable
      - concepts_evaluated   : how many required concepts were assessed
      - analysis_completeness: which modalities were actually available
    """
    cw = confidence_weights or {}

    words = len((transcript or "").split())
    amount = _clamp(words / 120.0)  # ~120 words considered a full explanation

    quality = 80.0
    quality_note = "transcript quality unknown; assumed adequate"
    if transcript_quality is not None:
        # Accept either a direct 0-100 score or a Whisper avg_logprob.
        try:
            tq = float(transcript_quality)
            quality = _clamp(tq) if tq <= 100 else _clamp(100.0 + tq * 20.0)
            quality_note = "transcript-quality signal provided by ASR"
        except (TypeError, ValueError):
            pass

    aq = 0.7
    aq_note = "audio quality unknown; assumed adequate"
    if audio_quality is not None:
        try:
            aqv = float(audio_quality)
            aq = _clamp(aqv) if aqv <= 100 else _clamp(aqv)
            aq_note = "audio-quality metric provided by acoustic analysis"
        except (TypeError, ValueError):
            pass

    concepts_factor = _clamp((num_concepts or 0) / 5.0)

    completeness = sum(1 for a in analyses_available if a) / max(1, len(analyses_available))

    score = (
        cw.get("explanation_amount", 0.20) * amount * 100.0
        + cw.get("transcript_quality", 0.20) * quality
        + cw.get("audio_quality", 0.15) * aq
        + cw.get("concepts_evaluated", 0.20) * concepts_factor * 100.0
        + cw.get("analysis_completeness", 0.25) * completeness * 100.0
    )
    return round(_clamp(score), 1), {
        "explanation_amount": {"words": words, "factor": round(amount, 2)},
        "transcript_quality": {"score": round(quality, 1), "note": quality_note},
        "audio_quality": {"score": round(aq, 1), "note": aq_note},
        "concepts_evaluated": {"count": num_concepts,
                               "factor": round(concepts_factor, 2)},
        "analysis_completeness": {"available_modalities":
                                  [str(i) for i, a in enumerate(analyses_available) if a],
                                  "ratio": round(completeness, 2)},
    }
