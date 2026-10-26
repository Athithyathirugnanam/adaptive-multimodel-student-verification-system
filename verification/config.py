"""Configuration loader for the AMS-LAS Student Understanding Verification module."""

import copy
import json
import os

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "verification_config.json",
)

DEFAULT_CONFIG = {
    "auva_weights": {
        "semantic_similarity": 0.35,
        "concept_coverage": 0.25,
        "attention": 0.15,
        "speech_confidence": 0.10,
        "knowledge_gap_penalty": 0.10,
        "misconception_penalty": 0.05,
    },
    "understanding_levels": [
        {"min": 90, "label": "Excellent Understanding"},
        {"min": 75, "label": "Strong Understanding"},
        {"min": 60, "label": "Moderate Understanding"},
        {"min": 40, "label": "Partial Understanding"},
        {"min": 0, "label": "Needs Improvement"},
    ],
    "severity_weights": {"low": 0.30, "medium": 0.60, "high": 1.00},
    "neutral_component_score": 50,
    "llm_model": "meta/llama-3.2-11b-vision-instruct",
    "concept_coverage": {
        "high_threshold": 0.60,
        "low_threshold": 0.38,
        "ambiguous_llm_fallback": False,
    },
    "embedding": {
        "strategy_order": ["sentence_transformers", "api", "lexical"],
        "sentence_transformer_model": "all-MiniLM-L6-v2",
        "api_embedding_model": "nvidia/nv-embedqa-e5-v5",
    },
    "confidence_weights": {
        "explanation_amount": 0.20,
        "transcript_quality": 0.20,
        "audio_quality": 0.15,
        "concepts_evaluated": 0.20,
        "analysis_completeness": 0.25,
    },
}


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class VerificationConfig:
    """Loads verification_config.json on top of safe defaults."""

    def __init__(self, path=None):
        self.path = path or _CONFIG_PATH
        self.data = copy.deepcopy(DEFAULT_CONFIG)
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self.data = _deep_merge(DEFAULT_CONFIG, json.load(fh))
            except (ValueError, OSError):
                # Fall back to defaults if the config file is malformed.
                self.data = copy.deepcopy(DEFAULT_CONFIG)

    # --- convenience accessors -------------------------------------------------
    @property
    def auva_weights(self):
        return self.data["auva_weights"]

    @property
    def understanding_levels(self):
        # Highest threshold first for straightforward lookup.
        return sorted(self.data["understanding_levels"], key=lambda l: l["min"], reverse=True)

    @property
    def severity_weights(self):
        return self.data["severity_weights"]

    @property
    def neutral_component_score(self):
        return self.data.get("neutral_component_score", 50)

    @property
    def llm_model(self):
        return self.data.get("llm_model", "meta/llama-3.2-11b-vision-instruct")

    @property
    def whisper_cfg(self):
        """Speech-to-text backend selection (backend/model_size/api_model)."""
        return self.data.get("whisper", {})

    @property
    def concept_coverage_cfg(self):
        return self.data.get("concept_coverage", {})

    @property
    def embedding_cfg(self):
        return self.data.get("embedding", {})

    @property
    def confidence_weights(self):
        return self.data["confidence_weights"]
