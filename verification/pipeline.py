"""End-to-end Student Understanding Verification and Analysis pipeline.

STUDENT VIDEO -> WHISPER TRANSCRIPT -> SEMANTIC ANALYSIS -> CONCEPT COVERAGE ->
KNOWLEDGE GAP DETECTION -> MISCONCEPTION DETECTION -> ATTENTION ANALYSIS ->
SPEECH CONFIDENCE ANALYSIS -> EMOTION ANALYSIS -> AUVA -> UNDERSTANDING SCORE
-> EXPLAINABLE VERIFICATION REPORT

The final report separates PRIMARY CONCEPTUAL EVIDENCE (semantic similarity,
concept coverage, knowledge gaps, misconceptions) from SUPPLEMENTARY
BEHAVIORAL EVIDENCE (attention, speech characteristics, emotion) and never
claims that behavioral signals prove conceptual understanding.
"""

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import auva as auva_mod
from . import behavioral
from . import concepts as concept_mod
from . import semantic as semantic_mod
from .config import VerificationConfig
from .study_knowledge import get_study_knowledge

_WEIGHT_NOTE = ("AUVA weights are the initial proposed configuration defined in "
                "verification_config.json; they are NOT scientifically validated "
                "universal weights and require experimental evaluation.")

# Module-level cache for extracted concepts keyed by (expected_representation, topic).
# Avoids repeated LLM calls when the same question is verified multiple times.
_concept_cache = {}


def _concept_cache_key(expected_representation, topic):
    """Produce a stable cache key from the expected representation and topic."""
    raw = (str(expected_representation) + "||" + str(topic)).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


class UnderstandingVerificationPipeline:
    def __init__(self, client=None, config=None):
        self.client = client  # OpenAI-compatible LLM/embedding client (optional)
        self.config = config or VerificationConfig()

    def run(self, subject="", topic="", expected_representation="",
            transcript="", video_analysis=None, audio_analysis=None,
            emotion_analysis=None, audio_bytes=None,
            transcript_quality=None, audio_quality=None,
            study_knowledge=None, transcription_time_ms=None):
        timings = {}
        t_pipeline_start = time.perf_counter()

        # Step 1: Preprocessing
        t_prep_start = time.perf_counter()
        clean_t = str(transcript or "").strip()
        is_non_speech = (not clean_t or len(clean_t) < 5 or clean_t.lower() in ("(no speech detected)", "no speech", "saved automatically"))
        timings["preprocessing_ms"] = round((time.perf_counter() - t_prep_start) * 1000)

        if is_non_speech:
            timings["non_speech_early_exit"] = True
            timings["embedding_ms"] = 0
            timings["semantic_ms"] = 0
            timings["concept_coverage_ms"] = 0
            timings["misconception_detection_ms"] = 0
            timings["behavioral_ms"] = 0
            timings["llm_status"] = "SKIPPED"
            timings["total_pipeline_ms"] = round((time.perf_counter() - t_pipeline_start) * 1000)
            
            # Print latency log
            tr_str = f"{transcription_time_ms} ms" if transcription_time_ms is not None else "N/A"
            print(f"[VERIFY] Transcription: {tr_str}")
            print(f"[VERIFY] Preprocessing: {timings['preprocessing_ms']} ms")
            print(f"[VERIFY] Embedding: 0 ms")
            print(f"[VERIFY] Semantic analysis: 0 ms")
            print(f"[VERIFY] Misconception detection: 0 ms")
            print(f"[VERIFY] Face/attention: 0 ms")
            print(f"[VERIFY] LLM: SKIPPED")
            print(f"[VERIFY] Total: {timings['total_pipeline_ms']} ms")

            return self._assemble_non_speech_report(subject, topic, clean_t, expected_representation, timings)

        # Resolve study_knowledge if not explicitly passed
        if study_knowledge is None:
            study_knowledge = get_study_knowledge(subject) or get_study_knowledge(topic)
            if study_knowledge is None and ":" in str(topic):
                study_knowledge = get_study_knowledge(str(topic).split(":", 1)[0].strip())

        # Step 2: Precompute single student answer embedding once for reuse
        t_emb_start = time.perf_counter()
        student_vecs = semantic_mod.embed_texts([clean_t], client=self.client, config=self.config)
        student_emb = student_vecs[0] if student_vecs is not None and len(student_vecs) > 0 else None
        timings["embedding_ms"] = round((time.perf_counter() - t_emb_start) * 1000)

        # ------------------------------------------------------------------ #
        # PARALLEL EXECUTION of independent steps                             #
        # ------------------------------------------------------------------ #
        semantic = None
        required_concepts = None
        misconception_result = None

        # Check for precomputed ideal answer embedding in study_knowledge
        stored_ideal_vec = None
        if study_knowledge and "embeddings" in study_knowledge:
            ideal_list = study_knowledge["embeddings"].get("ideal_answers")
            if ideal_list is not None and len(ideal_list) > 0:
                stored_ideal_vec = ideal_list[0]

        def _run_semantic():
            t0 = time.perf_counter()
            result = semantic_mod.compute_semantic_similarity(
                expected_representation, transcript,
                client=self.client, config=self.config,
                expected_embedding=stored_ideal_vec,
                student_embedding=student_emb)
            timings["semantic_ms"] = round((time.perf_counter() - t0) * 1000)
            return result

        # Check if concepts are already in study_knowledge or module cache
        stored_concepts = study_knowledge.get("concepts") if study_knowledge else None
        stored_concept_embs = study_knowledge.get("embeddings", {}).get("concepts") if study_knowledge else None
        cache_key = _concept_cache_key(expected_representation, topic)
        cached_concepts = _concept_cache.get(cache_key)

        def _run_extract_concepts():
            if stored_concepts:
                timings["concept_extraction_ms"] = 0
                timings["concept_extraction_cached"] = True
                return stored_concepts
            if cached_concepts is not None:
                timings["concept_extraction_ms"] = 0
                timings["concept_extraction_cached"] = True
                return cached_concepts
            t0 = time.perf_counter()
            result = concept_mod.extract_expected_concepts(
                expected_representation, topic, client=self.client,
                config=self.config, subject=subject)
            timings["concept_extraction_ms"] = round((time.perf_counter() - t0) * 1000)
            timings["concept_extraction_cached"] = False
            _concept_cache[cache_key] = result
            return result

        with ThreadPoolExecutor(max_workers=4) as executor:
            fut_semantic = executor.submit(_run_semantic)

            if stored_concepts is not None:
                required_concepts = stored_concepts
                timings["concept_extraction_ms"] = 0
                timings["concept_extraction_cached"] = True
                fut_coverage = executor.submit(
                    concept_mod.analyze_concept_coverage,
                    required_concepts, transcript, topic, subject=subject,
                    client=self.client, config=self.config,
                    concept_embeddings=stored_concept_embs
                )
            elif cached_concepts is not None:
                required_concepts = cached_concepts
                timings["concept_extraction_ms"] = 0
                timings["concept_extraction_cached"] = True
                fut_coverage = executor.submit(
                    concept_mod.analyze_concept_coverage,
                    required_concepts, transcript, topic, subject=subject,
                    client=self.client, config=self.config
                )
            else:
                fut_concepts = executor.submit(_run_extract_concepts)
                required_concepts = fut_concepts.result()
                t0 = time.perf_counter()
                fut_coverage = executor.submit(
                    concept_mod.analyze_concept_coverage,
                    required_concepts, transcript, topic, subject=subject,
                    client=self.client, config=self.config
                )

            semantic = fut_semantic.result()
            t0 = time.perf_counter()
            coverage = fut_coverage.result()
            timings["concept_coverage_ms"] = round((time.perf_counter() - t0) * 1000)

            # Misconception detection: uses stored study knowledge first, then intelligent gating
            t0 = time.perf_counter()
            misconception_result = concept_mod.detect_misconceptions(
                transcript, expected_representation, topic,
                client=self.client, config=self.config,
                semantic_score=semantic.get("score"),
                coverage_score=coverage.get("coverage_score"),
                study_knowledge=study_knowledge,
                student_emb=student_emb
            )
            timings["misconception_detection_ms"] = round((time.perf_counter() - t0) * 1000)

        knowledge_gaps = concept_mod.detect_knowledge_gaps(coverage)

        # ------------------------------------------------------------------ #
        # SUPPLEMENTARY: attention / speech / emotion (negligible cost)       #
        # ------------------------------------------------------------------ #
        t0 = time.perf_counter()
        attention = behavioral.analyze_attention(
            video_analysis, neutral=self.config.neutral_component_score)
        speech = behavioral.analyze_speech_confidence(
            audio_bytes=audio_bytes, audio_analysis=audio_analysis,
            neutral=self.config.neutral_component_score)
        emotion = behavioral.analyze_emotion(emotion_analysis)
        timings["behavioral_ms"] = round((time.perf_counter() - t0) * 1000)

        timings["llm_status"] = f"{timings['misconception_detection_ms']} ms" if misconception_result.get("llm_used") else "SKIPPED"
        timings["total_pipeline_ms"] = round((time.perf_counter() - t_pipeline_start) * 1000)

        # Print latency measurements in exact requested format
        tr_str = f"{transcription_time_ms} ms" if transcription_time_ms is not None else "N/A"
        print(f"[VERIFY] Transcription: {tr_str}")
        print(f"[VERIFY] Preprocessing: {timings['preprocessing_ms']} ms")
        print(f"[VERIFY] Embedding: {timings['embedding_ms']} ms")
        print(f"[VERIFY] Semantic analysis: {timings['semantic_ms']} ms")
        print(f"[VERIFY] Misconception detection: {timings['misconception_detection_ms']} ms")
        print(f"[VERIFY] Face/attention: {timings['behavioral_ms']} ms")
        print(f"[VERIFY] LLM: {timings['llm_status']}")
        print(f"[VERIFY] Total: {timings['total_pipeline_ms']} ms")

        return self._assemble_report(
            subject, topic, transcript, expected_representation,
            semantic, coverage, knowledge_gaps, misconception_result,
            attention, speech, emotion,
            transcript_quality, audio_quality, timings)

    def _assemble_non_speech_report(self, subject, topic, transcript, expected_representation, timings=None):
        cfg = self.config
        component_scores = {
            "semantic_similarity": 0.0,
            "concept_coverage": 0.0,
            "attention": 50.0,
            "speech_confidence": 0.0,
            "knowledge_gap_penalty": 0.0,
            "misconception_penalty": 0.0,
        }
        auva_score = auva_mod.compute_auva(component_scores, cfg.auva_weights)
        level = auva_mod.understanding_level(auva_score, cfg.understanding_levels)
        confidence, _ = auva_mod.verification_confidence(
            transcript=transcript,
            num_concepts=0,
            analyses_available=[True, True, False, False, False],
            transcript_quality="empty",
            audio_quality=0.0,
            confidence_weights=cfg.confidence_weights,
        )

        return {
            "topic": topic,
            "transcript": transcript or "(no speech detected)",
            "semantic_analysis": {
                "score": 0.0,
                "strategy": "none (non-speech early exit)",
                "reason": "No student speech was detected in the audio.",
            },
            "concept_analysis": {
                "coverage_score": 0.0,
                "covered_concepts": [],
                "missing_concepts": [],
                "reason": "No student speech detected to evaluate concept coverage.",
                "details": [],
                "llm_used": False,
            },
            "knowledge_gaps": [],
            "misconceptions": [],
            "attention_analysis": {
                "score": 50.0,
                "available": False,
                "reason": "Attention analysis not evaluated for empty speech.",
                "metrics": {},
            },
            "speech_analysis": {
                "score": 0.0,
                "available": False,
                "reason": "No spoken audio was detected in the recording.",
                "metrics": {},
            },
            "emotion_analysis": {
                "available": False,
                "dominant_emotion": "unknown",
                "distribution": {},
                "reason": "Emotion analysis not evaluated for empty speech.",
            },
            "auva": {
                "score": auva_score,
                "confidence": confidence,
                "formula": "AUVA = sum(w_i * S_i) - penalties",
                "weights": cfg.auva_weights,
                "component_scores": component_scores,
            },
            "understanding_level": level,
            "overall_explanation": "No student speech was detected in the recording. Please ensure your microphone is enabled and speak your explanation clearly.",
            "evidence": [
                {"type": "primary", "source": "semantic_similarity", "detail": "Semantic similarity 0.0/100"},
                {"type": "primary", "source": "concept_coverage", "detail": "Concept coverage 0.0/100"},
                {"type": "supplementary", "source": "speech_analysis", "detail": "No speech detected"},
            ],
            "limitations": [
                _WEIGHT_NOTE,
                "No speech was available to analyze for conceptual understanding."
            ],
            "timings": timings or {},
        }

    # ---------------------------------------------------------------------- #
    def _assemble_report(self, subject, topic, transcript,
                         expected_representation,
                         semantic, coverage, knowledge_gaps,
                         misconception_result, attention, speech, emotion,
                         transcript_quality, audio_quality, timings=None):
        cfg = self.config

        gap_penalty = auva_mod.knowledge_gap_penalty(
            knowledge_gaps, cfg.severity_weights)
        mis_penalty = auva_mod.misconception_penalty(
            misconception_result["misconceptions"], cfg.severity_weights)

        component_scores = {
            "semantic_similarity": semantic["score"],
            "concept_coverage": coverage["coverage_score"],
            "attention": attention["score"],
            "speech_confidence": speech["score"],
            "knowledge_gap_penalty": gap_penalty,
            "misconception_penalty": mis_penalty,
        }
        auva_score = auva_mod.compute_auva(component_scores, cfg.auva_weights)
        level = auva_mod.understanding_level(auva_score, cfg.understanding_levels)

        confidence, confidence_factors = auva_mod.verification_confidence(
            transcript=transcript,
            num_concepts=len(coverage.get("details", [])),
            analyses_available=[
                True, True,                       # semantic + coverage always ran
                attention["available"], speech["available"], emotion["available"],
            ],
            transcript_quality=transcript_quality,
            audio_quality=(audio_quality if audio_quality is not None
                           else speech.get("metrics", {}).get("energy_adequacy")),
            confidence_weights=cfg.confidence_weights,
        )

        # ------------- evidence: primary vs supplementary -------------------- #
        evidence = [
            {"type": "primary", "source": "semantic_similarity",
             "detail": "Semantic similarity %s/100 - %s"
                       % (semantic["score"], semantic["reason"])},
            {"type": "primary", "source": "concept_coverage",
             "detail": coverage["reason"]},
        ]
        for gap in knowledge_gaps:
            evidence.append({
                "type": "primary", "source": "knowledge_gap",
                "detail": "Concept '%s' (%s severity): %s Evidence: %s"
                          % (gap["concept"], gap["severity"], gap["reason"],
                             gap["evidence"])})
        for mis in misconception_result["misconceptions"]:
            evidence.append({
                "type": "primary", "source": "misconception",
                "detail": 'The student stated: "%s". Expected understanding: %s. %s'
                          % (mis["statement"], mis["expected_understanding"],
                             mis["explanation"])})
        for ind in attention["indicators"]:
            evidence.append({"type": "supplementary",
                             "source": "attention_analysis", "detail": ind})
        for ind in speech["indicators"]:
            evidence.append({"type": "supplementary",
                             "source": "speech_confidence_analysis", "detail": ind})
        if emotion["available"]:
            evidence.append({
                "type": "supplementary", "source": "emotion_analysis",
                "detail": "Emotion distribution recorded: %s. Supplementary only."
                          % emotion["emotion_summary"]})

        # ---------------- limitations ---------------------------------------- #
        limitations = [_WEIGHT_NOTE,
                       "Semantic similarity alone is NOT proof of understanding; it "
                       "is combined with concept-level analysis in AUVA.",
                       "Attention and speech confidence are SUPPLEMENTARY behavioral "
                       "evidence; they never directly prove or disprove conceptual "
                       "understanding.",
                       "Emotion is never used as direct evidence of understanding."]
        if not attention["available"]:
            limitations.append("No visual-analysis input was available, so the "
                               "attention score is a neutral placeholder.")
        if not speech["available"]:
            limitations.append("No audio data was available, so the speech-confidence "
                               "score is a neutral placeholder.")
        if not emotion["available"]:
            limitations.append("Facial-expression analysis was unavailable; the "
                               "emotion summary is empty/unavailable.")
        if not coverage.get("llm_used"):
            limitations.append("LLM-based concept evaluation was unavailable; a "
                               "heuristic mention-based fallback was used.")
        if not misconception_result.get("llm_used"):
            limitations.append("LLM-based misconception review was unavailable.")

        # ---------------- overall explanation -------------------------------- #
        gaps_txt = ("; ".join("'%s' - %s" % (g["concept"], g["reason"])
                              for g in knowledge_gaps)
                    or "no significant gaps identified")
        mis_txt = ("; ".join(m["explanation"] for m in
                             misconception_result["misconceptions"])
                   if misconception_result["misconceptions"]
                   else "no genuine conceptual errors detected")
        overall = (
            "Understanding level '%s' with an AUVA score of %.1f/100 and a separate "
            "verification confidence of %.1f/100. PRIMARY CONCEPTUAL EVIDENCE - "
            "SEMANTIC: %s CONCEPT COVERAGE: %s KNOWLEDGE GAPS: %s MISCONCEPTIONS: %s "
            "SUPPLEMENTARY BEHAVIORAL EVIDENCE (never proof of understanding): "
            "attention %s/100%s, speech confidence %s/100%s."
            % (level, auva_score, confidence, semantic["reason"],
               coverage["reason"], gaps_txt, mis_txt,
               attention["score"],
               "" if attention["available"] else " (neutral placeholder)",
               speech["score"],
               "" if speech["available"] else " (neutral placeholder)")
        )

        report = {
            "topic": topic,
            "subject": subject,
            "transcript": transcript,
            "expected_representation": expected_representation,
            "required_concepts": [c["concept"] for c in coverage.get("details", [])],
            "semantic_analysis": {"score": semantic["score"],
                                  "reason": semantic["reason"]},
            "concept_analysis": {
                "covered_concepts": coverage["covered_concepts"],
                "missing_concepts": coverage["missing_concepts"],
                "coverage_score": coverage["coverage_score"],
                "reason": coverage["reason"],
            },
            "knowledge_gaps": knowledge_gaps,
            "misconceptions": misconception_result["misconceptions"],
            "misconception_analysis_reason": misconception_result["reason"],
            "attention_analysis": {
                "score": attention["score"],
                "indicators": attention["indicators"],
                "available": attention["available"],
            },
            "speech_analysis": {
                "score": speech["score"],
                "indicators": speech["indicators"],
                "available": speech["available"],
                "metrics": speech.get("metrics", {}),
            },
            "emotion_analysis": emotion,
            "auva": {
                "score": auva_score,
                "confidence": confidence,
                "confidence_factors": confidence_factors,
                "component_scores": component_scores,
                "weights_used": dict(cfg.auva_weights),
                "note": _WEIGHT_NOTE,
            },
            "understanding_level": level,
            "overall_explanation": overall,
            "evidence_classification": {
                "primary_conceptual_evidence": [
                    "semantic_similarity", "concept_coverage",
                    "knowledge_gaps", "misconceptions"],
                "supplementary_behavioral_evidence": [
                    "attention", "speech_characteristics", "emotion"],
                "rule": ("Behavioral signals (attention, speech, emotion) are "
                         "supplementary only and NEVER prove conceptual "
                         "understanding."),
            },
            "evidence": evidence,
            "limitations": limitations,
        }
        if timings:
            report["timings"] = timings
        return report
