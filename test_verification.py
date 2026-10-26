"""Validation test for the Student Understanding Verification module.

Runs fully offline: client=None forces the deterministic fallback paths
(lexical semantic similarity + heuristic concept analysis), so no network is
required. Also unit-checks AUVA math, level mapping and speech analysis on a
synthesized WAV.
"""

import io
import sys
import wave

import numpy as np

sys.path.insert(0, r"d:\tsa")

from verification.pipeline import UnderstandingVerificationPipeline
from verification.config import VerificationConfig
from verification.auva import (compute_auva, understanding_level,
                               knowledge_gap_penalty, misconception_penalty,
                               verification_confidence)
from verification.behavioral import analyze_speech_confidence, analyze_emotion


def make_wav(seconds=6.0, rate=16000):
    """Synthesize speech-like audio: tone bursts separated by silences."""
    t_total = np.arange(0, seconds, 1 / rate)
    sig = np.zeros_like(t_total)
    for start in np.arange(0, seconds - 1, 1.5):
        idx = (t_total >= start) & (t_total < start + 1.0)
        t = t_total[idx] - start
        sig[idx] = (0.2 * np.sin(2 * np.pi * 150 * t)
                    * (0.8 + 0.2 * np.sin(2 * np.pi * 3 * t)))
    pcm = (sig * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def check(name, cond):
    print("[%s] %s" % ("PASS" if cond else "FAIL", name))
    return bool(cond)


def main():
    ok = True
    cfg = VerificationConfig()

    # ---- unit checks -------------------------------------------------------
    cs = {"semantic_similarity": 88, "concept_coverage": 82, "attention": 70,
          "speech_confidence": 60, "knowledge_gap_penalty": 24,
          "misconception_penalty": 0}
    expected_auva = round(min(100, max(0,
        0.35 * 88 + 0.25 * 82 + 0.15 * 70 + 0.10 * 60
        - 0.10 * 24 - 0.05 * 0)), 1)
    got = compute_auva(cs, cfg.auva_weights)
    ok &= check("AUVA formula matches spec (%s)" % expected_auva,
                abs(got - expected_auva) < 0.05)

    ok &= check("Level mapping 92 -> Excellent",
                understanding_level(92, cfg.understanding_levels) == "Excellent Understanding")
    ok &= check("Level mapping 80 -> Strong",
                understanding_level(80, cfg.understanding_levels) == "Strong Understanding")
    ok &= check("Level mapping 65 -> Moderate",
                understanding_level(65, cfg.understanding_levels) == "Moderate Understanding")
    ok &= check("Level mapping 45 -> Partial",
                understanding_level(45, cfg.understanding_levels) == "Partial Understanding")
    ok &= check("Level mapping 20 -> Needs Improvement",
                understanding_level(20, cfg.understanding_levels) == "Needs Improvement")

    ok &= check("Gap penalty scales with severity",
                knowledge_gap_penalty([{"severity": "high"}], cfg.severity_weights)
                > knowledge_gap_penalty([{"severity": "low"}], cfg.severity_weights))
    ok &= check("Misconception penalty heavier than gap penalty",
                misconception_penalty([{"severity": "high"}], cfg.severity_weights)
                > knowledge_gap_penalty([{"severity": "high"}], cfg.severity_weights))

    conf, _factors = verification_confidence(
        transcript="word " * 130, num_concepts=5,
        analyses_available=[True] * 5, confidence_weights=cfg.confidence_weights)
    low_conf, _f = verification_confidence(
        transcript="few words", num_concepts=2,
        analyses_available=[True, True, False, False, False],
        confidence_weights=cfg.confidence_weights)
    ok &= check("Confidence separate & sensitive to evidence (%d vs %d)"
                % (conf, low_conf), conf >= 75 and low_conf < conf)

    wav = make_wav()
    sp = analyze_speech_confidence(audio_bytes=wav)
    ok &= check("Speech analysis available on WAV", sp["available"])
    act = sp["metrics"].get("speech_activity_ratio", 0)
    ok &= check("Bursty speech activity detected (%.2f)" % act, 0.4 <= act <= 0.9)

    em = analyze_emotion({"emotion_summary": {"neutral": 7, "happy": 3}})
    ok &= check("Emotion normalized to percentages",
                em["available"] and abs(sum(em["emotion_summary"].values()) - 100) < 0.5)
    ok &= check("Emotion unavailable handled without inference",
                not analyze_emotion(None)["available"])

    # ---- full offline pipeline ----------------------------------------------
    expected_repr = (
        "Object-oriented programming organizes software around classes, which are "
        "blueprints that define attributes and methods. Objects are instances of "
        "classes created at runtime. Inheritance lets a child class reuse and extend "
        "the members of a parent class. Polymorphism allows different classes to be "
        "used through a common interface, each responding differently to the same "
        "method call."
    )
    transcript = (
        "So a class is like a blueprint and objects are things you build from it, "
        "each object has its own data. Inheritance means you can create a class "
        "that gets stuff from another class so you don't rewrite code. Umm... "
        "polymorphism I think it means many forms but I'm not sure how it works "
        "exactly. Composition is when a class contains objects of other classes, "
        "which is basically the same thing as inheritance."
    )

    pipeline = UnderstandingVerificationPipeline(client=None, config=cfg)
    report = pipeline.run(
        subject="Computer Science",
        topic="Object-Oriented Programming",
        expected_representation=expected_repr,
        transcript=transcript,
        video_analysis={"face_presence_ratio": 0.95, "frontal_ratio": 0.88,
                        "gaze_on_camera_ratio": 0.8, "distraction_events": 1},
        audio_analysis={"speech_activity_ratio": 0.72, "pause_count": 6,
                        "average_pause_seconds": 0.6, "duration_seconds": 60},
    )

    required_keys = ["topic", "transcript", "semantic_analysis",
                     "concept_analysis", "knowledge_gaps", "misconceptions",
                     "attention_analysis", "speech_analysis", "emotion_analysis",
                     "auva", "understanding_level", "overall_explanation",
                     "evidence", "limitations"]
    missing = [k for k in required_keys if k not in report]
    ok &= check("Report contains all spec keys %s" % (missing or ""), not missing)

    comp = report["auva"]["component_scores"]
    ok &= check("All six component scores present",
                {"semantic_similarity", "concept_coverage", "attention",
                 "speech_confidence", "knowledge_gap_penalty",
                 "misconception_penalty"} <= set(comp))
    ok &= check("AUVA != confidence (separate quantities)",
                report["auva"]["score"] != report["auva"]["confidence"])
    ok &= check("Attention available with visual input",
                report["attention_analysis"]["available"] is True)
    ok &= check("Knowledge gap flagged for partially explained concept",
                any("polymorph" in g["concept"].lower()
                    for g in report["knowledge_gaps"]))
    ok &= check("Gaps have evidence+reason+severity",
                all(g.get("evidence") and g.get("reason") and
                    g["severity"] in ("low", "medium", "high")
                    for g in report["knowledge_gaps"]))
    prim = {e["source"] for e in report["evidence"] if e["type"] == "primary"}
    supp = {e["source"] for e in report["evidence"] if e["type"] == "supplementary"}
    ok &= check("Evidence split primary/supplementary",
                {"semantic_similarity", "concept_coverage"} <= prim and
                {"attention_analysis", "speech_confidence_analysis"} <= supp)
    ok &= check("Limitations disclose non-validated weights",
                any("NOT scientifically validated" in l for l in report["limitations"]))
    ok &= check("Report JSON serializable",
                isinstance(__import__("json").dumps(report), str))

    print("\n===== REPORT SUMMARY =====")
    print("Semantic:", report["semantic_analysis"]["score"])
    print("Coverage:", report["concept_analysis"]["coverage_score"],
          "| missing:", [c["concept"] for c in
                         report["concept_analysis"]["missing_concepts"]])
    print("Gaps:", len(report["knowledge_gaps"]),
          "| Misconceptions:", len(report["misconceptions"]))
    print("Attention:", report["attention_analysis"]["score"],
          "| Speech:", report["speech_analysis"]["score"])
    print("AUVA:", report["auva"]["score"],
          "| Confidence:", report["auva"]["confidence"],
          "| Level:", report["understanding_level"])

    print("\n%s" % ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

