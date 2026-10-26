"""Concept coverage, knowledge-gap detection and misconception detection.

These are PRIMARY CONCEPTUAL EVIDENCE in the verification pipeline. The LLM is
instructed to judge whether each concept is clearly / partially / not explained
based on the student's actual explanation - never merely on name-dropping -
and to report only genuine conceptual errors as misconceptions.

Every function degrades gracefully to a deterministic heuristic fallback when
the LLM client is unavailable or returns unparseable output.
"""

import re
import numpy as np

from .llm import call_llm_json
from .semantic import _tokens, embed_texts, cosine_similarity

_STATUS_SCORE = {"clearly_explained": 1.0, "partially_explained": 0.5, "not_explained": 0.0}
_SEVERITIES = ("low", "medium", "high")

_COVERAGE_SYSTEM = """You are an expert examiner analyzing how well a student verbally explains a topic.
Rules you MUST follow:
- Mark a concept "clearly_explained" ONLY if the student actually explains how/why it works, with their own reasoning, example or mechanism.
- Merely MENTIONING a concept's name without explaining it is at most "partially_explained".
- Do NOT penalize different wording, different examples, grammar, accent or speaking style.
- Respond with STRICT JSON only. No commentary."""


def _heuristic_concepts(expected_representation, topic):
    """Fallback concept extraction: split the expected representation into
    individual explanatory statements and treat each as a required concept."""
    if isinstance(expected_representation, dict):
        concepts = expected_representation.get("key_concepts")
        if concepts:
            parsed = []
            for c in concepts:
                if isinstance(c, dict):
                    parsed.append({"concept": str(c.get("name", "")),
                                   "description": str(c.get("description", ""))})
                else:
                    parsed.append({"concept": str(c), "description": ""})
            return [p for p in parsed if p["concept"]]
        expected_representation = expected_representation.get("summary", "") or str(expected_representation)

    text = str(expected_representation or topic or "").strip()
    if not text:
        return [{"concept": str(topic or "topic"), "description": ""}]
    sentences = [s.strip() for s in re.split(r"[.;\n]+", text) if len(s.strip()) > 8]
    return ([{"concept": s[:120], "description": s} for s in sentences]
            or [{"concept": text[:120], "description": text}])


_CONCEPTS_EXTRACT_CACHE = {}


def extract_expected_concepts(expected_representation, topic, client=None,
                              config=None, subject=""):
    """Return a list of {"concept", "description"} required for the topic."""
    cache_key = (str(topic or "").strip().lower(), str(expected_representation or "").strip())
    if cache_key in _CONCEPTS_EXTRACT_CACHE:
        return _CONCEPTS_EXTRACT_CACHE[cache_key]

    if client is not None:
        try:
            payload = call_llm_json(
                client, config.llm_model,
                "You extract the essential concepts required to understand a topic. "
                "Respond with STRICT JSON only.",
                'Topic: "%s" (subject: %s).\nExpected conceptual representation of the topic:\n'
                '"""%s"""\n\nList every important concept a student MUST explain to demonstrate '
                "understanding.\nReturn JSON: "
                '{"concepts": [{"concept": "...", "description": "what must be understood about it"}]}'
                % (topic, subject or "general", expected_representation),
                temperature=0.1,
                max_tokens=300,
            )
            items = payload.get("concepts") if isinstance(payload, dict) else payload
            parsed = [{"concept": str(i.get("concept", "")).strip(),
                       "description": str(i.get("description", "")).strip()}
                      for i in items if isinstance(i, dict) and i.get("concept")]
            if parsed:
                _CONCEPTS_EXTRACT_CACHE[cache_key] = parsed
                return parsed
        except Exception:
            pass
    fallback = _heuristic_concepts(expected_representation, topic)
    if fallback:
        _CONCEPTS_EXTRACT_CACHE[cache_key] = fallback
    return fallback


_COVERAGE_USER_TMPL = """Topic: "{topic}"
Subject: {subject}

Required concepts (with what must be understood about each):
{concept_list}

Student's spoken explanation (transcript):
\"\"\"{transcript}\"\"\"

For EACH required concept decide:
- "status": "clearly_explained" | "partially_explained" | "not_explained"
- "evidence": short quote or paraphrase from the transcript supporting your decision (or "" if absent)
- If status is not "clearly_explained": "gap_reason" = why the explanation is insufficient, and
  "gap_severity": "low" | "medium" | "high" (high = core mechanism missing entirely).

Return STRICT JSON:
{{"coverage": [{{"concept": "...", "status": "...", "evidence": "...",
   "gap_reason": "...", "gap_severity": "..."}}]}}"""


_CONCEPT_EMBED_CACHE = {}
_MISCONCEPTION_CACHE = {}


def get_or_create_concept_embeddings(concepts, client=None, config=None):
    """Retrieve cached concept embeddings or generate them once."""
    if not concepts:
        return None
    key = tuple((c["concept"], c.get("description", "")) for c in concepts)
    if key in _CONCEPT_EMBED_CACHE:
        return _CONCEPT_EMBED_CACHE[key]

    texts = [f"{c['concept']}. {c.get('description', '')}".strip() for c in concepts]
    vecs = embed_texts(texts, client=client, config=config)
    if vecs is not None:
        _CONCEPT_EMBED_CACHE[key] = vecs
        return vecs
    return None


def analyze_concept_coverage(concepts, transcript, topic, subject="",
                             client=None, config=None, concept_embeddings=None):
    """Return coverage result using cached embedding cosine similarity, falling back to lexical analysis."""
    clean_t = str(transcript or "").strip()
    if not clean_t or len(clean_t) < 5 or clean_t.lower() in ("(no speech detected)", "no speech", "saved automatically"):
        details = [{
            "concept": c["concept"],
            "status": "not_explained",
            "evidence": "",
            "gap_reason": "No student speech detected to evaluate concept coverage.",
            "gap_severity": "high",
            "credit": 0.0
        } for c in concepts]
        return {
            "covered_concepts": [],
            "missing_concepts": [{"concept": c["concept"], "status": "not_explained"} for c in concepts],
            "coverage_score": 0.0,
            "reason": "No student speech detected to evaluate concept coverage.",
            "details": details,
            "llm_used": False,
        }

    # 1. Primary fast path: Cached Embedding-based Concept Coverage
    if concept_embeddings is None:
        concept_embeddings = get_or_create_concept_embeddings(concepts, client=client, config=config)

    cov_cfg = getattr(config, "concept_coverage_cfg", {}) if config else {}
    HIGH_THRESHOLD = cov_cfg.get("high_threshold", 0.65)
    LOW_THRESHOLD = cov_cfg.get("low_threshold", 0.38)

    details, llm_used = None, False

    if concept_embeddings is not None and len(concept_embeddings) == len(concepts):
        try:
            sentences = [s.strip() for s in re.split(r"[.!?\n]+", clean_t) if len(s.strip()) > 5]
            if not sentences:
                sentences = [clean_t]

            all_stu_texts = [clean_t] + sentences
            stu_vecs = embed_texts(all_stu_texts, client=client, config=config)

            if stu_vecs is not None and len(stu_vecs) == len(all_stu_texts):
                full_stu_vec = stu_vecs[0]
                sent_vecs = stu_vecs[1:] if len(stu_vecs) > 1 else [full_stu_vec]

                details = []
                for i, c in enumerate(concepts):
                    con_vec = concept_embeddings[i]
                    sim_full = cosine_similarity(con_vec, full_stu_vec)

                    sent_sims = [cosine_similarity(con_vec, sv) for sv in sent_vecs]
                    best_idx = int(np.argmax(sent_sims))
                    best_sim = float(sent_sims[best_idx])
                    best_sent = sentences[best_idx]

                    effective_sim = best_sim * 0.85 + sim_full * 0.15

                    if effective_sim >= HIGH_THRESHOLD:
                        status = "clearly_explained"
                        severity = "low"
                        evidence = best_sent
                        gap_reason = "Concept clearly demonstrated and explained in student explanation."
                    elif effective_sim >= LOW_THRESHOLD:
                        status = "partially_explained"
                        severity = "medium"
                        evidence = best_sent
                        gap_reason = "Concept was partially addressed, but key supporting details or mechanisms were omitted."
                    else:
                        status = "not_explained"
                        severity = "high"
                        evidence = ""
                        gap_reason = "Not addressed in the student's explanation."

                    details.append({
                        "concept": c["concept"],
                        "status": status,
                        "evidence": evidence,
                        "gap_reason": gap_reason,
                        "gap_severity": severity,
                    })
        except Exception:
            details = None

    # Fallback to lexical token-overlap matching if embeddings unavailable
    if details is None:
        stu_tokens = set(_tokens(clean_t))
        details = []
        for c in concepts:
            con_tokens = set(_tokens(c["concept"] + " " + c.get("description", "")))
            overlap = len(con_tokens & stu_tokens)
            ratio = overlap / len(con_tokens) if con_tokens else 0.0
            if ratio >= 0.7:
                status, severity = "clearly_explained", "low"
                reason = "Most key terms of this concept were used with supporting explanation in the transcript."
            elif ratio >= 0.35:
                status, severity = "partially_explained", "medium"
                reason = "Only part of this concept was addressed; the core mechanism is not fully explained in the transcript."
            else:
                status, severity = "not_explained", "high"
                reason = "Not addressed anywhere in the student's explanation."
            details.append({
                "concept": c["concept"],
                "status": status,
                "evidence": ("; ".join(sorted(con_tokens & stu_tokens)) if overlap else ""),
                "gap_reason": reason,
                "gap_severity": severity,
            })

    covered, missing = [], []
    total, achieved = 0.0, 0.0
    for item in details:
        weight = _STATUS_SCORE[item["status"]]
        total += 1.0
        achieved += weight
        item["credit"] = weight
        entry = {"concept": item["concept"], "status": item["status"]}
        (missing if item["status"] == "not_explained" else covered).append(entry)

    coverage_score = round(100.0 * achieved / total, 1) if total else 0.0
    reason = (
        "The student explained %d concept(s) clearly, partially explained %d, and did not "
        "address %d of %d required concept(s)."
        % (sum(1 for d in details if d["status"] == "clearly_explained"),
           sum(1 for d in details if d["status"] == "partially_explained"),
           len(missing), len(details))
    )
    return {
        "covered_concepts": covered,
        "missing_concepts": missing,
        "coverage_score": coverage_score,
        "reason": reason,
        "details": details,
        "llm_used": llm_used,
    }


def detect_knowledge_gaps(coverage_result):
    """Build knowledge gaps from concepts that were not clearly explained."""
    gaps = []
    for item in coverage_result["details"]:
        if item["status"] == "clearly_explained":
            continue
        gaps.append({
            "concept": item["concept"],
            "evidence": item["evidence"] or "(no relevant explanation found in transcript)",
            "reason": item["gap_reason"],
            "severity": item["gap_severity"],
        })
    return gaps


_MISCONCEPTION_SYSTEM = """You are a strict but fair subject-matter expert reviewing a student's verbal explanation for GENUINE conceptual errors.
You MUST NOT classify any of the following as misconceptions:
- different wording or phrasing of correct ideas
- different (but valid) examples or analogies
- grammar mistakes, accent artifacts or transcription errors
- speaking style, hesitations, or minor terminology variations
Only report statements that are factually or conceptually WRONG about the topic.
If there are none, return an empty list. Respond with STRICT JSON only."""


_MISCONCEPTION_USER_TMPL = """Topic: "{topic}"
Expected understanding:
\"\"\"{expected}\"\"\"

Student's explanation (transcript):
\"\"\"{transcript}\"\"\"

Identify genuine conceptual errors in the student's explanation.
For each error provide: statement (what the student said), expected_understanding (what is actually correct),
explanation (why the student's statement is wrong), severity (low|medium|high).
Return STRICT JSON: {{"misconceptions": [...]}}"""


def detect_misconceptions(transcript, expected_representation, topic,
                          client=None, config=None, semantic_score=None,
                          coverage_score=None, study_knowledge=None,
                          student_emb=None):
    clean_t = str(transcript or "").strip()
    if not clean_t or len(clean_t) < 8 or clean_t.lower() in ("(no speech detected)", "no speech", "saved automatically"):
        return {
            "misconceptions": [],
            "reason": "No student speech detected to evaluate for misconceptions.",
            "llm_used": False
        }

    # 1. Primary fast path: Compare against stored study-session knowledge
    if study_knowledge is not None:
        try:
            from .study_knowledge import match_stored_misconceptions
            matched = match_stored_misconceptions(
                clean_t, study_knowledge, student_emb=student_emb,
                client=client, config=config, threshold=0.55
            )
            if matched:
                return {
                    "misconceptions": matched,
                    "reason": "%d genuine conceptual error(s) matched from stored study knowledge." % len(matched),
                    "llm_used": False,
                    "source": "stored_study_knowledge"
                }
        except Exception:
            pass

    # 2. Intelligent Gating: For clearly aligned answers or off-topic, skip expensive LLM call
    if semantic_score is not None and coverage_score is not None:
        if semantic_score >= 70.0 and coverage_score >= 60.0:
            return {
                "misconceptions": [],
                "reason": "Lightweight analysis confirmed strong conceptual alignment with no misconceptions.",
                "llm_used": False
            }
        if semantic_score < 20.0:
            return {
                "misconceptions": [],
                "reason": "Answer does not address topic mechanism sufficiently to identify misconceptions (knowledge gap).",
                "llm_used": False
            }

    # Check in-memory cache
    cache_key = (str(topic).strip().lower(), str(expected_representation).strip(), clean_t)
    if cache_key in _MISCONCEPTION_CACHE:
        return _MISCONCEPTION_CACHE[cache_key]

    misconceptions, llm_used = [], False
    if client is not None:
        try:
            payload = call_llm_json(
                client, config.llm_model, _MISCONCEPTION_SYSTEM,
                _MISCONCEPTION_USER_TMPL.format(topic=topic,
                                                expected=expected_representation,
                                                transcript=clean_t),
                temperature=0.1,
                max_tokens=250)
            rows = payload.get("misconceptions") if isinstance(payload, dict) else payload
            for row in rows or []:
                sev = str(row.get("severity", "medium")).strip().lower()
                misconceptions.append({
                    "statement": str(row.get("statement", "")).strip(),
                    "expected_understanding": str(row.get("expected_understanding", "")).strip(),
                    "explanation": str(row.get("explanation", "")).strip(),
                    "severity": sev if sev in _SEVERITIES else "medium",
                })
            misconceptions = [m for m in misconceptions if m["statement"]]
            llm_used = True
        except Exception:
            misconceptions, llm_used = [], False

    if misconceptions:
        reason = ("%d genuine conceptual error(s) were identified in the student's "
                  "explanation." % len(misconceptions))
    else:
        reason = ("Misconception review found no genuine conceptual errors "
                  "in the student's explanation.")

    result = {"misconceptions": misconceptions, "reason": reason, "llm_used": llm_used}
    _MISCONCEPTION_CACHE[cache_key] = result
    return result
