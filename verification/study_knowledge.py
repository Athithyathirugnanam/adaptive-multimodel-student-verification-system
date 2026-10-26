"""Study session knowledge management, precomputation, and local vector matching.

Captures, persists, and vectorizes study materials generated during the learning
session so verification can execute 100% locally in milliseconds without
repeated LLM calls.
"""

import json
import os
import re
import threading
import numpy as np

from .semantic import embed_texts, cosine_similarity

_KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "study_sessions")
os.makedirs(_KNOWLEDGE_DIR, exist_ok=True)

_IN_MEMORY_STORE = {}
_STORE_LOCK = threading.Lock()


def _normalize_topic(topic):
    return str(topic or "").strip().lower()


def _extract_section(markdown_text, section_name):
    pattern = rf"##\s*{re.escape(section_name)}\s*\n([\s\S]*?)(?=\n##|$)"
    match = re.search(pattern, markdown_text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def build_study_knowledge_from_explanation(topic, level, explanation):
    """Parse generated study explanation markdown into a structured study_knowledge object."""
    explanation = str(explanation or "").strip()
    topic = str(topic or "").strip()

    definition_text = _extract_section(explanation, "Definition")
    key_concepts_text = _extract_section(explanation, "Key Concepts")
    how_it_works_text = _extract_section(explanation, "How It Works")
    real_world_text = _extract_section(explanation, "Real-World Example")
    misconceptions_text = _extract_section(explanation, "Common Misconceptions")
    summary_text = _extract_section(explanation, "Quick Summary")

    # Parse Key Concepts
    concepts = []
    key_points = []
    for line in key_concepts_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line_clean = re.sub(r"^[-*•\d.]+\s*", "", line).strip()
        if not line_clean:
            continue
        # Check for bold term: **Term**: Description
        m = re.match(r"^\*\*([^*]+)\*\*[:\s-]*(.*)$", line_clean)
        if m:
            c_name = m.group(1).strip()
            c_desc = m.group(2).strip() or c_name
            concepts.append({"concept": c_name, "description": c_desc})
            key_points.append(f"{c_name}: {c_desc}")
        else:
            parts = line_clean.split(":", 1)
            if len(parts) == 2 and len(parts[0]) < 60:
                concepts.append({"concept": parts[0].strip(), "description": parts[1].strip()})
            else:
                concepts.append({"concept": line_clean[:80], "description": line_clean})
            key_points.append(line_clean)

    if not concepts and definition_text:
        concepts.append({"concept": topic, "description": definition_text})
        key_points.append(definition_text)

    # Parse Common Misconceptions
    misconceptions = []
    for line in misconceptions_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line_clean = re.sub(r"^[-*•\d.]+\s*", "", line).strip()
        if not line_clean or len(line_clean) < 10:
            continue
        
        # Look for "Misconception: ... Reality/Fact: ..." or similar patterns
        m = re.search(r"\*\*([^*]+)\*\*[:\s-]*(.*)", line_clean)
        if m:
            statement = m.group(1).strip()
            explanation_part = m.group(2).strip()
        else:
            parts = line_clean.split(" - ", 1)
            if len(parts) == 2:
                statement = parts[0].strip()
                explanation_part = parts[1].strip()
            else:
                statement = line_clean
                explanation_part = f"This is a common misconception about {topic}."

        expected_under = f"Understanding that contrary to '{statement}', the correct mechanism operates as detailed in {topic}."
        misconceptions.append({
            "statement": statement,
            "expected_understanding": expected_under,
            "explanation": explanation_part or f"Incorrect understanding of {topic}.",
            "severity": "medium"
        })

    # Examples
    examples = [real_world_text] if real_world_text else []

    # Keywords
    words = re.findall(r"\b[a-zA-Z]{4,}\b", (definition_text + " " + key_concepts_text).lower())
    stopwords = {"this", "that", "with", "from", "have", "were", "when", "what", "which", "there", "their", "about"}
    keywords = sorted(list(set(w for w in words if w not in stopwords)))[:20]

    # Ideal answers / expected representations
    ideal_answers = []
    if definition_text and how_it_works_text:
        ideal_answers.append(f"{definition_text} {how_it_works_text}")
    elif definition_text:
        ideal_answers.append(definition_text)
    elif summary_text:
        ideal_answers.append(summary_text)

    knowledge = {
        "topic": topic,
        "level": level,
        "explanation": explanation,
        "definition": definition_text,
        "concepts": concepts,
        "examples": examples,
        "key_points": key_points,
        "keywords": keywords,
        "ideal_answers": ideal_answers,
        "misconceptions": misconceptions,
        "questions": [],
        "question_knowledge": {},
        "embeddings": {}
    }

    return knowledge


def precompute_knowledge_embeddings(knowledge, client=None, config=None):
    """Generate and cache embeddings for all concepts, ideal answers, and misconceptions once."""
    if not isinstance(knowledge, dict):
        return knowledge

    embeddings = {}

    # 1. Concept embeddings
    concepts = knowledge.get("concepts", [])
    if concepts:
        c_texts = [f"{c['concept']}. {c.get('description', '')}".strip() for c in concepts]
        c_vecs = embed_texts(c_texts, client=client, config=config)
        if c_vecs is not None:
            embeddings["concepts"] = c_vecs

    # 2. Misconception embeddings
    misconceptions = knowledge.get("misconceptions", [])
    if misconceptions:
        m_texts = [f"{m.get('statement', '')}. {m.get('explanation', '')}".strip() for m in misconceptions]
        m_vecs = embed_texts(m_texts, client=client, config=config)
        if m_vecs is not None:
            embeddings["misconceptions"] = m_vecs

    # 3. Ideal answer embeddings
    ideal_answers = knowledge.get("ideal_answers", [])
    if ideal_answers:
        i_vecs = embed_texts(ideal_answers, client=client, config=config)
        if i_vecs is not None:
            embeddings["ideal_answers"] = i_vecs

    # 4. Key points embeddings
    key_points = knowledge.get("key_points", [])
    if key_points:
        kp_vecs = embed_texts(key_points, client=client, config=config)
        if kp_vecs is not None:
            embeddings["key_points"] = kp_vecs

    knowledge["embeddings"] = embeddings
    return knowledge


def store_study_knowledge(knowledge):
    """Store study knowledge in-memory and persist to disk."""
    if not knowledge or not isinstance(knowledge, dict):
        return
    topic = knowledge.get("topic", "")
    key = _normalize_topic(topic)
    if not key:
        return

    with _STORE_LOCK:
        _IN_MEMORY_STORE[key] = knowledge

    # Persist to disk without raw numpy arrays (convert to list for JSON serialization)
    try:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", key)[:50]
        filepath = os.path.join(_KNOWLEDGE_DIR, f"{slug}.json")
        
        disk_copy = dict(knowledge)
        disk_embeddings = {}
        for ek, ev in knowledge.get("embeddings", {}).items():
            if isinstance(ev, np.ndarray):
                disk_embeddings[ek] = ev.tolist()
            elif isinstance(ev, list):
                disk_embeddings[ek] = [v.tolist() if isinstance(v, np.ndarray) else v for v in ev]
            else:
                disk_embeddings[ek] = ev
        disk_copy["embeddings"] = disk_embeddings

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(disk_copy, f, indent=2)
    except Exception:
        pass


def get_json_safe_knowledge(knowledge):
    """Return a JSON-serializable version of knowledge by converting ndarrays to lists."""
    if not knowledge or not isinstance(knowledge, dict):
        return {}
    safe_copy = dict(knowledge)
    safe_embeddings = {}
    for ek, ev in knowledge.get("embeddings", {}).items():
        if isinstance(ev, np.ndarray):
            safe_embeddings[ek] = ev.tolist()
        elif isinstance(ev, list):
            safe_embeddings[ek] = [v.tolist() if isinstance(v, np.ndarray) else v for v in ev]
        else:
            safe_embeddings[ek] = ev
    safe_copy["embeddings"] = safe_embeddings
    return safe_copy


def get_study_knowledge(topic):
    """Retrieve study knowledge from in-memory cache or disk."""
    key = _normalize_topic(topic)
    if not key:
        return None

    with _STORE_LOCK:
        if key in _IN_MEMORY_STORE:
            return _IN_MEMORY_STORE[key]

    # Try loading from disk
    try:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", key)[:50]
        filepath = os.path.join(_KNOWLEDGE_DIR, f"{slug}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Reconstruct numpy arrays for embeddings
            emb_dict = {}
            for ek, ev in data.get("embeddings", {}).items():
                if isinstance(ev, list):
                    emb_dict[ek] = np.asarray(ev, dtype=float)
                else:
                    emb_dict[ek] = ev
            data["embeddings"] = emb_dict
            with _STORE_LOCK:
                _IN_MEMORY_STORE[key] = data
            return data
    except Exception:
        pass

    return None


def match_stored_misconceptions(student_answer, knowledge, student_emb=None,
                                client=None, config=None, threshold=0.55):
    """Match student answer against stored misconceptions using local vector cosine similarity."""
    if not knowledge or not student_answer:
        return []

    misconceptions = knowledge.get("misconceptions", [])
    if not misconceptions:
        return []

    m_vecs = knowledge.get("embeddings", {}).get("misconceptions")
    if m_vecs is None or len(m_vecs) != len(misconceptions):
        # Generate on-demand if missing
        m_texts = [f"{m.get('statement', '')}. {m.get('explanation', '')}".strip() for m in misconceptions]
        m_vecs = embed_texts(m_texts, client=client, config=config)
        if m_vecs is not None:
            if "embeddings" not in knowledge:
                knowledge["embeddings"] = {}
            knowledge["embeddings"]["misconceptions"] = m_vecs

    if m_vecs is None:
        return []

    if student_emb is None:
        s_vecs = embed_texts([student_answer], client=client, config=config)
        if s_vecs is not None and len(s_vecs) > 0:
            student_emb = s_vecs[0]

    if student_emb is None:
        return []

    matched = []
    for i, m in enumerate(misconceptions):
        sim = cosine_similarity(student_emb, m_vecs[i])
        if sim >= threshold:
            m_copy = dict(m)
            m_copy["match_similarity"] = round(float(sim), 3)
            matched.append(m_copy)

    # Sort descending by similarity
    matched.sort(key=lambda x: x.get("match_similarity", 0), reverse=True)
    return matched
