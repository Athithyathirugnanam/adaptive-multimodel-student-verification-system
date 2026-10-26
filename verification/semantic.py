"""Semantic similarity between the expected conceptual representation and the
student transcript.

Strategy order (configurable):
1. sentence-transformers (Sentence-BERT style) if the library is installed
2. a remote embedding API through the existing OpenAI-compatible client
3. a deterministic lexical-overlap fallback

The score is cosine similarity normalized to 0-100. Semantic similarity is
NEVER treated as proof of understanding on its own - the returned reason says
so explicitly.
"""

import re

import numpy as np

_STOPWORDS = set(
    "a an and are as at be been being but by can could did do does for from had has have "
    "how i in into is it its may might must not of on or our shall should so some such than "
    "that the their them then there these they this those to was we were what when where "
    "which while who will with would you your".split()
)

_MODEL_CACHE = {}


def _st_model(model_name):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer  # optional dependency
    except ImportError:
        _MODEL_CACHE[model_name] = None
        return None
    model = SentenceTransformer(model_name)
    _MODEL_CACHE[model_name] = model
    return model


def _embed_sentence_transformers(texts, model_name):
    model = _st_model(model_name)
    if model is None:
        return None, "sentence-transformers not installed"
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=float), model_name


def _embed_api(client, api_model, texts):
    if client is None:
        return None, "no API client available"
    try:
        response = client.embeddings.create(model=api_model, input=texts)
        vecs = np.asarray([d.embedding for d in response.data], dtype=float)
        return vecs, api_model
    except Exception as exc:  # network / model availability issues
        return None, str(exc)


def cosine_similarity(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


_cosine = cosine_similarity


def embed_texts(texts, client=None, config=None):
    """Embed a list of text strings using SentenceTransformer (or API/lexical fallback)."""
    if not texts:
        return None
    model_name = "all-MiniLM-L6-v2"
    if config and hasattr(config, "embedding_cfg"):
        model_name = config.embedding_cfg.get("sentence_transformer_model", model_name)
    vecs, _ = _embed_sentence_transformers(texts, model_name)
    if vecs is not None:
        return vecs
    if client is not None and config:
        api_model = config.embedding_cfg.get("api_embedding_model", "nvidia/nv-embedqa-e5-v5")
        vecs, _ = _embed_api(client, api_model, texts)
        if vecs is not None:
            return vecs

    # Deterministic local lexical feature vectors (fallback if model/api unavailable)
    vecs = []
    dim = 256
    for t in texts:
        v = np.zeros(dim, dtype=float)
        toks = _tokens(t)
        for tok in toks:
            idx = abs(hash(tok)) % dim
            v[idx] += 1.0
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        vecs.append(v)
    return np.asarray(vecs, dtype=float)


def _tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def _lexical_similarity(expected, transcript):
    exp_toks, stu_toks = _tokens(expected), _tokens(transcript)
    if not exp_toks or not stu_toks:
        return 0.0
    exp_set, stu_set = set(exp_toks), set(stu_toks)
    overlap = len(exp_set & stu_set)
    precision = overlap / len(stu_set)
    recall = overlap / len(exp_set)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    # Weighted-token variant: rare (topic-specific) words count more.
    freq = {}
    for tok in exp_toks + stu_toks:
        freq[tok] = freq.get(tok, 0) + 1
    weighted_hits = sum(1.0 / freq[t] for t in exp_set & stu_set)
    weighted_total = sum(1.0 / freq[t] for t in exp_set)
    weighted_recall = weighted_hits / weighted_total if weighted_total else 0.0
    return max(0.0, min(1.0, 0.5 * f1 + 0.5 * weighted_recall))


def _band_reason(score):
    if score >= 85:
        return ("The student's explanation closely matches the expected conceptual "
                "meaning of the topic.")
    if score >= 70:
        return ("The student's explanation substantially aligns with the expected "
                "conceptual meaning, with some deviations in emphasis or completeness.")
    if score >= 50:
        return ("The student's explanation partially aligns with the expected conceptual "
                "meaning; several important ideas differ or are absent.")
    return ("The student's explanation diverges considerably from the expected "
            "conceptual meaning.")


def _expected_as_text(expected_representation):
    if isinstance(expected_representation, dict):
        parts = [str(v) for v in expected_representation.values() if isinstance(v, str)]
        for c in expected_representation.get("key_concepts") or []:
            parts.append(str(c) if not isinstance(c, dict)
                         else str(c.get("name", "")))
        return ". ".join(p for p in parts if p)
    return str(expected_representation or "")


def compute_semantic_similarity(expected_representation, transcript,
                                client=None, config=None,
                                expected_embedding=None, student_embedding=None):
    """Return {"score": 0-100, "reason": str} comparing expected vs. explained."""
    expected_text = _expected_as_text(expected_representation)
    transcript = (transcript or "").strip()
    if (not expected_text and expected_embedding is None) or (not transcript and student_embedding is None):
        return {
            "score": 0.0,
            "reason": ("Semantic similarity could not be computed because the expected "
                       "representation or the student transcript was empty."),
        }

    sim, strategy_desc = None, ""

    # Fast path: precomputed embeddings
    if expected_embedding is not None and student_embedding is not None:
        sim = _cosine(expected_embedding, student_embedding)
        strategy_desc = "Precomputed Sentence-BERT embeddings (instant)"
    elif expected_embedding is not None and transcript:
        model_name = config.embedding_cfg.get("sentence_transformer_model", "all-MiniLM-L6-v2") if config else "all-MiniLM-L6-v2"
        s_vecs, desc = _embed_sentence_transformers([transcript], model_name)
        if s_vecs is not None and len(s_vecs) > 0:
            sim = _cosine(expected_embedding, s_vecs[0])
            strategy_desc = "Sentence-BERT-style embeddings (%s)" % desc

    if sim is None:
        order = config.embedding_cfg.get(
            "strategy_order", ["sentence_transformers", "api", "lexical"]) if config else ["sentence_transformers", "lexical"]
        for strategy in order:
            if sim is not None:
                break
            if strategy == "sentence_transformers":
                vecs, desc = _embed_sentence_transformers(
                    [expected_text, transcript],
                    config.embedding_cfg.get("sentence_transformer_model",
                                             "all-MiniLM-L6-v2") if config else "all-MiniLM-L6-v2")
                if vecs is not None:
                    sim = _cosine(vecs[0], vecs[1])
                    strategy_desc = "Sentence-BERT-style embeddings (%s)" % desc
            elif strategy == "api":
                api_model = config.embedding_cfg.get("api_embedding_model",
                                                     "nvidia/nv-embedqa-e5-v5") if config else "nvidia/nv-embedqa-e5-v5"
                vecs, _err = _embed_api(client, api_model, [expected_text, transcript])
                if vecs is not None:
                    sim = _cosine(vecs[0], vecs[1])
                    strategy_desc = "API embedding model (%s)" % api_model
            elif strategy == "lexical" and sim is None:
                sim = _lexical_similarity(expected_text, transcript)
                strategy_desc = ("lexical-overlap fallback "
                                 "(no embedding model was available)")

    if sim is None:
        sim = _lexical_similarity(expected_text, transcript)
        strategy_desc = "lexical-overlap fallback"

    # Map cosine (-1..1, typically 0..1 for text) to 0..100.
    score = round(max(0.0, min(1.0, (sim + 1.0) / 2.0)) * 100.0, 1)

    reason = ("%s Computed via %s. IMPORTANT: this score alone is NOT proof of "
              "understanding; it must be combined with concept coverage, "
              "knowledge-gap and misconception analysis."
              % (_band_reason(score), strategy_desc))
    return {"score": score, "reason": reason}
