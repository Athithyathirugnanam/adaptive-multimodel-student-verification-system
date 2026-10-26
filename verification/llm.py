"""Helpers for calling the LLM and reliably parsing JSON out of its output."""

import json
import re


def extract_json(text):
    """Extract the first JSON object/array found in an LLM response.

    Handles markdown code fences and surrounding prose. Raises ValueError when
    no parseable JSON can be recovered.
    """
    if text is None:
        raise ValueError("LLM returned no content")
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()

    candidates = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        if start == -1:
            continue
        end = cleaned.rfind(closer)
        if end > start:
            candidates.append(cleaned[start:end + 1])
    if not candidates:
        candidates.append(cleaned)

    for candidate in candidates:
        for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                return json.loads(attempt)
            except ValueError:
                continue
    raise ValueError("Could not parse JSON from LLM output")


def call_llm_json(client, model, system_prompt, user_prompt,
                  temperature=0.1, max_tokens=500):
    """Call the chat model (non-streaming) and return parsed JSON."""
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": 0.7,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if "nemotron" in str(model).lower():
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_budget": 0}
    completion = client.chat.completions.create(**kwargs)
    return extract_json(completion.choices[0].message.content)
