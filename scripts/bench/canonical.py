"""canonical.py — order-insensitive normalization for exact task scoring.

Prerequisites: none.
Outputs:       none; pure functions used by score.py.

Contract: every list in an answer is treated as a SET. Answers are therefore
    designed so that order never carries meaning, which is what lets one
    normalizer serve all families instead of one per schema. Ordered sequences
    that DO carry meaning — a path, a ranking — are wrapped in
    {"ordered": [...]} and preserved verbatim.

Why: reviewed tasks answer with identifiers, literals, counts and nested
    groups. A single recursive canonicalization avoids scorer behaviour that
    drifts by task family.
"""

from __future__ import annotations

import json
import re

QID = re.compile(r"^[Qq]\d+$")
# A year asked "as a JSON string" reaches the scorer as a number when the model
# passes FINAL("1935"): serialize_final sends the text as is, and json reads
# 1935. Both sides fold a whole-integer string to the integer, so the year
# compares equal however it was written.
INTEGER = re.compile(r"^-?\d+$")
PID = re.compile(r"^[Pp]\d+$")

# Typographic variants of the same character. Wikidata stores the curly forms;
# a model reproducing a description faithfully often emits the straight ones,
# and the two would otherwise score as different strings:
#   stored  'This book explores over a century of Germany’s relations...'
#   model   "This book explores over a century of Germany's relations..."
# Folding is limited to punctuation that carries no meaning. Accents and case
# are NOT folded: in six languages those distinguish real values.
TYPOGRAPHY = str.maketrans({
    "’": "'", "‘": "'", "ʼ": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ", " ": " ", " ": " ",
})


class InvalidAnswer(ValueError):
    """The final answer could not be parsed or normalized."""


def canonical(value):
    """Recursively canonicalize: dicts by key, lists as sorted sets."""
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        items = [canonical(v) for v in value]
        return sorted(items, key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, str):
        text = value.strip()
        if QID.match(text) or PID.match(text):
            return text[0].upper() + text[1:]
        if INTEGER.match(text):
            return int(text)
        return " ".join(text.translate(TYPOGRAPHY).split())
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def canonical_ordered(value):
    """Same, but lists keep their order — for paths and rankings."""
    if isinstance(value, dict):
        if set(value) == {"ordered"}:
            return {"ordered": [canonical_ordered(v) for v in value["ordered"]]}
        return {str(k): canonical_ordered(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return sorted((canonical_ordered(v) for v in value),
                      key=lambda v: json.dumps(v, sort_keys=True))
    return canonical(value)


def parse_answer(text: str):
    """Read the model's FINAL payload, tolerating a fenced code block."""
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.DOTALL)
    try:
        return json.loads(body)
    except json.JSONDecodeError as error:
        raise InvalidAnswer(f"final answer is not valid JSON: {error.msg}") \
            from error


def canonical_json(value) -> str:
    return json.dumps(canonical_ordered(value), sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"))
