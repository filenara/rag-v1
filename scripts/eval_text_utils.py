import re
import unicodedata
from typing import Any, Iterable, List, Tuple


def normalize_eval_text(text: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).lower()

    normalized = normalized.replace("ı", "i")
    normalized = normalized.replace("İ", "i")

    normalized = re.sub(r"[•●▪▫◦‣⁃·]", " ", normalized)
    normalized = re.sub(r"[-_/\\]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9.%+#]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def compact_eval_text(text: Any) -> str:
    return re.sub(r"[^a-z0-9.%+#]+", "", normalize_eval_text(text))


def phrase_matches(text: str, phrase: Any) -> bool:
    normalized_text = normalize_eval_text(text)
    normalized_phrase = normalize_eval_text(phrase)

    if not normalized_phrase:
        return True

    if normalized_phrase in normalized_text:
        return True

    compact_text = compact_eval_text(text)
    compact_phrase = compact_eval_text(phrase)

    return bool(compact_phrase and compact_phrase in compact_text)


def normalize_expected_groups(items: Iterable[Any]) -> List[List[str]]:
    groups = []

    for item in items:
        if isinstance(item, dict):
            if "any" in item:
                values = item["any"]
                if isinstance(values, list):
                    groups.append([str(value) for value in values])
                else:
                    groups.append([str(values)])
            elif "all" in item:
                values = item["all"]
                if isinstance(values, list):
                    groups.extend([[str(value)] for value in values])
                else:
                    groups.append([str(values)])
            else:
                groups.append([str(item)])
        elif isinstance(item, list):
            groups.append([str(value) for value in item])
        else:
            groups.append([str(item)])

    return groups


def evaluate_phrase_group_hit_rate(
    text: str,
    expected_items: List[Any],
) -> Tuple[float, List[str], List[List[str]]]:
    groups = normalize_expected_groups(expected_items)

    if not groups:
        return 1.0, [], []

    hit_groups = []
    missed_groups = []

    for group in groups:
        if any(phrase_matches(text, phrase) for phrase in group):
            hit_groups.append(" | ".join(group))
        else:
            missed_groups.append(group)

    hit_rate = len(hit_groups) / len(groups)

    return hit_rate, hit_groups, missed_groups


def evaluate_forbidden_phrases(
    text: str,
    forbidden_items: List[Any],
) -> Tuple[bool, List[str]]:
    groups = normalize_expected_groups(forbidden_items)
    violations = []

    for group in groups:
        for phrase in group:
            if phrase_matches(text, phrase):
                violations.append(phrase)

    return len(violations) == 0, violations

FORBIDDEN_OUTPUT_MARKERS = [
    "<thinking",
    "</thinking",
    "<think",
    "</think",
    "analyze document context",
    "synthesize and resolve conflict",
    "final plan:",
    "the user is asking",
    "i will construct",
    "i will provide",
    "let's re-read",
    "no conversational fillers",
    "ensure the answer is direct",
    "answer directly",
    "omit conversational fillers",
    "[image attached to this message]",
]


def detect_forbidden_output_markers(text: Any) -> List[str]:
    normalized = str(text or "").lower()
    violations = []

    for marker in FORBIDDEN_OUTPUT_MARKERS:
        if marker in normalized:
            violations.append(marker)

    return violations