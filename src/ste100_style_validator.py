import re
from typing import Any, Dict, List, Optional


def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)?", text))


def split_sentences(text: str) -> List[str]:
    candidates = re.split(r"(?<=[.!?])\s+|\n+", str(text or ""))

    return [
        candidate.strip()
        for candidate in candidates
        if candidate.strip()
    ]


def evaluate_sentence_lengths(
    answer: str,
    max_words: int,
) -> Dict[str, Any]:
    sentences = split_sentences(answer)
    violations = []

    for sentence in sentences:
        word_count = count_words(sentence)

        if word_count > max_words:
            violations.append(
                {
                    "sentence": sentence,
                    "word_count": word_count,
                }
            )

    return {
        "passed": len(violations) == 0,
        "violations": violations,
    }


def is_imperative_like_line(line: str) -> bool:
    stripped = str(line or "").strip()

    if not stripped:
        return False

    numbered_or_bulleted = re.match(
        r"^\s*(?:\d+[\).\s-]+|[A-Z][\).\s-]+|[-*•]\s+)",
        stripped,
        flags=re.IGNORECASE,
    )

    if numbered_or_bulleted:
        return True

    words = re.findall(r"[A-Za-z][A-Za-z-]*", stripped)

    if not words:
        return False

    first_word = words[0].lower()

    imperative_starters = {
        "adjust",
        "apply",
        "attach",
        "calibrate",
        "check",
        "clean",
        "close",
        "connect",
        "disconnect",
        "do",
        "examine",
        "install",
        "make",
        "measure",
        "move",
        "open",
        "perform",
        "press",
        "remove",
        "replace",
        "set",
        "start",
        "stop",
        "turn",
        "use",
        "verify",
        "write",
    }

    return first_word in imperative_starters


def count_instruction_lines(answer: str) -> int:
    count = 0

    for line in str(answer or "").splitlines():
        if is_imperative_like_line(line):
            count += 1

    return count


def has_safety_marker(answer: str) -> bool:
    normalized = str(answer or "").lower()
    markers = ["warning", "caution", "danger"]

    return any(marker in normalized for marker in markers)

INTERNAL_OUTPUT_MARKERS = [
    "i must ignore",
    "image provided in the context",
    "the image provided in the context",
    "provided in the context",
    "conflicts with the text source",
    "primary authority",
    "text source is the primary authority",
    "i need to",
    "i will",
    "key information to include",
    "xml structure",
    "` tags",
]

INTERNAL_OUTPUT_LINE_PATTERNS = [
    r"^\s*tags\.\s*$",
    r"^\s*plan\s*:\s*$",
]


def detect_internal_output_markers(text: Any) -> List[str]:
    raw_text = str(text or "")
    normalized = raw_text.lower()
    violations = []

    for marker in INTERNAL_OUTPUT_MARKERS:
        if marker in normalized:
            violations.append(marker)

    for line in raw_text.splitlines():
        for pattern in INTERNAL_OUTPUT_LINE_PATTERNS:
            if re.match(pattern, line.strip(), flags=re.IGNORECASE):
                violations.append(pattern)

    return sorted(set(violations))


def get_default_style_limits(
    template_type: str,
    min_instruction_lines: Optional[int] = None,
    max_words_per_sentence: Optional[int] = None,
) -> Dict[str, int]:
    template = str(template_type or "").lower()

    if template == "procedure":
        default_max_words = 20
        default_min_instruction_lines = 1
    elif template == "descriptive":
        default_max_words = 25
        default_min_instruction_lines = 0
    elif template == "safety":
        default_max_words = 25
        default_min_instruction_lines = 0
    else:
        default_max_words = 25
        default_min_instruction_lines = 0

    return {
        "max_words_per_sentence": int(
            max_words_per_sentence or default_max_words
        ),
        "min_instruction_lines": int(
            default_min_instruction_lines
            if min_instruction_lines is None
            else min_instruction_lines
        ),
    }


def validate_ste100_style(
    answer: str,
    template_type: str,
    min_instruction_lines: Optional[int] = None,
    max_words_per_sentence: Optional[int] = None,
    require_safety_marker: bool = False,
    require_output_hygiene: bool = True,
) -> Dict[str, Any]:
    limits = get_default_style_limits(
        template_type=template_type,
        min_instruction_lines=min_instruction_lines,
        max_words_per_sentence=max_words_per_sentence,
    )

    sentence_result = evaluate_sentence_lengths(
        answer=answer,
        max_words=limits["max_words_per_sentence"],
    )

    instruction_line_count = count_instruction_lines(answer)
    template = str(template_type or "").lower()

    if template == "procedure":
        instruction_like = (
            instruction_line_count >= limits["min_instruction_lines"]
        )
    else:
        instruction_like = True

    if template == "safety" and require_safety_marker:
        safety_marker_ok = has_safety_marker(answer)
    else:
        safety_marker_ok = True

    feedback = []

    if require_output_hygiene:
        output_hygiene_violations = detect_internal_output_markers(answer)
    else:
        output_hygiene_violations = []

    output_hygiene_ok = len(output_hygiene_violations) == 0

    if not sentence_result["passed"]:
        feedback.append(
            "Write short sentences. Use a maximum of "
            f"{limits['max_words_per_sentence']} words in each sentence."
        )

    if not instruction_like:
        feedback.append(
            "Write the procedure as imperative instruction lines. "
            "Start each step with a command verb, for example CHECK, "
            "CONNECT, REMOVE, SET, or VERIFY."
        )

    if not safety_marker_ok:
        feedback.append(
            "Start the safety text with an applicable risk word: "
            "WARNING, CAUTION, or DANGER."
        )

    if not output_hygiene_ok:
        feedback.append(
            "Remove internal planning, prompt fragments, context analysis, "
            "and non-answer text from the final output."
        )

    passed = (
        sentence_result["passed"]
        and instruction_like
        and safety_marker_ok
        and output_hygiene_ok
    )

    return {
        "passed": passed,
        "sentence_length_ok": sentence_result["passed"],
        "sentence_violations": sentence_result["violations"],
        "instruction_line_count": instruction_line_count,
        "instruction_like": instruction_like,
        "safety_marker_ok": safety_marker_ok,
        "feedback": feedback,
        "output_hygiene_ok": output_hygiene_ok,
        "output_hygiene_violations": output_hygiene_violations,
    }