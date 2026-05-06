import re
from typing import List


TECHNICAL_TOKEN_PATTERN = re.compile(
    r"""
    [a-zA-Z0-9]+(?:[-_/][a-zA-Z0-9]+)+
    |
    [a-zA-Z]*\d+[a-zA-Z0-9.]*
    |
    \d+(?:\.\d+)?
    |
    [a-zA-Z]+
    """,
    re.VERBOSE,
)


def technical_tokenize(text: str) -> List[str]:
    if not text:
        return []

    normalized_text = text.lower()
    raw_tokens = TECHNICAL_TOKEN_PATTERN.findall(normalized_text)

    tokens = []

    for token in raw_tokens:
        token = token.strip()

        if not token:
            continue

        tokens.append(token)

        split_parts = re.split(r"[-_/]", token)
        for part in split_parts:
            part = part.strip()

            if part and part != token:
                tokens.append(part)

    measurement_tokens = _build_compact_measurement_tokens(raw_tokens)
    tokens.extend(measurement_tokens)

    return _deduplicate_preserve_order(tokens)


def _build_compact_measurement_tokens(tokens: List[str]) -> List[str]:
    compact_tokens = []

    for idx in range(len(tokens) - 1):
        current_token = tokens[idx]
        next_token = tokens[idx + 1]

        if _is_number(current_token) and _is_unit(next_token):
            compact_tokens.append(f"{current_token}{next_token}")

    return compact_tokens


def _is_number(token: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", token))


def _is_unit(token: str) -> bool:
    allowed_units = {
        "mm",
        "cm",
        "m",
        "km",
        "in",
        "inch",
        "ft",
        "kg",
        "g",
        "mg",
        "v",
        "vdc",
        "vac",
        "a",
        "ma",
        "hz",
        "khz",
        "mhz",
        "nm",
        "n",
        "pa",
        "kpa",
        "mpa",
        "bar",
        "psi",
        "deg",
        "rpm",
    }

    return token.lower() in allowed_units


def _deduplicate_preserve_order(tokens: List[str]) -> List[str]:
    seen = set()
    unique_tokens = []

    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)

    return unique_tokens