import argparse
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from eval_text_utils import (
    detect_forbidden_output_markers,
    evaluate_forbidden_phrases,
    evaluate_phrase_group_hit_rate,
    normalize_eval_text,
)
from src.rag_engine import RAGEngine
from src.utils import load_config
from src.ste100_style_validator import validate_ste100_style


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


NOT_FOUND_MARKERS = [
    "information not found",
    "not found in provided documents",
    "not found in provided fragment",
]


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    cases = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line {line_number}: {exc}"
                ) from exc

    return cases


def normalize_source_name(source: str) -> str:
    return os.path.basename(str(source)).lower().strip()


def is_not_found_answer(answer: str) -> bool:
    normalized_answer = normalize_eval_text(answer)

    return any(marker in normalized_answer for marker in NOT_FOUND_MARKERS)


def evaluate_source_hit(
    returned_sources: List[Dict[str, Any]],
    expected_sources: List[str],
) -> bool:
    if not expected_sources:
        return True

    returned = {
        normalize_source_name(source.get("source", ""))
        for source in returned_sources
    }
    expected = {
        normalize_source_name(source)
        for source in expected_sources
    }

    return bool(returned.intersection(expected))


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
    stripped = line.strip()

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
    normalized = normalize_eval_text(answer)
    markers = ["warning", "caution", "danger"]

    return any(marker in normalized for marker in markers)


def get_style_limits(
    template_type: str,
    case: Dict[str, Any],
) -> Dict[str, int]:
    template = str(template_type).lower()

    if template == "procedure":
        default_max_words = 20
    elif template == "descriptive":
        default_max_words = 25
    elif template == "safety":
        default_max_words = 25
    else:
        default_max_words = 25

    return {
        "max_words_per_sentence": int(
            case.get("max_words_per_sentence", default_max_words)
        ),
        "min_instruction_lines": int(
            case.get("min_instruction_lines", 0)
        ),
    }


def evaluate_template_style(
    answer: str,
    template_type: str,
    case: Dict[str, Any],
) -> Dict[str, Any]:
    return validate_ste100_style(
        answer=answer,
        template_type=template_type,
        min_instruction_lines=case.get("min_instruction_lines"),
        max_words_per_sentence=case.get("max_words_per_sentence"),
        require_safety_marker=bool(case.get("require_safety_marker", False)),
    )


def evaluate_single_mode(
    engine: RAGEngine,
    case: Dict[str, Any],
    collection_name: str,
    strict_mode: bool,
    min_include_hit_rate: float,
) -> Dict[str, Any]:
    query = case["query"]
    template_type = case.get("template_type", "Procedure")
    expected_sources = case.get("expected_sources", [])
    must_include = case.get("must_include", [])
    must_not_include = case.get("must_not_include", [])
    expected_not_found = bool(case.get("expected_not_found", False))

    (
        final_text,
        context_text,
        is_compliant,
        was_corrected,
        feedback_report,
        sources,
    ) = engine.search_and_answer(
        query=query,
        collection_name=collection_name,
        history=[],
        use_ste100=True,
        strict_mode=strict_mode,
        template_type=template_type,
    )

    not_found = is_not_found_answer(final_text)
    source_hit = evaluate_source_hit(
        returned_sources=sources,
        expected_sources=expected_sources,
    )

    include_hit_rate, include_hits, include_misses = (
        evaluate_phrase_group_hit_rate(
            text=final_text,
            expected_items=must_include,
        )
    )

    no_forbidden_phrase, forbidden_violations = evaluate_forbidden_phrases(
        text=final_text,
        forbidden_items=must_not_include,
    )

    reasoning_violations = detect_forbidden_output_markers(final_text)
    no_reasoning_leak = len(reasoning_violations) == 0

    style_result = evaluate_template_style(
        answer=final_text,
        template_type=template_type,
        case=case,
    )

    if expected_not_found:
        content_passed = not_found
    else:
        content_passed = (
            not not_found
            and source_hit
            and include_hit_rate >= min_include_hit_rate
        )

    hygiene_passed = no_forbidden_phrase and no_reasoning_leak

    if strict_mode:
        passed = (
            content_passed
            and hygiene_passed
            and is_compliant
            and style_result["passed"]
        )
    else:
        passed = content_passed and hygiene_passed

    return {
        "query": query,
        "template_type": template_type,
        "strict_mode": strict_mode,
        "passed": passed,
        "expected_not_found": expected_not_found,
        "not_found": not_found,
        "source_hit": source_hit,
        "include_hit_rate": include_hit_rate,
        "include_hits": include_hits,
        "include_misses": include_misses,
        "no_forbidden_phrase": no_forbidden_phrase,
        "forbidden_violations": forbidden_violations,
        "no_reasoning_leak": no_reasoning_leak,
        "reasoning_violations": reasoning_violations,
        "is_compliant": is_compliant,
        "was_corrected": was_corrected,
        "feedback_report": feedback_report,
        "style_result": style_result,
        "context_length": len(context_text or ""),
        "final_text": final_text,
        "returned_sources": sources,
    }


def print_mode_result(index: int, result: Dict[str, Any]) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    strict_label = "strict=True" if result["strict_mode"] else "strict=False"

    print("-" * 80)
    print(
        f"[{index}] {status} | {result['template_type']} | {strict_label}"
    )
    print(f"Query: {result['query']}")
    print(f"Returned not found: {result['not_found']}")
    print(f"Source hit: {result['source_hit']}")
    print(f"Include hit rate: {result['include_hit_rate']:.2f}")
    print(f"STE100 compliant: {result['is_compliant']}")
    print(f"Was corrected: {result['was_corrected']}")
    print(f"No forbidden phrase: {result['no_forbidden_phrase']}")
    print(f"No reasoning leak: {result['no_reasoning_leak']}")

    style_result = result["style_result"]
    print(f"Style pass: {style_result['passed']}")
    print(f"Sentence length OK: {style_result['sentence_length_ok']}")
    print(f"Instruction-like: {style_result['instruction_like']}")
    print(f"Instruction lines: {style_result['instruction_line_count']}")
    print(f"Safety marker OK: {style_result['safety_marker_ok']}")
    print(
        "Output hygiene OK: "
        f"{style_result.get('output_hygiene_ok', True)}"
    )

    if result.get("include_hits"):
        print("Include hits:")
        for phrase in result["include_hits"]:
            print(f"  + {phrase}")

    if result.get("include_misses"):
        print("Include misses:")
        for group in result["include_misses"]:
            print(f"  - {' | '.join(group)}")

    if result.get("feedback_report"):
        print("STE100 feedback:")
        for item in result["feedback_report"]:
            print(f"  - {item}")

    if style_result.get("sentence_violations"):
        print("Sentence length violations:")
        for item in style_result["sentence_violations"][:5]:
            print(
                f"  - {item['word_count']} words: "
                f"{item['sentence']}"
            )

    if result.get("reasoning_violations"):
        print("Reasoning or prompt leakage markers:")
        for marker in result["reasoning_violations"]:
            print(f"  - {marker}")
    
    if style_result.get("output_hygiene_violations"):
        print("Output hygiene violations:")
        for marker in style_result["output_hygiene_violations"]:
            print(f"  - {marker}")

    print("Final answer:")
    print(result["final_text"])

    if result["returned_sources"]:
        print("Returned sources:")
        for source in result["returned_sources"]:
            source_name = source.get("source", "Unknown")
            page = source.get("page", "?")
            score = source.get("rerank_score", 0.0)
            parent_context = source.get("parent_context", "")

            print(
                f"  - {source_name} | page={page} | "
                f"rerank={score:.4f} | context={parent_context}"
            )
    else:
        print("Returned sources: []")


def summarize_results(results: List[Dict[str, Any]]) -> None:
    total = len(results)

    if total == 0:
        print("No STE100 evaluation results found.")
        return

    passed_count = sum(1 for result in results if result["passed"])
    strict_true_results = [
        result for result in results
        if result["strict_mode"]
    ]
    strict_false_results = [
        result for result in results
        if not result["strict_mode"]
    ]

    def pass_rate(items: List[Dict[str, Any]]) -> str:
        if not items:
            return "N/A"

        passed = sum(1 for item in items if item["passed"])
        return f"{passed}/{len(items)}"

    compliant_count = sum(1 for result in results if result["is_compliant"])
    corrected_count = sum(1 for result in results if result["was_corrected"])
    leak_free_count = sum(
        1 for result in results
        if result["no_reasoning_leak"]
    )

    print("=" * 80)
    print("ASD-STE100 Generation Evaluation Summary")
    print("=" * 80)
    print(f"Total mode-runs: {total}")
    print(f"Passed mode-runs: {passed_count}/{total}")
    print(f"strict=False pass rate: {pass_rate(strict_false_results)}")
    print(f"strict=True pass rate: {pass_rate(strict_true_results)}")
    print(f"STE100 compliant outputs: {compliant_count}/{total}")
    print(f"Corrected outputs: {corrected_count}/{total}")
    print(f"Reasoning/prompt leak-free outputs: {leak_free_count}/{total}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ASD-STE100 generation behavior."
    )

    parser.add_argument(
        "--eval-file",
        default="eval/ste100_eval_set.jsonl",
        help="Path to the STE100 JSONL evaluation set.",
    )

    parser.add_argument(
        "--collection-name",
        default="",
        help="Chroma collection name. Defaults to config.",
    )

    parser.add_argument(
        "--min-include-hit-rate",
        type=float,
        default=0.5,
        help="Minimum must_include hit rate.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    collection_name = args.collection_name or config.get(
        "vector_db",
        {},
    ).get("collection_name", "doc_store")

    cases = load_jsonl(args.eval_file)

    logger.info("RAGEngine baslatiliyor...")
    engine = RAGEngine()

    results = []
    run_index = 1

    for case in cases:
        strict_modes = case.get("strict_modes", [False, True])

        for strict_mode in strict_modes:
            result = evaluate_single_mode(
                engine=engine,
                case=case,
                collection_name=collection_name,
                strict_mode=bool(strict_mode),
                min_include_hit_rate=args.min_include_hit_rate,
            )
            results.append(result)
            print_mode_result(run_index, result)
            run_index += 1

    summarize_results(results)


if __name__ == "__main__":
    main()