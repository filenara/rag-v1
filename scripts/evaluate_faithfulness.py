import argparse
import json
import logging
import os
from typing import Any, Dict, List

from src.rag_engine import RAGEngine
from src.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

NOT_FOUND_MARKERS = [
    "information not found",
    "not found in provided documents",
    "not found in provided fragment",
    "ilgili bilgi",
    "bulunamadi",
    "bulunamadı",
]


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    cases = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line {line_number}: {exc}"
                ) from exc

            cases.append(item)

    return cases


def normalize_text(text: str) -> str:
    return str(text).lower().strip()


def normalize_source_name(source: str) -> str:
    return os.path.basename(str(source)).lower().strip()


def is_not_found_answer(answer: str) -> bool:
    normalized_answer = normalize_text(answer)

    return any(
        marker in normalized_answer
        for marker in NOT_FOUND_MARKERS
    )


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


def evaluate_must_include(
    answer: str,
    must_include: List[str],
) -> float:
    if not must_include:
        return 1.0

    normalized_answer = normalize_text(answer)
    hit_count = 0

    for phrase in must_include:
        if normalize_text(phrase) in normalized_answer:
            hit_count += 1

    return hit_count / len(must_include)


def evaluate_must_not_include(
    answer: str,
    must_not_include: List[str],
) -> bool:
    normalized_answer = normalize_text(answer)

    for phrase in must_not_include:
        if normalize_text(phrase) in normalized_answer:
            return False

    return True


def evaluate_case(
    engine: RAGEngine,
    case: Dict[str, Any],
    collection_name: str,
    min_include_hit_rate: float,
) -> Dict[str, Any]:
    query = case["query"]
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
        use_ste100=False,
        strict_mode=False,
        template_type="General",
    )

    not_found = is_not_found_answer(final_text)
    source_hit = evaluate_source_hit(
        returned_sources=sources,
        expected_sources=expected_sources,
    )
    include_hit_rate = evaluate_must_include(
        answer=final_text,
        must_include=must_include,
    )
    no_forbidden_phrase = evaluate_must_not_include(
        answer=final_text,
        must_not_include=must_not_include,
    )

    if expected_not_found:
        passed = not_found and no_forbidden_phrase
    else:
        passed = (
            not not_found
            and source_hit
            and include_hit_rate >= min_include_hit_rate
            and no_forbidden_phrase
        )

    return {
        "query": query,
        "passed": passed,
        "expected_not_found": expected_not_found,
        "not_found": not_found,
        "source_hit": source_hit,
        "include_hit_rate": include_hit_rate,
        "no_forbidden_phrase": no_forbidden_phrase,
        "final_text": final_text,
        "context_length": len(context_text or ""),
        "is_compliant": is_compliant,
        "was_corrected": was_corrected,
        "feedback_report": feedback_report,
        "returned_sources": sources,
    }


def print_case_result(index: int, result: Dict[str, Any]) -> None:
    status = "PASS" if result["passed"] else "FAIL"

    print("-" * 80)
    print(f"[{index}] {status}")
    print(f"Query: {result['query']}")
    print(f"Expected not found: {result['expected_not_found']}")
    print(f"Returned not found: {result['not_found']}")
    print(f"Source hit: {result['source_hit']}")
    print(f"Include hit rate: {result['include_hit_rate']:.2f}")
    print(f"No forbidden phrase: {result['no_forbidden_phrase']}")
    print(f"Context length: {result['context_length']}")
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
        print("No faithfulness evaluation cases found.")
        return

    passed_count = sum(1 for result in results if result["passed"])
    source_hits = sum(1 for result in results if result["source_hit"])
    not_found_cases = [
        result for result in results
        if result["expected_not_found"]
    ]
    not_found_passes = sum(
        1 for result in not_found_cases
        if result["passed"]
    )
    avg_include_hit_rate = sum(
        result["include_hit_rate"] for result in results
    ) / total
    forbidden_passes = sum(
        1 for result in results
        if result["no_forbidden_phrase"]
    )

    print("=" * 80)
    print("Faithfulness Evaluation Summary")
    print("=" * 80)
    print(f"Total cases: {total}")
    print(f"Passed cases: {passed_count}/{total}")
    print(f"Source hit rate: {source_hits / total:.2f}")
    print(f"Average include hit rate: {avg_include_hit_rate:.2f}")
    print(f"Forbidden phrase pass rate: {forbidden_passes / total:.2f}")

    if not_found_cases:
        print(
            "Expected-not-found pass rate: "
            f"{not_found_passes}/{len(not_found_cases)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate answer faithfulness for the local RAG system."
    )

    parser.add_argument(
        "--eval-file",
        default="eval/faithfulness_eval_set.jsonl",
        help="Path to the JSONL faithfulness evaluation set.",
    )

    parser.add_argument(
        "--collection-name",
        default="",
        help="Chroma collection name. Defaults to config vector_db.collection_name.",
    )

    parser.add_argument(
        "--min-include-hit-rate",
        type=float,
        default=0.5,
        help="Minimum must_include hit rate required for a case to pass.",
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

    for index, case in enumerate(cases, start=1):
        result = evaluate_case(
            engine=engine,
            case=case,
            collection_name=collection_name,
            min_include_hit_rate=args.min_include_hit_rate,
        )
        results.append(result)
        print_case_result(index, result)

    summarize_results(results)


if __name__ == "__main__":
    main()