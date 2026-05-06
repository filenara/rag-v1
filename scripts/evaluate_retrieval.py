import argparse
import json
import logging
import os
from typing import Any, Dict, List

from src.rag_engine import RAGEngine
from src.utils import load_config
from eval_text_utils import evaluate_phrase_group_hit_rate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


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


def normalize_source_name(source: str) -> str:
    return os.path.basename(str(source)).lower().strip()


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


def evaluate_page_hit(
    returned_sources: List[Dict[str, Any]],
    expected_pages: List[int],
    page_tolerance: int,
) -> bool:
    if not expected_pages:
        return True

    returned_pages = []

    for source in returned_sources:
        page = source.get("page")

        try:
            returned_pages.append(int(page))
        except (TypeError, ValueError):
            continue

    for expected_page in expected_pages:
        for returned_page in returned_pages:
            if abs(returned_page - int(expected_page)) <= page_tolerance:
                return True

    return False


def evaluate_keyword_hit_rate(
    context_text: str,
    expected_keywords: List[Any],
) -> Dict[str, Any]:
    hit_rate, hits, misses = evaluate_phrase_group_hit_rate(
        text=context_text,
        expected_items=expected_keywords,
    )

    return {
        "hit_rate": hit_rate,
        "hits": hits,
        "misses": misses,
    }


def evaluate_case(
    engine: RAGEngine,
    case: Dict[str, Any],
    collection_name: str,
    page_tolerance: int,
    min_keyword_hit_rate: float,
) -> Dict[str, Any]:
    query = case["query"]
    expected_sources = case.get("expected_sources", [])
    expected_pages = case.get("expected_pages", [])
    expected_keywords = case.get("expected_keywords", [])

    context_text, sources, _input_image = engine.retrieve_context(
        query=query,
        collection_name=collection_name,
    )

    source_hit = evaluate_source_hit(
        returned_sources=sources,
        expected_sources=expected_sources,
    )
    page_hit = evaluate_page_hit(
        returned_sources=sources,
        expected_pages=expected_pages,
        page_tolerance=page_tolerance,
    )
    keyword_result = evaluate_keyword_hit_rate(
        context_text=context_text,
        expected_keywords=expected_keywords,
    )
    keyword_hit_rate = keyword_result["hit_rate"]

    passed = (
        bool(context_text.strip())
        and source_hit
        and page_hit
        and keyword_hit_rate >= min_keyword_hit_rate
    )

    return {
        "query": query,
        "passed": passed,
        "source_hit": source_hit,
        "page_hit": page_hit,
        "keyword_hit_rate": keyword_hit_rate,
        "returned_sources": sources,
        "keyword_hits": keyword_result["hits"],
        "keyword_misses": keyword_result["misses"],
    }


def print_case_result(index: int, result: Dict[str, Any]) -> None:
    status = "PASS" if result["passed"] else "FAIL"

    print("-" * 80)
    print(f"[{index}] {status}")
    print(f"Query: {result['query']}")
    print(f"Source hit: {result['source_hit']}")
    print(f"Page hit: {result['page_hit']}")
    print(f"Keyword hit rate: {result['keyword_hit_rate']:.2f}")

    if result.get("keyword_hits"):
        print("Keyword hits:")
        for keyword in result["keyword_hits"]:
            print(f"  + {keyword}")

    if result.get("keyword_misses"):
        print("Keyword misses:")
        for group in result["keyword_misses"]:
            print(f"  - {' | '.join(group)}")

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
        print("No evaluation cases found.")
        return

    passed_count = sum(1 for result in results if result["passed"])
    source_hits = sum(1 for result in results if result["source_hit"])
    page_hits = sum(1 for result in results if result["page_hit"])
    avg_keyword_hit_rate = sum(
        result["keyword_hit_rate"] for result in results
    ) / total

    print("=" * 80)
    print("Retrieval Evaluation Summary")
    print("=" * 80)
    print(f"Total cases: {total}")
    print(f"Passed cases: {passed_count}/{total}")
    print(f"Source hit rate: {source_hits / total:.2f}")
    print(f"Page hit rate: {page_hits / total:.2f}")
    print(f"Average keyword hit rate: {avg_keyword_hit_rate:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality for the local RAG index."
    )

    parser.add_argument(
        "--eval-file",
        default="eval/retrieval_eval_set.jsonl",
        help="Path to the JSONL retrieval evaluation set.",
    )

    parser.add_argument(
        "--collection-name",
        default="",
        help="Chroma collection name. Defaults to config vector_db.collection_name.",
    )

    parser.add_argument(
        "--page-tolerance",
        type=int,
        default=1,
        help="Allowed page offset for page hit calculation.",
    )

    parser.add_argument(
        "--min-keyword-hit-rate",
        type=float,
        default=0.5,
        help="Minimum keyword hit rate required for a case to pass.",
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
            page_tolerance=args.page_tolerance,
            min_keyword_hit_rate=args.min_keyword_hit_rate,
        )
        results.append(result)
        print_case_result(index, result)

    summarize_results(results)


if __name__ == "__main__":
    main()