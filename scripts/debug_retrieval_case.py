import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from src.rag_engine import RAGEngine
from src.tokenization import technical_tokenize
from src.utils import load_config


def normalize_source(source: str) -> str:
    return os.path.basename(str(source)).lower().strip()


def get_collection_name(args: argparse.Namespace) -> str:
    if args.collection_name:
        return args.collection_name

    config = load_config()
    return config.get("vector_db", {}).get("collection_name", "doc_store")


def get_where_clause(source_filter: str) -> Dict[str, Any] | None:
    if not source_filter:
        return None

    return {
        "source": {
            "$contains": source_filter,
        }
    }


def preview_text(text: str, preview_chars: int) -> str:
    clean_text = " ".join(str(text).split())

    if len(clean_text) <= preview_chars:
        return clean_text

    return clean_text[:preview_chars] + "..."


def format_meta(meta: Dict[str, Any]) -> str:
    source = os.path.basename(str(meta.get("source", "Unknown")))
    page = meta.get("page", "?")
    parent_context = meta.get("parent_context", "")

    return (
        f"source={source} | "
        f"page={page} | "
        f"context={parent_context}"
    )


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def run_vector_search(
    engine: RAGEngine,
    query: str,
    collection_name: str,
    source_filter: str,
    top_n: int,
) -> List[Dict[str, Any]]:
    embedder = engine.llm_manager.load_embedder()
    collection = engine.db.get_collection(collection_name)

    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True,
    )

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=top_n,
        where=get_where_clause(source_filter),
        include=["documents", "metadatas", "distances"],
    )

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    rows = []

    for rank, doc_id in enumerate(ids):
        rows.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "text": documents[rank] if rank < len(documents) else "",
                "meta": metadatas[rank] if rank < len(metadatas) else {},
                "distance": distances[rank] if rank < len(distances) else None,
                "rrf_score": 1 / (engine.k_constant + rank),
            }
        )

    return rows


def run_bm25_search(
    engine: RAGEngine,
    query: str,
    source_filter: str,
    top_n: int,
) -> List[Dict[str, Any]]:
    if not engine.bm25:
        return []

    query_tokens = technical_tokenize(query)
    scores = np.array(engine.bm25.get_scores(query_tokens))

    if source_filter:
        filter_lower = source_filter.lower()
        valid_indices = []

        for source, indices in engine.source_to_indices.items():
            if filter_lower in source:
                valid_indices.extend(indices)

        if valid_indices:
            mask = np.ones(len(scores), dtype=bool)
            mask[valid_indices] = False
            scores[mask] = -np.inf
        else:
            scores[:] = -np.inf

    sorted_indices = np.argsort(scores)[::-1]

    rows = []

    for rank, index in enumerate(sorted_indices[:top_n]):
        score = float(scores[index])

        if not np.isfinite(score):
            continue

        doc_id = engine.ids[index]
        text = engine.doc_text_map.get(doc_id, "")
        meta = engine.doc_meta_map.get(doc_id, {})

        rows.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "text": text,
                "meta": meta,
                "bm25_score": score,
                "rrf_score": 1 / (engine.k_constant + rank),
            }
        )

    return rows


def merge_candidates(
    vector_rows: List[Dict[str, Any]],
    bm25_rows: List[Dict[str, Any]],
    candidate_limit: int,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for row in vector_rows:
        doc_id = row["doc_id"]

        if doc_id not in merged:
            merged[doc_id] = {
                "doc_id": doc_id,
                "text": row["text"],
                "meta": row["meta"],
                "vector_rank": None,
                "vector_distance": None,
                "bm25_rank": None,
                "bm25_score": None,
                "rrf_score": 0.0,
            }

        merged[doc_id]["vector_rank"] = row["rank"]
        merged[doc_id]["vector_distance"] = row["distance"]
        merged[doc_id]["rrf_score"] += row["rrf_score"]

    for row in bm25_rows:
        doc_id = row["doc_id"]

        if doc_id not in merged:
            merged[doc_id] = {
                "doc_id": doc_id,
                "text": row["text"],
                "meta": row["meta"],
                "vector_rank": None,
                "vector_distance": None,
                "bm25_rank": None,
                "bm25_score": None,
                "rrf_score": 0.0,
            }

        merged[doc_id]["bm25_rank"] = row["rank"]
        merged[doc_id]["bm25_score"] = row["bm25_score"]
        merged[doc_id]["rrf_score"] += row["rrf_score"]

    candidates = sorted(
        merged.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return candidates[:candidate_limit]


def rerank_candidates(
    engine: RAGEngine,
    query: str,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    reranker = engine.llm_manager.load_reranker()

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    reranked = []

    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        item["rerank_score"] = float(score)
        reranked.append(item)

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return reranked


def print_vector_rows(
    rows: List[Dict[str, Any]],
    preview_chars: int,
) -> None:
    print_section("VECTOR SEARCH TOP RESULTS")

    if not rows:
        print("No vector results.")
        return

    for row in rows:
        print(f"\n[VECTOR RANK {row['rank']}]")
        print(f"doc_id={row['doc_id']}")
        print(f"distance={row['distance']}")
        print(f"rrf_score={row['rrf_score']:.6f}")
        print(format_meta(row["meta"]))
        print(preview_text(row["text"], preview_chars))


def print_bm25_rows(
    rows: List[Dict[str, Any]],
    preview_chars: int,
) -> None:
    print_section("BM25 TOP RESULTS")

    if not rows:
        print("No BM25 results.")
        return

    for row in rows:
        print(f"\n[BM25 RANK {row['rank']}]")
        print(f"doc_id={row['doc_id']}")
        print(f"bm25_score={row['bm25_score']:.6f}")
        print(f"rrf_score={row['rrf_score']:.6f}")
        print(format_meta(row["meta"]))
        print(preview_text(row["text"], preview_chars))


def print_merged_candidates(
    candidates: List[Dict[str, Any]],
    preview_chars: int,
) -> None:
    print_section("MERGED RRF CANDIDATES")

    if not candidates:
        print("No merged candidates.")
        return

    for rank, candidate in enumerate(candidates):
        print(f"\n[MERGED RANK {rank}]")
        print(f"doc_id={candidate['doc_id']}")
        print(f"rrf_score={candidate['rrf_score']:.6f}")
        print(f"vector_rank={candidate['vector_rank']}")
        print(f"vector_distance={candidate['vector_distance']}")
        print(f"bm25_rank={candidate['bm25_rank']}")
        print(f"bm25_score={candidate['bm25_score']}")
        print(format_meta(candidate["meta"]))
        print(preview_text(candidate["text"], preview_chars))


def print_reranked_candidates(
    reranked: List[Dict[str, Any]],
    min_rerank_score: float,
    preview_chars: int,
) -> None:
    print_section("RERANKED CANDIDATES")

    if not reranked:
        print("No reranked candidates.")
        return

    best_score = reranked[0]["rerank_score"]
    print(f"best_rerank_score={best_score:.6f}")
    print(f"configured_min_rerank_score={min_rerank_score}")
    print(f"passes_threshold={best_score >= min_rerank_score}")

    for rank, candidate in enumerate(reranked):
        print(f"\n[RERANKED RANK {rank}]")
        print(f"doc_id={candidate['doc_id']}")
        print(f"rerank_score={candidate['rerank_score']:.6f}")
        print(f"rrf_score={candidate['rrf_score']:.6f}")
        print(f"vector_rank={candidate['vector_rank']}")
        print(f"bm25_rank={candidate['bm25_rank']}")
        print(format_meta(candidate["meta"]))
        print(preview_text(candidate["text"], preview_chars))


def print_final_retrieve_context(
    engine: RAGEngine,
    query: str,
    collection_name: str,
    source_filter: str,
    preview_chars: int,
) -> None:
    print_section("FINAL retrieve_context() OUTPUT")

    context_text, sources, _input_image = engine.retrieve_context(
        query=query,
        collection_name=collection_name,
        source_filter=source_filter,
    )

    print(f"context_length={len(context_text)}")
    print(f"source_count={len(sources)}")

    if sources:
        print("\nSources:")
        for source in sources:
            print(source)
    else:
        print("\nSources: []")

    print("\nContext preview:")
    print(preview_text(context_text, preview_chars))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug vector, BM25, RRF, and reranker retrieval stages."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Query to debug.",
    )

    parser.add_argument(
        "--collection-name",
        default="",
        help="Chroma collection name. Defaults to config value.",
    )

    parser.add_argument(
        "--source-filter",
        default="",
        help="Optional source filename filter.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of vector and BM25 results to inspect.",
    )

    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=15,
        help="Number of merged RRF candidates to rerank.",
    )

    parser.add_argument(
        "--preview-chars",
        type=int,
        default=800,
        help="Number of text characters to print per candidate.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collection_name = get_collection_name(args)

    print_section("DEBUG CONFIG")
    print(f"query={args.query}")
    print(f"collection_name={collection_name}")
    print(f"source_filter={args.source_filter or None}")
    print(f"top_n={args.top_n}")
    print(f"candidate_limit={args.candidate_limit}")

    engine = RAGEngine()

    print("\nRuntime retrieval config:")
    print(f"n_results={engine.n_results}")
    print(f"k_constant={engine.k_constant}")
    print(f"top_k_rerank={engine.top_k_rerank}")
    print(f"min_rerank_score={engine.min_rerank_score}")

    vector_rows = run_vector_search(
        engine=engine,
        query=args.query,
        collection_name=collection_name,
        source_filter=args.source_filter,
        top_n=args.top_n,
    )

    bm25_rows = run_bm25_search(
        engine=engine,
        query=args.query,
        source_filter=args.source_filter,
        top_n=args.top_n,
    )

    candidates = merge_candidates(
        vector_rows=vector_rows,
        bm25_rows=bm25_rows,
        candidate_limit=args.candidate_limit,
    )

    reranked = rerank_candidates(
        engine=engine,
        query=args.query,
        candidates=candidates,
    )

    print_vector_rows(vector_rows, args.preview_chars)
    print_bm25_rows(bm25_rows, args.preview_chars)
    print_merged_candidates(candidates, args.preview_chars)
    print_reranked_candidates(
        reranked=reranked,
        min_rerank_score=engine.min_rerank_score,
        preview_chars=args.preview_chars,
    )
    print_final_retrieve_context(
        engine=engine,
        query=args.query,
        collection_name=collection_name,
        source_filter=args.source_filter,
        preview_chars=args.preview_chars,
    )


if __name__ == "__main__":
    main()