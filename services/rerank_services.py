import os
from typing import List

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI(title="BGE Reranker Service")

MODEL_NAME = os.getenv(
    "RERANK_MODEL_NAME",
    "BAAI/bge-reranker-v2-m3",
)
DEVICE = os.getenv(
    "RERANK_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)
BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "8"))
MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "8192"))


def get_torch_dtype(device: str):
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


print(f"Loading reranker tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

print(f"Loading reranker model: {MODEL_NAME} on {DEVICE}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    torch_dtype=get_torch_dtype(DEVICE),
    trust_remote_code=True,
).to(DEVICE)

model.eval()
print("Reranker model loaded.")


class RerankRequest(BaseModel):
    texts: List[List[str]]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
    }


def get_scores(logits):
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]

    if logits.ndim == 2:
        return logits[:, -1]

    return logits


@app.post("/rerank")
def rerank(req: RerankRequest):
    pairs = req.texts
    all_scores = []

    for start_idx in range(0, len(pairs), BATCH_SIZE):
        batch_pairs = pairs[start_idx:start_idx + BATCH_SIZE]

        queries = []
        docs = []

        for pair in batch_pairs:
            query = pair[0] if len(pair) > 0 else ""
            doc = pair[1] if len(pair) > 1 else ""
            queries.append(query)
            docs.append(doc)

        encoded = tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            scores = get_scores(outputs.logits)

        all_scores.extend(scores.detach().cpu().float().tolist())

    return [{"score": float(score)} for score in all_scores]