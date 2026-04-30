import os
from typing import List, Union

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

app = FastAPI(title="BGE-M3 Embedding Service")

MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
DEVICE = os.getenv(
    "EMBED_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
MAX_LENGTH = int(os.getenv("EMBED_MAX_LENGTH", "8192"))
POOLING_MODE = os.getenv("EMBED_POOLING_MODE", "cls").lower()


def get_torch_dtype(device: str):
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


print(f"Loading embedding tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

print(f"Loading embedding model: {MODEL_NAME} on {DEVICE}")
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=get_torch_dtype(DEVICE),
    trust_remote_code=True,
).to(DEVICE)

model.eval()
print("Embedding model loaded.")


class EmbedRequest(BaseModel):
    inputs: Union[str, List[str]]
    normalize: bool = True


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "pooling_mode": POOLING_MODE,
    }


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def pool_embeddings(last_hidden_state, attention_mask):
    if POOLING_MODE == "mean":
        return mean_pooling(last_hidden_state, attention_mask)

    return last_hidden_state[:, 0]


@app.post("/embed")
def embed(req: EmbedRequest):
    texts = req.inputs if isinstance(req.inputs, list) else [req.inputs]
    all_embeddings = []

    for start_idx in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start_idx:start_idx + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            embeddings = pool_embeddings(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )

            if req.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.extend(
            embeddings.detach().cpu().float().tolist()
        )

    return all_embeddings