# In-House RAG System

The current tested stack uses:

- **LLM / Vision model:** `Qwen/Qwen3.5-4B` served with vLLM
- **Embedding model:** `BAAI/bge-m3`
- **Reranker model:** `BAAI/bge-reranker-v2-m3`
- **PDF parser:** Docling
- **Vector database:** ChromaDB
- **Sparse retrieval:** BM25
- **Backend API:** FastAPI
- **Frontend:** Streamlit

---

## 1. System Architecture

The runtime architecture is service-based.

```text
PDF files
   ↓
Docling Document Parser
   ↓
Vision captioning through Qwen/Qwen3.5-4B
   ↓
Semantic / hierarchical chunking
   ↓
BGE-M3 embedding service
   ↓
ChromaDB vector index + BM25 cache
   ↓
Retriever + reranker
   ↓
Qwen/Qwen3.5-4B answer generation
   ↓
FastAPI /ask endpoint
```

Default ports:

| Service | Port | Endpoint |
|---|---:|---|
| Qwen vLLM server | `8000` | `/v1/chat/completions` |
| Embedding service | `8081` | `/embed` |
| Reranker service | `8082` | `/rerank` |
| FastAPI backend | `8050` | `/ask` |
| Streamlit app | `8501` | UI |

---

## 2. Repository Structure

```text
.
├── api.py
├── app.py
├── pipeline_orchestrator.py
├── document_parser.py
├── semantic_splitter.py
├── vector_indexer.py
├── vision_processor.py
├── config/
│   ├── settings.yaml
│   ├── prompts.yaml
│   ├── secrets.yaml
│   ├── ste100_core_rules.json
│   └── ste100_rules.json
├── data/
│   ├── *.pdf
│   └── assets/
├── services/
│   ├── embed_service.py
│   └── rerank_service.py
├── scripts/
│   ├── reset_index.py
│   ├── start_qwen_vllm.sh
│   └── start_services.sh
└── src/
    ├── database.py
    ├── llm_manager.py
    ├── rag_engine.py
    ├── ste100_guard.py
    └── utils.py
```

---

## 3. Environment Setup

Create and activate a Python environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

For GPU model serving, install vLLM separately if it is not already installed:

```bash
pip install vllm
```

On Colab, Kaggle, or custom GPU servers, vLLM installation can depend on the CUDA and PyTorch versions of the environment.

---

## 4. Configuration

Main runtime settings are in:

```text
config/settings.yaml
```

Important fields:

```yaml
vector_db:
  persist_path: "./chroma_db"
  distance_metric: "cosine"
  collection_name: "doc_store"
  bm25_cache_path: "data/bm25_cache.pkl"

models:
  vllm_api_url: "http://127.0.0.1:8000"
  vision_model_name: "Qwen/Qwen3.5-4B"
  tei_embed_url: "http://127.0.0.1:8081"
  tei_rerank_url: "http://127.0.0.1:8082"

document_parser:
  accelerator_device: "cpu"
  accelerator_threads: 4
  generate_picture_images: true
```

### Docling Accelerator

By default, the parser should use CPU:

```yaml
document_parser:
  accelerator_device: "cpu"
```

This avoids CUDA/NVRTC errors in cloud notebook environments. If the local CUDA setup is stable, this can be changed to:

```yaml
document_parser:
  accelerator_device: "cuda"
```

---

## 5. Starting Model Services

The system expects three external model services before ingestion and question answering:

```text
Qwen vLLM server -> 8000
Embedding service -> 8081
Reranker service -> 8082
```

### 5.1 Start Qwen vLLM

```bash
bash scripts/start_qwen_vllm.sh
```

Default model:

```text
Qwen/Qwen3.5-4B
```

Health check:

```bash
curl http://127.0.0.1:8000/v1/models
```

A simple chat test:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-4B",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Say OK only."}
        ]
      }
    ],
    "max_tokens": 32,
    "temperature": 0.0,
    "chat_template_kwargs": {
      "enable_thinking": false
    }
  }'
```

Expected result:

```text
content: "OK"
```

### 5.2 Start Embedding and Reranker Services

```bash
bash scripts/start_services.sh
```

Health checks:

```bash
curl http://127.0.0.1:8081/health
curl http://127.0.0.1:8082/health
```

Embedding test:

```bash
curl http://127.0.0.1:8081/embed \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": ["hydraulic system test"],
    "normalize": true
  }'
```

Reranker test:

```bash
curl http://127.0.0.1:8082/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      ["what is lidar", "this document explains lidar sensor specifications"],
      ["what is lidar", "this page is about cooking"]
    ]
  }'
```

---

## 6. Resetting the Index

Whenever the embedding model changes, the ChromaDB index must be reset.

Example:

```text
bge-small-en-v1.5 -> 384 dimensions
bge-m3 -> 1024 dimensions
```

If the old ChromaDB index remains, ChromaDB can raise this error:

```text
Collection expecting embedding with dimension of 384, got 1024
```

Reset the index before a fresh ingestion:

```bash
python scripts/reset_index.py
```

This removes:

```text
chroma_db
data/bm25_cache.pkl
data/vision_cache.json
data/ingest_checkpoint.json
data/assets
```

It does not remove PDF files.

---

## 7. Adding Documents

Place PDF files under:

```text
data/
```

Example:

```text
data/test.pdf
data/hardware-user-manual.pdf
```

For a controlled first test, keep only one PDF active in `data/`.

---

## 8. Running Ingestion

Before ingestion, make sure these services are running:

```text
Qwen vLLM server -> 8000
Embedding service -> 8081
Reranker service -> 8082
```

Then run:

```bash
python pipeline_orchestrator.py
```

Successful ingestion should include logs similar to:

```text
Docling DocumentParser baslatiliyor...
Accelerator device: 'cpu'
Dokuman basariyla yapilandirildi ve gorseller cikarildi.
Toplam N adet izole edilmis parca olusturuldu.
N adet metin parcasi vektorlestiriliyor...
Batch basariyla ChromaDB'ye kaydedildi.
BM25 indeksi basariyla kaydedildi.
Tum islemler tamamlandi.
```

---

## 9. Verifying ChromaDB Dimension

After ingestion, verify that the active embedding dimension matches the ChromaDB collection.

```python
import requests
from src.database import DatabaseManager

q_vec = requests.post(
    "http://127.0.0.1:8081/embed",
    json={"inputs": ["Summarize this document briefly."], "normalize": True},
    timeout=120,
).json()

print("Query embedding dimension:", len(q_vec[0]))

db = DatabaseManager()
col = db.get_collection("doc_store")

print("Collection count:", col.count())

res = col.query(
    query_embeddings=q_vec,
    n_results=1,
)

print("Chroma query OK")
print(res["documents"][0][0][:500])
```

Expected output for `BAAI/bge-m3`:

```text
Query embedding dimension: 1024
Collection count: > 0
Chroma query OK
```

---

## 10. Starting the FastAPI Backend

After successful ingestion:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8050
```

API docs:

```text
http://127.0.0.1:8050/docs
```

Test the `/ask` endpoint:

```bash
curl http://127.0.0.1:8050/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize this document briefly.",
    "collection_name": "doc_store",
    "history": [],
    "use_ste100": false,
    "strict_mode": false,
    "template_type": "General"
  }'
```

Expected result:

```text
Status: 200
final_text: generated answer
context_text: retrieved document fragments
```

---

## 11. Starting the Streamlit App

After the backend is running:

```bash
streamlit run app.py
```

Default Streamlit URL:

```text
http://127.0.0.1:8501
```

---

## 12. Full Local Run Order

Use this order for a clean local run:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Start Qwen vLLM
bash scripts/start_qwen_vllm.sh

# 3. In another terminal, start embedding and reranker services
bash scripts/start_services.sh

# 4. Reset old indexes
python scripts/reset_index.py

# 5. Run ingestion
python pipeline_orchestrator.py

# 6. Start backend API
python -m uvicorn api:app --host 0.0.0.0 --port 8050

# 7. Start UI
streamlit run app.py
```

---

## 13. Cloud Notebook Run Order

For Colab or Kaggle:

1. Clone the repository.
2. Install dependencies.
3. Start Qwen vLLM on port `8000`.
4. Start embedding service on port `8081`.
5. Start reranker service on port `8082`.
6. Reset ChromaDB and cache files.
7. Run ingestion.
8. Start FastAPI backend.
9. Test `/ask`.

Recommended notebook order:

```bash
git clone <repo-url>
cd rag-v1

pip install -r requirements.txt
python -m spacy download en_core_web_sm

bash scripts/start_qwen_vllm.sh
bash scripts/start_services.sh

python scripts/reset_index.py
python pipeline_orchestrator.py

python -m uvicorn api:app --host 0.0.0.0 --port 8050
```

In notebooks, long-running services are usually started through `subprocess.Popen` instead of blocking shell commands.

---

## 14. Common Issues

### 14.1 ChromaDB Dimension Mismatch

Error:

```text
Collection expecting embedding with dimension of 384, got 1024
```

Cause:

```text
Old ChromaDB index was created with another embedding model.
```

Fix:

```bash
python scripts/reset_index.py
python pipeline_orchestrator.py
```

### 14.2 Qwen Returns `content: null`

Cause:

```text
Qwen thinking mode is enabled and final answer is not emitted into content.
```

Fix:

The request payload must include:

```json
"chat_template_kwargs": {
  "enable_thinking": false
}
```

This is handled in `src/llm_manager.py`.

### 14.3 Docling CUDA / NVRTC Error

Error:

```text
nvrtc: error: failed to open libnvrtc-builtins.so
```

Cause:

```text
Docling accelerator selected CUDA in an incompatible environment.
```

Fix:

Use CPU parser settings:

```yaml
document_parser:
  accelerator_device: "cpu"
  accelerator_threads: 4
  generate_picture_images: true
```

### 14.4 Missing `fitz`

Error:

```text
ModuleNotFoundError: No module named 'fitz'
```

Fix:

```bash
pip install PyMuPDF
```

### 14.5 API Fails on Startup

The backend checks required index files before starting. If these are missing:

```text
chroma_db
data/vision_cache.json
data/ingest_checkpoint.json
data/bm25_cache.pkl
data/assets
```

Run ingestion first:

```bash
python pipeline_orchestrator.py
```

---

## 15. Notes for Production Hardening

Recommended next improvements:

- Add Docker Compose for all services.
- Add a single process manager or service launcher.
- Add environment-specific config files.
- Add health check script for all endpoints.
- Add regression tests for ingestion and `/ask`.
- Add index metadata validation for embedding model name and dimension.
- Add prompt cleanup to prevent reasoning tags from entering ChromaDB context.
- Add structured logs for retrieval, reranking, and generation latency.
