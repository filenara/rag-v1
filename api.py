import os
import logging
import secrets
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel

from src.rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

engine = None
API_TOKEN_ENV = "RAG_API_TOKEN"

REQUIRED_PATHS = [
    "chroma_db",
    "data/vision_cache.json",
    "data/ingest_checkpoint.json",
    "data/bm25_cache.pkl",
    "data/assets"
]

def verify_api_token(authorization: str = Header(default="")) -> None:
    expected_token = os.environ.get(API_TOKEN_ENV, "").strip()

    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication token is not configured.",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    supplied_token = authorization.removeprefix("Bearer ").strip()

    if not secrets.compare_digest(supplied_token, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine

    if not os.environ.get(API_TOKEN_ENV, "").strip():
        error_msg = (
            f"{API_TOKEN_ENV} environment variable is not configured. "
            "FastAPI backend cannot start without API authentication."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    missing_paths = [path for path in REQUIRED_PATHS if not os.path.exists(path)]
    
    if missing_paths:
        error_msg = (
            f"Gerekli veri dosyalari veya klasorleri eksik: {', '.join(missing_paths)}. "
            "Lutfen once ingest islemini calistirin."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info("Veri butunlugu dogrulandi. FastAPI: RAG Engine baslatiliyor...")
    engine = RAGEngine()
    yield

app = FastAPI(title="STE100 RAG Backend API", version="1.0", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str
    collection_name: str
    history: List[Dict[str, Any]] = []
    use_ste100: bool = False
    strict_mode: bool = False
    template_type: str = "General"

@app.post("/ask")
def ask_question(
    req: QueryRequest,
    _: None = Depends(verify_api_token),
) -> Dict[str, Any]:
    if not engine:
        raise HTTPException(status_code=500, detail="Motor henuz baslatilamadi.")
        
    try:
        final_text, context_text, is_compliant, was_corrected, feedback_report = engine.search_and_answer(
            query=req.query,
            collection_name=req.collection_name,
            history=req.history,
            use_ste100=req.use_ste100,
            strict_mode=req.strict_mode,
            template_type=req.template_type
        )
        
        return {
            "final_text": final_text,
            "context_text": context_text,
            "is_compliant": is_compliant,
            "was_corrected": was_corrected,
            "feedback_report": feedback_report
        }
    except Exception as e:
        logger.error("API isleme hatasi: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))