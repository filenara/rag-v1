import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="STE100 RAG Backend API", version="1.0")
engine = None


class QueryRequest(BaseModel):
    query: str
    collection_name: str
    history: List[Dict[str, Any]] = []
    use_ste100: bool = False
    strict_mode: bool = False
    template_type: str = "General"


@app.on_event("startup")
def startup_event() -> None:
    global engine
    logger.info("FastAPI: RAG Engine baslatiliyor...")
    engine = RAGEngine()


@app.post("/ask")
def ask_question(req: QueryRequest) -> Dict[str, Any]:
    if not engine:
        raise HTTPException(status_code=500, detail="Motor henüz baslatilamadi.")
        
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