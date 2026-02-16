
from fastapi import APIRouter
from services.rag_engine import run_rag_pipeline
from services.nl2sql_engine import generate_sql
from services.cache import cache_response

router = APIRouter()

@router.post("/rag")
def rag_query(query: str):
    cached = cache_response(query)
    if cached:
        return {"cached": True, "result": cached}
    return run_rag_pipeline(query)

@router.post("/nl2sql")
def nl2sql(query: str):
    return {"sql": generate_sql(query)}
