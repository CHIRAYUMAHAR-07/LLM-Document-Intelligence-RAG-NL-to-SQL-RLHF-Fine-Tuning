
from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="Distributed LLM Data Platform")
app.include_router(router)
