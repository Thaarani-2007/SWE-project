from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.services import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite schema on startup
    db.init_schema()
    yield


app = FastAPI(
    title="Codebase-Aware RAG API",
    description="AST, NetworkX Code Graph, and Hybrid Retrieval RAG backend with local LLM",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "Codebase-Aware RAG API",
        "status": "running",
        "frontend_url": "http://localhost:5173",
        "interactive_docs": "/docs",
        "health_check": "/health",
        "message": "Open http://localhost:5173 in your browser to access the full UI."
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "code-rag-backend"}
