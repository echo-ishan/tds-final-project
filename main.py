import subprocess
import sys

# Automatically install missing dependencies on startup
SUBPROCESS_ARGS = [sys.executable, "-m", "pip", "install"]
REQUIRED_PACKAGES = [
    "fastapi",
    "uvicorn[standard]",
    "numpy",
    "httpx",
    "python-dotenv",
    "google-generativeai",
    "Pillow",
    "scikit-learn"
]

subprocess.run(SUBPROCESS_ARGS + REQUIRED_PACKAGES, check=True)


import os
import sys
import time
import json
import asyncio
import base64
import tempfile
import numpy as np
import httpx

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager

# Utility imports (assumes utils package is present)
from utils.auth import verify_api_key      # API key validation
from utils.rate_limiter import RateLimiter # Rate limiting per IP
from utils.logging_config import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Constants and configuration
AIPROXY_URL     = "https://aiproxy.sanand.workers.dev"
EMBEDDINGS_FILE = "knowledge_embeddings.npz"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL       = "gpt-4o-mini"

# Rate limiter (60 requests/minute per client IP)
rate_limiter = RateLimiter(max_requests=60, time_window=60)

# Pydantic models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    image: Optional[str]

class ReferenceLink(BaseModel):
    url: str
    text: str

class APIResponse(BaseModel):
    answer: str
    links: List[ReferenceLink]
    confidence: float
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    embeddings_loaded: bool
    total_chunks: int

# Lifespan: load embeddings once on startup, clear on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
        app.state.embeddings_data = {
            "chunks": data["chunks"],
            "embeddings": data["embeddings"].astype(np.float32),
            "metadata": json.loads(data["metadata"].item())
        }
        logger.info(f"Loaded {len(app.state.embeddings_data['chunks'])} chunks") 
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}") 
        app.state.embeddings_data = None
    yield
    app.state.embeddings_data = None
    logger.info("Cleaned up embeddings") 

# Create FastAPI app with lifespan handler
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

async def get_embedding(text: str) -> List[float]:
    """Generate text embeddings via AI Proxy with retry logic."""
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
    payload = {"model": EMBEDDING_MODEL, "input": text}
    for i in range(3):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(f"{AIPROXY_URL}/openai/v1/embeddings",
                                      headers=headers, json=payload)
                r.raise_for_status()
                return r.json()["data"][0]["embedding"]
        except Exception as e:
            logger.warning(f"Embedding attempt {i+1} failed: {e}") 
            await asyncio.sleep(2**i)
    raise HTTPException(status_code=503, detail="Embedding service unavailable") 

async def analyze_image(image_b64: str) -> str:
    """Use Google Gemini to generate concise image descriptions."""
    try:
        from google import generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        img_data = base64.b64decode(image_b64)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_data)
            path = tmp.name

        try:
            uploaded = model.upload_file(path)
            prompt = ("Provide a concise 1–2 sentence technical summary: "
                      "transcribe code/errors, note key UI elements, omit decoration.")
            resp = model.generate_content([uploaded, prompt])
            return resp.text.strip()
        finally:
            os.unlink(path)
    except Exception as e:
        logger.error(f"Image analysis failed: {e}") 
        return "Image description unavailable" 

async def generate_answer(context: str, question: str) -> str:
    """Generate an LLM answer using GPT-4o-mini via AI Proxy."""
    system_prompt = (
        "You are a data science TA. Answer ONLY from the context. "
        "Be concise, format code with triple backticks, bold key terms. "
        "If insufficient info, reply 'The provided materials do not contain enough information.'"
    )
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        "temperature": 0.1,
        "max_tokens": 800
    }
    for i in range(3):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(f"{AIPROXY_URL}/openai/v1/chat/completions",
                                      headers=headers, json=payload)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"LLM attempt {i+1} failed: {e}") 
            await asyncio.sleep(2**i)
    raise HTTPException(status_code=503, detail="Answer generation failed") 

def extract_references(indices: List[int], sims: np.ndarray) -> List[ReferenceLink]:
    """Extract up to 5 unique reference links from metadata."""
    refs, seen = [], set()
    ed = app.state.embeddings_data
    for idx in indices:
        meta = ed["metadata"][idx]
        url = meta.get("topic_url") or meta.get("original_url")
        snippet = ed["chunks"][idx][:150].replace("\n", " ").strip() + "..."
        if url and url not in seen and sims[idx] >= 0.3:
            seen.add(url)
            refs.append(ReferenceLink(url=url, text=snippet))
        if len(refs) >= 5:
            break
    return refs 

@app.get("/health", response_model=HealthResponse)
async def health():
    """Service health check."""
    ed = app.state.embeddings_data
    loaded = ed is not None
    count  = len(ed["chunks"]) if loaded else 0
    return HealthResponse(status="healthy", embeddings_loaded=loaded, total_chunks=count) 

@app.post("/api/", response_model=APIResponse, dependencies=[Depends(verify_api_key)])
async def answer_question(request: QueryRequest, req: Request):
    """Main Q&A endpoint combining image and text retrieval."""
    start = time.time()
    await rate_limiter.check_rate_limit(req.client.host)

    ed = app.state.embeddings_data
    if not ed:
        raise HTTPException(status_code=503, detail="Embeddings not loaded")

    q = request.question
    if request.image:
        desc = await analyze_image(request.image)
        q += "\n\nImage: " + desc

    emb = await get_embedding(q)
    embs = ed["embeddings"]
    sims = (embs @ emb) / (np.linalg.norm(embs,axis=1)*np.linalg.norm(emb))
    top = np.argsort(sims)[-5:][::-1]
    context = "\n\n---\n\n".join(ed["chunks"][i] for i in top)

    answer = await generate_answer(context, request.question)
    links  = extract_references(top, sims)
    duration = time.time() - start
    conf = float(np.mean(sims[top]))

    logger.info(f"Answered in {duration:.2f}s with confidence {conf:.2f}") 
    return APIResponse(answer=answer, links=links, confidence=conf, processing_time=duration) 

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
