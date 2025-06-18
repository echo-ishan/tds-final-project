import asyncio
import base64
import json
import numpy as np
import os
import time
import httpx
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from utils.auth import verify_api_key
from utils.rate_limiter import RateLimiter
from utils.logging_config import setup_logging, get_logger
import tempfile
from contextlib import asynccontextmanager

# Initialize logging
setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="TDS Virtual TA API",
    description="RAG-powered Q&A system for data science course",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
AIPROXY_BASE_URL = "https://aiproxy.sanand.workers.dev"
EMBEDDINGS_FILE = "knowledge_embeddings.npz"

# Rate limiter
rate_limiter = RateLimiter(max_requests=60, time_window=60)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    image: Optional[str] = Field(None, description="Base64 encoded image")

class ReferenceLink(BaseModel):
    url: str
    text: str

class APIResponse(BaseModel):
    answer: str
    links: List[ReferenceLink]
    confidence: Optional[float] = None
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    embeddings_loaded: bool
    total_chunks: int

# Global variables for embeddings
embeddings_data = None

@app.on_event("startup")
async def load_embeddings():
    """Load precomputed embeddings and metadata on startup"""
    global embeddings_data
    try:
        if not os.path.exists(EMBEDDINGS_FILE):
            raise FileNotFoundError(f"Embeddings file {EMBEDDINGS_FILE} not found")
        
        data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
        embeddings_data = {
            "chunks": data["chunks"],
            "embeddings": data["embeddings"].astype(np.float32),
            "metadata": json.loads(data["metadata"].item()) if isinstance(data["metadata"].item(), str) else data["metadata"].item()
        }
        logger.info(f"Loaded {len(embeddings_data['chunks'])} chunks and embeddings")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        raise

async def get_embedding(text: str) -> List[float]:
    """Get embedding using AI Proxy with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{AIPROXY_BASE_URL}/openai/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": EMBEDDING_MODEL,
                        "input": text
                    }
                )
                response.raise_for_status()
                return response.json()["data"][0]["embedding"]
        except Exception as e:
            logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=503, detail="Failed to generate embedding")
            await asyncio.sleep(2 ** attempt)

async def analyze_image_with_gemini(image_b64: str) -> str:
    """Analyze image using Google Gemini API"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_b64)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(image_data)
            tmp_file_path = tmp_file.name
        
        try:
            # Initialize Gemini client
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Upload and analyze image
            uploaded_file = genai.upload_file(tmp_file_path)
            
            prompt = """Analyze this technical image and provide a concise description (2-3 sentences max) focusing on:
1. Visible code snippets/error messages (transcribe exactly)
2. UI elements relevant to technical problems
3. Data visualizations/charts
4. Key textual content related to technical issues
Omit decorative elements and focus on actionable technical details."""
            
            response = model.generate_content([uploaded_file, prompt])
            return response.text.strip()
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return "Unable to analyze the provided image"

async def generate_answer(context: str, question: str) -> str:
    """Generate answer using GPT-4o-mini via AI Proxy"""
    system_prompt = """You are an expert teaching assistant for a data science course. Use ONLY the provided context to answer questions accurately and concisely.

FORMATTING RULES:
- Use **bold** for key terms and concepts
- Format code blocks with triple backticks and language specification
- Use bullet points for multi-step explanations
- Include relevant examples when available in context

CONTENT RULES:
- Answer directly and concisely
- Always reference specific course materials or discussions when applicable
- For code questions, provide exact syntax and examples
- If multiple approaches exist, mention the recommended one first
- For unclear questions, ask for clarification

CITATION RULES:
- Never mention "context" or "provided information" directly
- If information is insufficient, state: "This specific information is not covered in the available course materials"
- Focus on practical, actionable guidance

EXAMPLE RESPONSE:
For model selection in assignments, use **gpt-3.5-turbo-0125** as specified in the course requirements.


**Key considerations:**
- Cost efficiency for assignments
- Consistent results across students
- API availability and rate limits"""
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{AIPROXY_BASE_URL}/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": LLM_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                        ],
                        "temperature": 0.2,
                        "max_tokens": 800,
                        "top_p": 0.9
                    }
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=503, detail="Failed to generate answer")
            await asyncio.sleep(2 ** attempt)

def extract_references(top_indices: List[int], similarities: np.ndarray) -> List[ReferenceLink]:
    """Extract reference links with relevance-based text"""
    references = []
    seen_urls = set()
    
    for i, idx in enumerate(top_indices):
        if i >= 3:  # Limit to top 3 references
            break
            
        meta = embeddings_data["metadata"][idx]
        chunk_text = embeddings_data["chunks"][idx]
        similarity_score = similarities[idx]
        
        # Skip low-relevance chunks
        if similarity_score < 0.3:
            continue
            
        urls_to_process = []
        
        if meta.get("source_type") == "discourse":
            if meta.get("topic_url"):
                urls_to_process.append(("topic", meta["topic_url"]))
            for post_url in meta.get("post_urls", []):
                urls_to_process.append(("post", post_url))
        elif meta.get("source_type") == "tds_course":
            if meta.get("original_url"):
                urls_to_process.append(("course", meta["original_url"]))
        
        for url_type, url in urls_to_process:
            if url and url not in seen_urls:
                seen_urls.add(url)
                
                # Extract meaningful text snippet
                text_snippet = chunk_text[:200].strip()
                if len(chunk_text) > 200:
                    text_snippet += "..."
                
                # Clean up text snippet
                text_snippet = text_snippet.replace('\n', ' ').replace('  ', ' ')
                
                references.append(ReferenceLink(
                    url=url,
                    text=text_snippet
                ))
    
    return references

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        embeddings_loaded=embeddings_data is not None,
        total_chunks=len(embeddings_data["chunks"]) if embeddings_data else 0
    )

@app.post("/api/", response_model=APIResponse)
async def answer_question(
    request: QueryRequest,
    http_request: Request,
    _: str = Depends(verify_api_key)
):
    """Main API endpoint for answering questions"""
    start_time = time.time()
    
    try:
        # Rate limiting
        client_ip = http_request.client.host
        await rate_limiter.check_rate_limit(client_ip)
        
        logger.info(f"Processing question: {request.question[:100]}...")
        
        # Check if embeddings are loaded
        if embeddings_data is None:
            raise HTTPException(status_code=503, detail="Embeddings not loaded")
        
        # Process image if provided
        enhanced_question = request.question
        if request.image:
            logger.info("Analyzing provided image")
            image_description = await analyze_image_with_gemini(request.image)
            enhanced_question = f"{request.question}\n\nImage content: {image_description}"
        
        # Get question embedding
        question_embed = await get_embedding(enhanced_question)
        
        # Calculate cosine similarities
        question_norm = np.linalg.norm(question_embed)
        chunk_norms = np.linalg.norm(embeddings_data["embeddings"], axis=1)
        
        similarities = np.dot(embeddings_data["embeddings"], question_embed) / (chunk_norms * question_norm)
        
        # Get top 5 most similar chunks
        top_indices = np.argsort(similarities)[-5:][::-1]
        top_similarities = similarities[top_indices]
        
        # Build context from top chunks
        context_chunks = []
        for i, idx in enumerate(top_indices):
            chunk = embeddings_data["chunks"][idx]
            score = top_similarities[i]
            context_chunks.append(f"[Source {i+1}] (Relevance: {score:.2f})\n{chunk}")
        
        context = "\n\n---\n\n".join(context_chunks)
        
        # Generate answer
        answer = await generate_answer(context, request.question)
        
        # Extract references
        references = extract_references(top_indices, similarities)
        
        processing_time = time.time() - start_time
        confidence = float(np.mean(top_similarities))
        
        logger.info(f"Question answered in {processing_time:.2f}s with confidence {confidence:.2f}")
        
        return APIResponse(
            answer=answer,
            links=references,
            confidence=confidence,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return {"error": exc.detail, "status_code": exc.status_code}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
