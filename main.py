from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import json

from vector_db import VectorDB
from document_db import DocumentDB, chunk_text
from ollama_client import OllamaClient
from distance import get_dist_fn

# Constants
DIMS = 16

# Initialize databases and clients
db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

# FastAPI app
app = FastAPI(title="VectorDB", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# =====================================================================
#  JSON HELPERS
# =====================================================================

def json_string(s: str) -> str:
    """Escape string for JSON."""
    return json.dumps(s)


def json_vector(v: List[float]) -> str:
    """Convert vector to JSON string."""
    return json.dumps([round(x, 4) for x in v])


def parse_vector(s: str) -> List[float]:
    """Parse comma-separated vector string."""
    try:
        return [float(x.strip()) for x in s.split(',')]
    except ValueError:
        return []


def extract_string(body: str, key: str) -> str:
    """Extract JSON string field value."""
    try:
        data = json.loads(body)
        return data.get(key, "")
    except json.JSONDecodeError:
        return ""


def extract_int(body: str, key: str, default: int = 0) -> int:
    """Extract JSON integer field value."""
    try:
        data = json.loads(body)
        return data.get(key, default)
    except json.JSONDecodeError:
        return default


# =====================================================================
#  DEMO DATA  (16D categorical vectors)
# =====================================================================

def load_demo():
    """Load 20 demo vectors across 4 categories."""
    dist_fn = get_dist_fn("cosine")
    # Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
    db.insert("Linked List: nodes connected by pointers", "cs",
              [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06], dist_fn)
    db.insert("Binary Search Tree: O(log n) search and insert", "cs",
              [0.88, 0.82, 0.78, 0.74, 0.15, 0.10, 0.08, 0.12, 0.06, 0.07, 0.08, 0.05, 0.09, 0.06, 0.07, 0.10], dist_fn)
    db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
              [0.82, 0.76, 0.88, 0.80, 0.20, 0.18, 0.12, 0.09, 0.07, 0.06, 0.08, 0.07, 0.08, 0.09, 0.06, 0.07], dist_fn)
    db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
              [0.85, 0.80, 0.75, 0.82, 0.18, 0.14, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.10, 0.08, 0.09, 0.07], dist_fn)
    db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
              [0.87, 0.78, 0.70, 0.76, 0.13, 0.11, 0.09, 0.14, 0.08, 0.07, 0.06, 0.08, 0.07, 0.10, 0.08, 0.09], dist_fn)
    db.insert("Calculus: derivatives integrals and limits", "math",
              [0.12, 0.15, 0.18, 0.10, 0.91, 0.86, 0.78, 0.72, 0.08, 0.06, 0.07, 0.09, 0.07, 0.08, 0.06, 0.10], dist_fn)
    db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
              [0.20, 0.18, 0.15, 0.12, 0.88, 0.90, 0.82, 0.76, 0.09, 0.07, 0.08, 0.06, 0.10, 0.07, 0.08, 0.09], dist_fn)
    db.insert("Probability: distributions random variables Bayes theorem", "math",
              [0.15, 0.12, 0.20, 0.18, 0.84, 0.80, 0.88, 0.82, 0.07, 0.08, 0.06, 0.10, 0.09, 0.06, 0.09, 0.08], dist_fn)
    db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
              [0.22, 0.16, 0.14, 0.20, 0.80, 0.85, 0.76, 0.90, 0.08, 0.09, 0.07, 0.06, 0.08, 0.10, 0.07, 0.06], dist_fn)
    db.insert("Combinatorics: permutations combinations generating functions", "math",
              [0.18, 0.20, 0.16, 0.14, 0.86, 0.78, 0.84, 0.80, 0.06, 0.07, 0.09, 0.08, 0.06, 0.09, 0.10, 0.07], dist_fn)
    db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
              [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.90, 0.86, 0.78, 0.72, 0.08, 0.06, 0.09, 0.07], dist_fn)
    db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
              [0.06, 0.08, 0.07, 0.09, 0.09, 0.06, 0.08, 0.07, 0.86, 0.90, 0.82, 0.76, 0.07, 0.09, 0.06, 0.08], dist_fn)
    db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
              [0.09, 0.07, 0.06, 0.08, 0.08, 0.09, 0.07, 0.06, 0.82, 0.78, 0.90, 0.84, 0.09, 0.07, 0.08, 0.06], dist_fn)
    db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
              [0.07, 0.09, 0.08, 0.06, 0.06, 0.07, 0.09, 0.08, 0.78, 0.82, 0.86, 0.90, 0.06, 0.08, 0.07, 0.09], dist_fn)
    db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
              [0.06, 0.07, 0.10, 0.09, 0.10, 0.06, 0.07, 0.10, 0.85, 0.80, 0.76, 0.82, 0.09, 0.07, 0.10, 0.06], dist_fn)
    db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
              [0.09, 0.07, 0.08, 0.10, 0.08, 0.09, 0.07, 0.06, 0.08, 0.07, 0.09, 0.06, 0.91, 0.85, 0.78, 0.72], dist_fn)
    db.insert("Football: tackles touchdowns field goals and strategy", "sports",
              [0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08, 0.07, 0.09, 0.08, 0.07, 0.87, 0.89, 0.82, 0.76], dist_fn)
    db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
              [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.09, 0.06, 0.07, 0.08, 0.83, 0.80, 0.88, 0.82], dist_fn)
    db.insert("Chess: openings endgames tactics strategic board game", "sports",
              [0.25, 0.20, 0.22, 0.18, 0.22, 0.18, 0.20, 0.15, 0.06, 0.08, 0.07, 0.09, 0.80, 0.84, 0.78, 0.90], dist_fn)
    db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
              [0.06, 0.08, 0.07, 0.09, 0.08, 0.06, 0.09, 0.07, 0.10, 0.08, 0.06, 0.07, 0.85, 0.82, 0.86, 0.80], dist_fn)


# Load demo data at startup
load_demo()


# =====================================================================
#  DEMO VECTOR ENDPOINTS
# =====================================================================

@app.get("/search")
def search(v: str, k: int = 5, metric: str = "cosine", algo: str = "hnsw"):
    """K-NN search."""
    q = parse_vector(v)
    if len(q) != DIMS:
        raise HTTPException(status_code=400, detail=f"need {DIMS}D vector")

    out = db.search(q, k, metric, algo)

    results = []
    for hit in out.hits:
        results.append({
            "id": hit.id,
            "metadata": hit.meta,
            "category": hit.cat,
            "distance": round(hit.dist, 6),
            "embedding": [round(x, 4) for x in hit.emb]
        })

    return {
        "results": results,
        "latencyUs": out.us,
        "algo": out.algo,
        "metric": out.metric
    }


@app.post("/insert")
def insert(request: Request):
    """Insert a demo vector."""
    body = request.body.decode()
    meta = extract_string(body, "metadata")
    cat = extract_string(body, "category")
    emb_str = extract_string(body, "embedding")

    try:
        if isinstance(emb_str, str):
            emb = json.loads(emb_str)
        else:
            emb = emb_str
    except json.JSONDecodeError:
        emb = []

    if not meta or not emb or len(emb) != DIMS:
        raise HTTPException(status_code=400, detail="invalid body")

    id_ = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return {"id": id_}


@app.delete("/delete/{id}")
def delete(id: int):
    """Delete by ID."""
    ok = db.remove(id)
    return {"ok": ok}


@app.get("/items")
def items():
    """List all demo vectors."""
    items = db.all()
    return [{
        "id": v.id,
        "metadata": v.metadata,
        "category": v.category,
        "embedding": [round(x, 4) for x in v.emb]
    } for v in items]


@app.get("/benchmark")
def benchmark(v: str, k: int = 5, metric: str = "cosine"):
    """Compare all 3 algorithms."""
    q = parse_vector(v)
    if len(q) != DIMS:
        raise HTTPException(status_code=400, detail=f"need {DIMS}D vector")

    b = db.benchmark(q, k, metric)
    return {
        "bruteforceUs": b.bf_us,
        "kdtreeUs": b.kd_us,
        "hnswUs": b.hnsw_us,
        "itemCount": b.n
    }


@app.get("/hnsw-info")
def hnsw_info():
    """HNSW graph structure and layer stats."""
    gi = db.hnsw_info()
    return gi


@app.get("/stats")
def stats():
    """Database statistics."""
    return {
        "count": db.size(),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"]
    }


# =====================================================================
#  DOCUMENT + RAG ENDPOINTS
# =====================================================================

@app.post("/doc/insert")
def doc_insert(request: Request):
    """Embed and store document."""
    body = request.body.decode()
    title = extract_string(body, "title")
    text = extract_string(body, "text")

    if not title or not text:
        raise HTTPException(status_code=400, detail="need title and text")

    chunks = chunk_text(text, 250, 30)
    ids = []

    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            raise HTTPException(
                status_code=500,
                detail="Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text && ollama pull llama3.2"
            )
        chunk_title = title if len(chunks) == 1 else f"{title} [{i+1}/{len(chunks)}]"
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return {
        "ids": ids,
        "chunks": len(chunks),
        "dims": doc_db.get_dims()
    }


@app.delete("/doc/delete/{id}")
def doc_delete(id: int):
    """Delete document chunk."""
    ok = doc_db.remove(id)
    return {"ok": ok}


@app.get("/doc/list")
def doc_list():
    """List all stored documents."""
    docs = doc_db.all()
    return [{
        "id": d.id,
        "title": d.title,
        "preview": d.text[:120] + "…" if len(d.text) > 120 else d.text,
        "words": len(d.text.split())
    } for d in docs]


@app.post("/doc/search")
def doc_search(request: Request):
    """Fast retrieval for UI visualizer."""
    body = request.body.decode()
    question = extract_string(body, "question")
    k = extract_int(body, "k", 3)

    if not question:
        raise HTTPException(status_code=400, detail="need question")

    q_emb = ollama.embed(question)
    if not q_emb:
        raise HTTPException(status_code=500, detail="Ollama unavailable")

    hits = doc_db.search(q_emb, k)

    return {
        "contexts": [{
            "id": h[1].id,
            "title": h[1].title,
            "distance": round(h[0], 4)
        } for h in hits]
    }


@app.post("/doc/ask")
def doc_ask(request: Request):
    """Full RAG pipeline: embed → retrieve → generate."""
    body = request.body.decode()
    question = extract_string(body, "question")
    k = extract_int(body, "k", 3)

    if not question:
        raise HTTPException(status_code=400, detail="need question")

    # Step 1: embed the question
    q_emb = ollama.embed(question)
    if not q_emb:
        raise HTTPException(status_code=500, detail="Ollama unavailable")

    # Step 2: retrieve top-k relevant chunks
    hits = doc_db.search(q_emb, k)

    # Step 3: build prompt
    ctx_parts = []
    for i, (dist, doc) in enumerate(hits):
        ctx_parts.append(f"[{i+1}] {doc.title}:\n{doc.text}\n\n")
    context = "".join(ctx_parts)

    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
        "Just answer the question naturally.\n\n"
        f"Context:\n{context}"
        f"Question: {question}\n\n"
        "Answer:"
    )

    # Step 4: generate answer
    answer = ollama.generate(prompt)

    # Step 5: return everything
    return {
        "answer": answer,
        "model": ollama.gen_model,
        "contexts": [{
            "id": h[1].id,
            "title": h[1].title,
            "text": h[1].text,
            "distance": round(h[0], 4)
        } for h in hits],
        "docCount": doc_db.size()
    }


@app.get("/status")
def status():
    """Ollama status and model info."""
    up = ollama.is_available()
    return {
        "ollamaAvailable": up,
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "docCount": doc_db.size(),
        "docDims": doc_db.get_dims(),
        "demoDims": DIMS,
        "demoCount": db.size()
    }


# =====================================================================
#  SERVE FRONTEND
# =====================================================================

@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve index.html."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == "__main__":
    import uvicorn

    ollama_up = ollama.is_available()
    print("=== VectorDB Engine ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}")
        print(f"  gen model: {ollama.gen_model}")

    uvicorn.run(app, host="0.0.0.0", port=8080)
