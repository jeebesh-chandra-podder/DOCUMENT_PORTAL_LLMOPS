import os
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from src.document_ingestion.data_ingestion import (
    DocHandler,
    DocumentComparator,
    ChatIngestor,
    FaissManager,
)
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparatorLLM
from src.document_chat.retrieval import ConversationalRAG

''' - Config variables telling us where to save indexes and uploaded data.
    - FAISS (vector DB) needs persistence, uploads need storage. Defaults exist, but environment overrides let your app adapt without changing code.'''
FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")  # <--- keep consistent with save_local()

''' - Creating the FastAPI app instance.
    -  All routes (@app.get, @app.post) "attach" onto this object.'''
app = FastAPI(title="Document Portal API", version="0.1")

''' - Getting the project’s root directory (two levels up from this file). 
    - Makes static/templates paths relative → portable across dev/prod.
    - Path(__file__).resolve().parent.parent is a robust way to find the project root regardless of where the script runs from.
    - Mounting static files allows serving CSS, JS, images directly. '''
BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

''' - Configures Cross-Origin Resource Sharing to allow browser requests from any domain.
    - During development, ["*"] is convenient, but this is a security anti-pattern for production.
    - This screams "development configuration." In production, I'd specify exact origins like ["https://myapp.com", "https://api.myapp.com"].'''
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

''' - Serves an HTML page as the application's homepage. 
    - response_class=HTMLResponse tells FastAPI this returns HTML, improving documentation.
    - no-store prevents browser caching, useful during development. 
    - Passing request to template is required by Jinja2Templates. '''
@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    resp = templates.TemplateResponse("index.html", {"request": request})
    resp.headers["Cache-Control"] = "no-store"
    return resp

''' - Provides a simple endpoint to verify the service is running. 
    - Essential for production monitoring. Load balancers, orchestrators (Kubernetes), and monitoring systems use this to check service health.
    - Notice this is synchronous (no async) because it's intentionally simple - we want health checks to be fast and not depend on external resources.'''
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "document-portal"}

# ---------- ANALYZE ----------
''' - Accepts a PDF upload, processes it, and returns analysis results. 
    - FastAPIFileAdapter bridges FastAPI's UploadFile with the internal DocHandler interface.
    - Specific HTTPExceptions are re-raised (they have proper status codes), while generic exceptions become 500 errors. '''
@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Any:
    try:
        dh = DocHandler()
        saved_path = dh.save_pdf(FastAPIFileAdapter(file))
        text = _read_pdf_via_handler(dh, saved_path)
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

# ---------- COMPARE ----------
''' - Compares two uploaded documents and returns differences in a structured format.
        - Multiple file uploads using separate parameter names for semantic clarity.
        - _ = ref_path, act_path is a senior engineer pattern - acknowledging unused variables to prevent linter warnings while keeping them for debugging.
        - The df.to_dict(orient="records") conversion suggests the backend uses pandas DataFrames but the API returns JSON-serializable dictionaries. '''
@app.post("/compare")
async def compare_documents(reference: UploadFile = File(...), actual: UploadFile = File(...)) -> Any:
    try:
        dc = DocumentComparator()
        ref_path, act_path = dc.save_uploaded_files(
            FastAPIFileAdapter(reference), FastAPIFileAdapter(actual)
        )
        _ = ref_path, act_path
        combined_text = dc.combine_documents()
        comp = DocumentComparatorLLM()
        df = comp.compare_documents(combined_text)
        return {"rows": df.to_dict(orient="records"), "session_id": dc.session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

# ---------- CHAT: INDEX ----------
''' - Builds a FAISS index for document retrieval from multiple uploaded files.
    - Why this approach:
        - Flexible parameter design - session isolation is optional.
        - Sensible defaults for chunking parameters based on typical document processing needs.
        - Type hints make the API self-documenting. '''
@app.post("/chat/index")
async def chat_build_index(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
) -> Any:
    try:
        wrapped = [FastAPIFileAdapter(f) for f in files]
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )
        # NOTE: ensure your ChatIngestor saves with index_name="index" or FAISS_INDEX_NAME
        # e.g., if it calls FAISS.save_local(dir, index_name=FAISS_INDEX_NAME)
        ci.built_retriver(  # if your method name is actually build_retriever, fix it there as well
            wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k
        )
        return {"session_id": ci.session_id, "k": k, "use_session_dirs": use_session_dirs}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

# ---------- CHAT: QUERY ----------
@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
) -> Any:
    try:
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE  # type: ignore
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)  # build retriever + chain
        response = rag.invoke(question, chat_history=[])

        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


# ---------- Helpers ----------
class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile -> .name + .getbuffer() API"""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename
    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()

def _read_pdf_via_handler(handler: DocHandler, path: str) -> str:
    if hasattr(handler, "read_pdf"):
        return handler.read_pdf(path)  # type: ignore
    if hasattr(handler, "read_"):
        return handler.read_(path)  # type: ignore
    raise RuntimeError("DocHandler has neither read_pdf nor read_ method.")



# command for executing the fast api
# uvicorn api.main:app --reload    
#uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload