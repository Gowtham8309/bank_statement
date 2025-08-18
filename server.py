import os, math, traceback, logging
from typing import Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import uvicorn

# ---------- Logging & Debug ----------
DEBUG = os.getenv("DEBUG", "1") not in ("0", "false", "False", "")
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger("bank-extractor")

# ---------- Optional deps ----------
try:
    import numpy as np
except Exception:
    np = None
try:
    import pandas as pd
except Exception:
    pd = None

# ---------- Import pipeline (rename 'pipeline (8).py' to 'pipeline.py') ----------
PIPELINE_AVAILABLE = True
_import_err = None
try:
    import pipeline  # must expose extract_from_pdf(...) or process_pdf_bytes(...)
    logger.info("pipeline imported OK")
except Exception as e:
    PIPELINE_AVAILABLE = False
    _import_err = f"{type(e).__name__}: {e}"
    logger.exception("Pipeline import failed")

# ---------- JSON sanitizer ----------
def _is_nan_or_inf(x: Any) -> bool:
    try:
        if isinstance(x, float):
            return not math.isfinite(x)
        if np is not None and isinstance(x, (np.floating,)):
            return not math.isfinite(float(x))
    except Exception:
        return False
    return False

def _sanitize_json(obj: Any) -> Any:
    # Scalars
    if _is_nan_or_inf(obj):
        return None

    # numpy → Python
    if np is not None:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj) if math.isfinite(float(obj)) else None
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return [_sanitize_json(x) for x in obj.tolist()]

    # pandas → JSON safe
    if pd is not None:
        if isinstance(obj, pd.DataFrame):
            return _sanitize_json(obj.where(pd.notnull(obj), None).to_dict(orient="records"))
        if isinstance(obj, pd.Series):
            return _sanitize_json(obj.where(pd.notnull(obj), None).to_dict())

    # containers
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_json(v) for v in obj]

    return obj

# ---------- FastAPI app ----------
app = FastAPI(title="Bank Statement Extractor API", version="1.0.0")

# CORS (open for demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    status = "ok" if PIPELINE_AVAILABLE else "pipeline_import_error"
    return {"status": status, "debug": DEBUG, "import_error": _import_err}

@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    llm_fallback: bool = Query(True),
    use_gemini: bool = Query(True),
):
    if not PIPELINE_AVAILABLE:
        raise HTTPException(status_code=500, detail=f"Pipeline import error: {_import_err}")

    fn = file.filename or ""
    if not fn.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file (.pdf).")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        if hasattr(pipeline, "extract_from_pdf"):
            result = pipeline.extract_from_pdf(content, llm_fallback=llm_fallback, use_gemini=use_gemini)
        elif hasattr(pipeline, "process_pdf_bytes"):
            result = pipeline.process_pdf_bytes(content, llm_fallback=llm_fallback, use_gemini=use_gemini)
        else:
            raise RuntimeError("Expected `extract_from_pdf` or `process_pdf_bytes` in pipeline.py")

        # Make JSON-safe
        result = _sanitize_json(result)

        # Extra safety: common key “transactions” being a DataFrame or has NaN
        if isinstance(result, dict) and "transactions" in result:
            result["transactions"] = _sanitize_json(result["transactions"])

        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Extraction failed: %s\n%s", e, tb)
        if DEBUG:
            return JSONResponse(
                status_code=500,
                content={"detail": f"{type(e).__name__}: {e}", "traceback": tb[:8000]},
            )
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False, log_level="debug" if DEBUG else "info")
