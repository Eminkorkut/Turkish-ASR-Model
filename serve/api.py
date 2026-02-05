"""
FastAPI server for Turkish ASR model.

Endpoints:
- POST /transcribe: Transcribe audio file
- GET /health: Health check
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import torch
from typing import Optional
from pydantic import BaseModel

# Import ASR components
from inference import ASRInference


# =====================================================
# Configuration
# =====================================================

class ServerConfig:
    """Server configuration."""
    MODEL_PATH: str = os.environ.get("ASR_MODEL_PATH", "./runs/best_model.pt")
    N_MEL_CHANNELS: int = int(os.environ.get("N_MEL_CHANNELS", "80"))
    D_MODEL: int = int(os.environ.get("D_MODEL", "256"))
    N_HEADS: int = int(os.environ.get("N_HEADS", "4"))
    N_BLOCKS: int = int(os.environ.get("N_BLOCKS", "8"))
    USE_BEAM_SEARCH: bool = os.environ.get("USE_BEAM_SEARCH", "false").lower() == "true"
    BEAM_WIDTH: int = int(os.environ.get("BEAM_WIDTH", "10"))


# =====================================================
# Models
# =====================================================

class TranscriptionResponse(BaseModel):
    """Response model for transcription."""
    text: str
    duration_ms: float
    
class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str


# =====================================================
# App Setup
# =====================================================

app = FastAPI(
    title="Turkish ASR API",
    description="Automatic Speech Recognition API for Turkish language",
    version="1.0.0"
)

# Global ASR instance
asr_model: Optional[ASRInference] = None


@app.on_event("startup")
async def load_model():
    """Load ASR model on startup."""
    global asr_model
    
    config = ServerConfig()
    
    if not os.path.exists(config.MODEL_PATH):
        print(f"Warning: Model not found at {config.MODEL_PATH}")
        return
        
    try:
        asr_model = ASRInference(
            model_path=config.MODEL_PATH,
            n_mel_channels=config.N_MEL_CHANNELS,
            d_model=config.D_MODEL,
            n_heads=config.N_HEADS,
            n_blocks=config.N_BLOCKS,
            use_beam_search=config.USE_BEAM_SEARCH,
            beam_width=config.BEAM_WIDTH
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")


# =====================================================
# Endpoints
# =====================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=asr_model is not None,
        device=str(asr_model.device) if asr_model else "N/A"
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe uploaded audio file.
    
    Accepts: wav, mp3, flac, ogg formats
    """
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )
    
    # Save to temp file
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # Transcribe
    try:
        import time
        start = time.time()
        
        text = asr_model.transcribe(tmp_path)
        
        duration_ms = (time.time() - start) * 1000
        
        return TranscriptionResponse(text=text, duration_ms=duration_ms)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
        
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/transcribe/batch")
async def transcribe_batch(files: list[UploadFile] = File(...)):
    """
    Transcribe multiple audio files.
    
    Returns list of transcriptions.
    """
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            # Save temp file
            ext = os.path.splitext(file.filename or "")[1].lower() or ".wav"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                contents = await file.read()
                tmp.write(contents)
                tmp_path = tmp.name
            
            # Transcribe
            text = asr_model.transcribe(tmp_path)
            results.append({"filename": file.filename, "text": text, "error": None})
            
            # Cleanup
            os.unlink(tmp_path)
            
        except Exception as e:
            results.append({"filename": file.filename, "text": "", "error": str(e)})
    
    return {"results": results}


# =====================================================
# Run Server
# =====================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
