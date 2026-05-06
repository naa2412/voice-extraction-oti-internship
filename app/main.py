import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import transcribe
from schemas import TranscriptionResponse

app = FastAPI(
    title="Voice Extraction API",
    description="Fine-Tuned Whisper Model for Fish Inventory in Makassar",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Voice Extraction API is running.",
        "endpoint": "POST /predict — upload an audio file to get transcription"
    }


@app.post("/predict", response_model=TranscriptionResponse)
async def predict(file: UploadFile = File(...), language: str = "id"):
    """
    Transcribe an uploaded audio file.

    - file: Audio file (.wav, .mp3, .flac, .ogg, .m4a)
    - language:  'id' for Indonesian

    """
    # validate extension
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Use: {ALLOWED_EXTENSIONS}"
        )

    # save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = transcribe(tmp_path, language=language)
        return TranscriptionResponse(
            filename=file.filename,
            language=language,
            transcription=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)