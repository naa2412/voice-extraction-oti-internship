from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    filename: str
    language: str
    transcription: str