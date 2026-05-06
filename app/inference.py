import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration


MODEL_ID = "azharm1224/juku-version-2"


print(f"[INFO] Loading model from: {MODEL_ID}")
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
model.eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(DEVICE)
print(f"[INFO] Model loaded. Running on: {DEVICE}")


def transcribe(audio_path: str, language: str = "id") -> str:
    """
    Transcribe an audio file using the fine-tuned Whisper model.

    Args:
        audio_path (str): Path to the input audio file (.wav / .mp3).
        language (str): 'id' for Indonesian.

    Returns:
        str: Transcription result.

    """

    # load and resample audio to 16kHz 
    audio, sr = librosa.load(audio_path, sr=16000)

    # preprocess
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).to(DEVICE)

    # inference
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs["input_features"],
            language=language,
            task="transcribe"
        )

    # decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0].strip()
