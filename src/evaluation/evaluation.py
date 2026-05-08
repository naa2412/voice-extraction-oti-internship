import pandas as pd
import numpy as np
import torch
import librosa
from jiwer import wer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# load model manual
print("[INFO] Memuat model...")
processor = AutoProcessor.from_pretrained(lokasi_model_lokal)
model = AutoModelForSpeechSeq2Seq.from_pretrained(lokasi_model_lokal)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"[INFO] Memulai evaluasi pada {len(df_test)} data audio...")

teks_asli = []
teks_prediksi_ai = []

for index, row in df_test.iterrows():
    audio_path = row['audio_path']
    referensi = row['sentence'].lower()

    try:
        audio_array, sr = librosa.load(audio_path, sr=16000)
        audio_array = audio_array.astype(np.float32)

        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                **inputs,
                language="indonesian",
                task="transcribe"
            )

        prediksi = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower()

        teks_asli.append(referensi)
        teks_prediksi_ai.append(prediksi)

        if (index + 1) % 15 == 0:
            print(f"> Selesai memproses {index + 1}/300 audio testing...")

    except Exception as e:
        print(f"[ERROR] Gagal memproses audio {index}: {e}")

# hitung WER
error_rate = wer(teks_asli, teks_prediksi_ai)
akurasi_kata = (1 - error_rate) * 100

print("\n" + "="*50)
print("HASIL EVALUASI MODEL")
print("="*50)
print(f"Total Data Dites  : {len(teks_asli)} audio")
print(f"Word Error Rate   : {error_rate:.2%}")
print(f"Akurasi Model     : {akurasi_kata:.2f}%")
print("="*50)

df_hasil = pd.DataFrame({"Audio Asli": teks_asli, "Tebakan AI": teks_prediksi_ai})
df_hasil.to_csv(f"{folder_tujuan}/hasil_evaluasi.csv", index=False)
print("[INFO] Rincian tebakan disimpan di hasil_evaluasi.csv")