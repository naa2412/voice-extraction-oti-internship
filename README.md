# Voice Extraction System for Fish Inventory in Makassar Based on Fine-Tuned Whisper

Proyek ini merupakan sistem Automatic Speech Recognition (ASR) untuk membantu proses pencatatan inventori ikan di Makassar menggunakan input suara. Sistem ini dikembangkan dengan melakukan fine-tuning model Whisper Small agar mampu mengenali kosakata khusus pada domain perikanan, seperti nama ikan, grade kualitas, jumlah berat, serta beberapa sapaan lokal Makassar/Bugis.

Sistem menerima file audio sebagai input, kemudian mengembalikan hasil transkripsi dalam format JSON melalui API berbasis FastAPI.

---

## Deskripsi Proyek

Pada proses inventori ikan di pelabuhan atau pasar ikan, pencatatan data seperti jenis ikan, grade, dan berat masih sering dilakukan secara manual. Proses manual tersebut dapat menyebabkan pencatatan menjadi lambat, kurang efisien, dan rentan terhadap kesalahan.

Untuk mengatasi permasalahan tersebut, proyek ini mengembangkan sistem berbasis suara yang dapat mentranskripsikan ucapan pekerja menjadi teks secara otomatis. Model yang digunakan adalah Whisper Small yang telah di-fine-tune menggunakan dataset sintetis bertema inventori ikan di Makassar.

Contoh input suara:

```text
masuk ikan sunu grade a dua puluh lima kilo
```

Contoh output transkripsi:

```json
{
  "filename": "test1.mp3",
  "language": "id",
  "transcription": "masuk ikan sunu grade a dua puluh lima kilo"
}
```

---

## Fitur Utama

- Fine-tuned Whisper untuk domain inventori ikan.
- Mendukung bahasa Indonesia dengan kosakata lokal Makassar/Bugis.
- Mendukung beberapa format audio, yaitu `.wav`, `.mp3`, `.flac`, `.ogg`, dan `.m4a`.
- API inference menggunakan FastAPI.
- Deployment menggunakan Docker pada HuggingFace Spaces.
- Output API berbentuk JSON.


---

## Teknologi yang Digunakan

- Python
- PyTorch
- HuggingFace Transformers
- OpenAI Whisper
- FastAPI
- Docker
- HuggingFace Spaces
- gTTS
- Audiomentations

---

## Model

Base model yang digunakan:

```text
openai/whisper-small
```

Model hasil fine-tuning:

```text
azharm1224/juku-version-2
```

Model hasil fine-tuning tidak di-upload langsung ke repository GitHub karena ukurannya cukup besar, yaitu sekitar 967 MB. Oleh karena itu, model di-load langsung dari HuggingFace Hub saat aplikasi dijalankan.

---

## Dataset

Dataset yang digunakan merupakan dataset sintetis domain inventori ikan di Makassar. Dataset dibuat menggunakan pendekatan:

1. Template-based sentence generation
2. Text-to-Speech menggunakan gTTS
3. Augmentasi audio menggunakan audiomentations

Jumlah dataset:

| Split | Jumlah Audio |
|---|---:|
| Training | 2.499 audio |
| Testing | 300 audio |
| Total | 2.799 audio |

Dataset tidak dimasukkan ke repository GitHub karena ukurannya terlalu besar. Dataset dapat diakses melalui Google Drive berikut:

Training: https://drive.google.com/drive/folders/1tvvYxp2yXVUquTyumL-teN-j2D3U_jiT?usp=drive_link

Testing: https://drive.google.com/drive/folders/13eucayAiqc5FSOb2km8BJocvTvKQQY8_?usp=drive_link

---

## Struktur Repository


```text
voice-extraction-oti-internship/
├── README.md
├── report.pdf
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   └── (model file: .pkl / .pth / dll)
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   └── utils/
└── app/
    ├── main.py         
    ├── inference.py     
    └── schemas.py       
```

---

## Instalasi Lokal

Clone repository:

```bash
git clone https://github.com/naa2412/voice-extraction-oti-internship.git
cd voice-extraction-oti-internship
```

Buat virtual environment:

```bash
python -m venv venv
```

Aktifkan virtual environment.

Untuk Windows:

```bash
venv\Scripts\activate
```

Untuk Linux/Mac:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Menjalankan API Secara Lokal

Jalankan server FastAPI:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Jika berhasil, API akan berjalan pada:

```text
http://localhost:7860
```

Dokumentasi Swagger dapat dibuka melalui:

```text
http://localhost:7860/docs
```

---

## Cara Tes Model

Model dapat dites dengan mengirim file audio ke endpoint `/predict`.

---

### 1. Tes melalui HuggingFace Spaces

Model yang sudah di-deploy dapat diakses melalui URL berikut:

```text
https://azharm1224-voice-extraction-oti.hf.space
```

Endpoint utama:

```text
POST /predict
```

Format input:

| Field | Type | Keterangan |
|---|---|---|
| `file` | File | File audio yang ingin ditranskripsikan |

Contoh menggunakan `curl`:

```bash
curl -X POST "https://azharm1224-voice-extraction-oti.hf.space/predict" \
  -F "file=@sample_audio/test1.mp3" \
  -F "language=id"
```

Contoh response:

```json
{
  "filename": "test1.mp3",
  "language": "id",
  "transcription": "masuk ikan sunu grade a dua puluh lima kilo"
}
```

---

### 2. Tes melalui Postman

Langkah-langkah pengujian menggunakan Postman:

1. Buka Postman.
2. Pilih method `POST`.
3. Masukkan URL berikut:

```text
https://azharm1224-voice-extraction-oti.hf.space/predict
```

4. Masuk ke tab `Body`.
5. Pilih `form-data`.
6. Tambahkan field berikut:

| Key | Type | Value |
|---|---|---|
| `file` | File | Pilih file audio `.mp3`, `.wav`, `.flac`, `.ogg`, atau `.m4a` |


7. Klik `Send`.
8. Jika berhasil, API akan mengembalikan hasil transkripsi dalam format JSON.

---

### 3. Tes secara Lokal

Jika API dijalankan secara lokal, gunakan perintah berikut:

```bash
curl -X POST "http://localhost:7860/predict" \
  -F "file=@sample_audio/test1.mp3" \
  -F "language=id"
```

Contoh output:

```json
{
  "filename": "test1.mp3",
  "language": "id",
  "transcription": "masuk ikan sunu grade a dua puluh lima kilo"
}
```

---

## Format Audio yang Didukung

API mendukung format audio berikut:

```text
.wav
.mp3
.flac
.ogg
.m4a
```

Jika format file tidak didukung, API akan mengembalikan error.

---

## Hasil Evaluasi

Model dievaluasi menggunakan 300 data testing sintetis yang terpisah dari data training.

| Metrik | Nilai |
|---|---:|
| Word Error Rate (WER) | 0.20% |
| Akurasi Model | 99.80% |

Hasil ini menunjukkan bahwa model mampu mengenali kosakata inventori ikan dengan sangat baik pada dataset uji sintetis. Namun, hasil ini juga mengindikasikan kemungkinan overfitting karena dataset training dan testing dibuat dari template dan kamus kata yang sama.

---

## Keterbatasan

Beberapa keterbatasan sistem:

1. Dataset masih bersifat sintetis dan belum sepenuhnya merepresentasikan suara asli nelayan atau pekerja di lapangan.
2. Model berpotensi overfitting karena pola data training dan testing masih berasal dari template dan kamus kata yang sama.
3. Kosakata model masih terbatas pada kamus ikan, grade, sapaan, dan pola kalimat yang digunakan saat pembuatan dataset.
4. Deployment berjalan di CPU sehingga response time dapat lebih lambat, sekitar 3–5 detik per request.
5. Sistem saat ini baru menghasilkan transkripsi dan belum melakukan ekstraksi entitas seperti nama ikan, grade, dan berat secara terstruktur.

---