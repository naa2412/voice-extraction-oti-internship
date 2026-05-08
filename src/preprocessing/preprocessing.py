import time
import os
import librosa
import soundfile as sf
from gtts import gTTS
from audiomentations import Compose, AddBackgroundNoise, TimeStretch, PitchShift, AddGaussianNoise, Gain, ClippingDistortion, LowPassFilter
import pandas as pd
import random

# siapkan komponen kata-kata 
tindakan = ["masuk", "keluarkan", "ambil", "simpan", "tambah", "catat"]
ikan = ["samaleng", "bulaeng", "jabiri", "balang-balang", "bala-balang", "kanjilo",
        "cambang-cambang", "bogolo", "bitte", "londeng", "kalengkere",
        "sunu", "bolu", "tuing-tuing", "mairo", "lure", "buntala", "katombo",
        "eja", "cakalang", "baronang", "cakalang", "tongkol", "layang", "cumi-cumi",
        "sikayu", "doang", "donge-donge", "tenggiri", "layang", "kerapu", "bandeng",
        "lele", "emas", "mujair", "betik", "betok", "gabus", "sepat", "betutu",
        "cupang", "belut", "sidat", "bandeng", "teri", "buntal", "kembung",
        "kakap merah", "kerapu", "cakalang", "baronang", "tongkol", "layang",
        "cumi-cumi", "kepiting", "udang", "rumput laut"]
grade = ["grade a", "grade b", "grade c", "kualitas a", "super", "biasa", "rijek"]
angka_ratusan = ["", "seratus", "dua ratus", "tiga ratus", "empat ratus", "lima ratus", "enam ratus", "tujuh ratus", "delapan ratus", "sembilan ratus"]
angka_puluhan = ["sepuluh", "dua puluh", "tiga puluh", "empat puluh", "lima puluh", "enam puluh", "tujuh puluh",
                 "delapan puluh", "sembilan puluh"]
angka_belasan = ["sebelas", "dua belas", "tiga belas", "empat belas", "lima belas", "enam belas", "tujuh belas", "delapan belas", "sembilan belas"]
angka_satuan = ["", "lima", "dua", "tiga", "satu", "empat", "enam", "tujuh", "delapan", "sembilan"]
satuan_wadah = ["kotak", "boks", "keranjang", "gabus", "ember", "coolbox"]
sapaan = ["daeng", "iye", "bos", "bosku", "pak", ""]
waktu = ["siang ini", "malam ini", "pagi ini", "sore ini", "barusan", "sekarang", "tadi", "baru masuk", ""]
keterangan = ["tolong", "cepat", "langsung", "catat maki", ""]

# fungsi pembuat kombinasi angka (misal: "dua puluh lima")
def generate_berat():
    # mengacak apakah mau pakai belasan atau puluhan+satuan
    if random.choice([True, False]):
        berat = random.choice(angka_belasan)
    else:
        ratusan = random.choice(angka_ratusan)
        puluhan = random.choice(angka_puluhan)
        satuan = random.choice(angka_satuan)
        berat = f"{ratusan} {puluhan} {satuan}".strip()

    # memastikan tidak return string kosong
    if not berat:
        berat = "sepuluh"

    return berat + random.choice([" kilo", " kg", " kilogram"])

def generate_wadah():
    angka = random.choice(["dua", "tiga", "empat", "lima", "sepuluh"])
    wadah = random.choice(satuan_wadah)
    return f"{angka} {wadah}"

# kumpulan template kalimat (SOP & kebiasaan bicara)
templates = [
    "{keterangan} {tindakan} {ikan} {wadah} {grade} {berat} {sapaan}",
    "{sapaan} {tindakan} {ikan} {berat} {grade} {waktu}",
    "{waktu} ada {ikan} {grade} {wadah} {berat} {tindakan} {sapaan}",
    "{ikan} {berat} {grade} {sapaan}",
    "{tindakan} {ikan} {iye} {berat} {grade} {waktu}",
    "{sapaan} {keterangan} {tindakan} {ikan} kualitas {grade} berat {berat}",
    "{ikan} {wadah} {berat} {grade} {tindakan} {waktu}"
]

# generate 2500 
skrip_kalimat = []
jumlah_data_yang_diinginkan = 2500

print(f"[INFO] Merakit {jumlah_data_yang_diinginkan} kalimat sintetis...")

for _ in range(jumlah_data_yang_diinginkan):
    template_terpilih = random.choice(templates)

    kalimat = template_terpilih.format(
        tindakan=random.choice(tindakan),
        ikan=random.choice(ikan),
        grade=random.choice(grade),
        berat=generate_berat(),
        wadah=random.choice(["", generate_wadah()]), # kadang pakai wadah, kadang tidak
        sapaan=random.choice(sapaan),
        iye=random.choice(["iye", ""]),
        keterangan=random.choice(keterangan),
        waktu=random.choice(waktu)
    )

    # bersihkan spasi ganda kalau ada variabel yang kosong
    kalimat_bersih = " ".join(kalimat.split())
    skrip_kalimat.append(kalimat_bersih)

# hapus duplikat
skrip_kalimat = list(set(skrip_kalimat))

# cetak 5 contoh untuk melihat hasilnya
print("\nContoh Hasil Generate:")
for i in range(5):
    print(f"{i+1}. {skrip_kalimat[i]}")

print(f"\nTotal kalimat unik siap di-generate audionya: {len(skrip_kalimat)}")
os.makedirs("dataset_sintetis", exist_ok=True)
data_list = []

# augmentasi dengan menambahkan noise suara pantai dan angin
augmentasi_juku = Compose([
    # 1. suara noise
    AddBackgroundNoise(sounds_path="/content/drive/Shareddrives/oti_intership/noise_pelabuhan", min_snr_db=5.0, max_snr_db=15.0, p=0.7),

    # 2. kecepatan bicara (santai vs suru-buru)
    TimeStretch(min_rate=0.8, max_rate=1.25, leave_length_unchanged=False, p=0.5),

    # 3. variasi pita suara (berat vs melengking)
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),

    # 4. suara pecah/teriak (clipping)
    ClippingDistortion(min_percentile_threshold=10, max_percentile_threshold=30, p=0.2),

    # 5. suara mendem / mic tertutup tangan
    LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=4000, p=0.3),

    # 6. efek mic kresek (sinyal jelek/statis)
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),

    # 7. volume total (jauh/dekat)
    Gain(min_gain_db=-10, max_gain_db=5, p=0.4)
])

print("[INFO] Mulai generate data sintetis...")
folder_tujuan = "/content/drive/Shareddrives/oti_intership/data_sintesis"
os.makedirs(folder_tujuan, exist_ok=True)

data_list = []
temp_audio = "temp_sementara.mp3"

for i, teks in enumerate(skrip_kalimat):
    file_nama = f"audio_{i}.wav"
    path_output = os.path.join(folder_tujuan, file_nama)

    try:
        # generate suara dasar (TTS) ke file sementara
        tts = gTTS(text=teks, lang='id', slow=False)
        tts.save(temp_audio)

        # load audio sementara untuk ditambahkan noise
        y, sr = librosa.load(temp_audio, sr=16000)

        # aplikasikan noise
        try:
            y_augmented = augmentasi_juku(samples=y, sample_rate=sr)
        except Exception as e:
            print(f"  [Warning] Augmentasi gagal di audio {i}, pakai suara asli. Error: {e}")
            y_augmented = y

        # simpan hasil akhir ke Google Drive
        sf.write(path_output, y_augmented, 16000)

        # simpan ke daftar dataset (gunakan path lokal Drive untuk CSV)
        data_list.append({"audio_path": path_output, "sentence": teks})

        # hapus file sementara agar memori Colab tidak penuh
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

        time.sleep(1.5)

        # indikator progres
        if (i + 1) % 50 == 0:
            print(f"> Berhasil memproses {i + 1} / {len(skrip_kalimat)} audio...")

    except Exception as e:
        print(f"[ERROR] Gagal memproses kalimat ke-{i}: {e}")
        time.sleep(10)

# jadikan dataframe
df = pd.DataFrame(data_list)

# simpan metadata ke drive
csv_path = os.path.join(folder_tujuan, "metadata_dataset.csv")
df.to_csv(csv_path, index=False)

print("\n[SUKSES] Dataset selesai dibuat dan CSV tersimpan di Drive!")
print(df.head())