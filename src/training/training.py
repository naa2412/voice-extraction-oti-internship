from datasets import Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))

# load processor dan model whisper-small
model_id = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_id, language="Indonesian", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_id)

# fungsi untuk memproses audio menjadi format yang dipahami whisper
def prepare_dataset(batch):
    audio = batch["audio_path"]
    # input fitur dari audio
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # label dari teks
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

print("[INFO] Memproses dataset untuk Whisper...")
mapped_dataset = dataset.map(prepare_dataset, remove_columns=["audio_path", "sentence"])

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import default_data_collator

# argumen Training
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-new-vision",
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    num_train_epochs=30,
    fp16=True,
    predict_with_generate=True,
    generation_max_length=225,
)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # pisahkan dan pad input audio
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # pisahkan dan pad label teks
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # ganti token padding dengan -100 agar tidak dihitung saat loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # hapus awalan bos_token jika ada
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# inisialisasi trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=mapped_dataset,
    data_collator=data_collator,
)

# mulai training
print("[INFO] Memulai proses Fine-Tuning...")
trainer.train()

# simpan model
model.save_pretrained("./whisper-juku-vision-final")
processor.save_pretrained("./whisper-juku-vision-final")
print("[INFO] Model berhasil disimpan!")