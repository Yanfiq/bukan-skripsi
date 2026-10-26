# Eksperimen ke Third

TL;DR\
Reimplementasi arsitektur MV-CLIP dari MMSD2.0 dengan struktur training notebook
yang lebih baru. Model memakai text encoder multilingual
`sentence-transformers/clip-ViT-B-32-multilingual-v1`, image encoder CLIP
ViT-B/32, fusion dimension 512, dan ensemble tiga head dari implementasi
upstream. Karena tokenizer multilingual tidak memiliki konvensi EOT milik
CLIP, cabang teks menggunakan attention-masked mean pooling bawaan model.

## Varian pooling

- `mv_clip_multilingual_mean_mean.py`: menggunakan masked-mean pooling pada
  text-only branch dan fused-text branch. Ini adalah varian yang paling sesuai
  dengan konfigurasi pooling checkpoint multilingual.
- `mv_clip_multilingual_mean_cls.py`: tetap menggunakan masked-mean pooling
  pada text-only branch, tetapi mengambil token `[CLS]` setelah multimodal
  fusion. Varian ini digunakan sebagai ablation yang lebih menyerupai pemilihan
  token EOT pada implementasi MV-CLIP asli.

## Perbedaan dengan implementasi asli MMSD2.0

- Text encoder asli dari `openai/clip-vit-base-patch32` diganti dengan
  `sentence-transformers/clip-ViT-B-32-multilingual-v1`, sehingga teks dalam
  Bahasa Indonesia dan lebih dari 50 bahasa lain dapat dipetakan ke embedding
  space CLIP.
- Image encoder tetap memakai OpenAI CLIP ViT-B/32. Berbeda dari kode asli
  yang memuat text dan image encoder melalui satu `CLIPModel`, eksperimen ini
  memuat multilingual text encoder dan image encoder secara terpisah.
- Hidden state text encoder berukuran 768 dimensi, lalu diproyeksikan ke shared
  CLIP space berukuran 512 dimensi menggunakan projection layer bawaan model
  multilingual. Kode asli langsung menghasilkan token text 512 dimensi.
- Text-only branch menggunakan attention-masked mean pooling. Fused-text branch
  menggunakan masked mean pada varian Mean/Mean dan token `[CLS]` pada varian
  Mean/CLS. Kode asli memakai token EOT melalui `input_ids.argmax()`, tetapi
  asumsi tersebut tidak berlaku untuk tokenizer multilingual DistilBERT.
- Struktur utama MV-CLIP tetap dipertahankan: 50 image tokens, fusion
  transformer 512 dimensi, adaptive text-image weighting, tiga classification
  heads, penjumlahan ketiga probability outputs, dan penjumlahan ketiga
  cross-entropy losses.
- Training loop mencatat loss, accuracy, precision, recall, dan F1 untuk train
  serta validation ke Weights & Biases. Kode asli hanya mencatat train loss dan
  validation metrics yang lebih terbatas.
- Test set hanya dievaluasi sekali menggunakan checkpoint dengan validation
  accuracy terbaik. Kode asli mengevaluasi test set setiap kali validation
  accuracy meningkat.
- Model checkpoint dicatat sebagai W&B artifact, dan tersedia mekanisme
  optional encoder freezing, learning rate terpisah, linear warmup/decay
  scheduler, pemeriksaan VRAM sebelum unfreezing, serta pembersihan CUDA cache
  sebelum test evaluation.

## Komparasi performa dengan implementasi asli MMSD2.0
| Model | Accuracy   | Precision | Recall  | F1       |
|-------|------------|-----------|---------|----------|
| MV-CLIP (MMSD2.0) | 85.4296 | 80.4078 | 87.4638 | 83.7875 |
| MV-MCLIP (3rd experiment) | 83.8107 | 83.8107 | 83.8107 | 83.8107 |
