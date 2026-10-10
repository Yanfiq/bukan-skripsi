# Eksperimen ke Third

TL;DR\
Reimplementasi arsitektur MV-CLIP dari MMSD2.0 dengan struktur training notebook
yang lebih baru. Model memakai text encoder multilingual
`sentence-transformers/clip-ViT-B-32-multilingual-v1`, image encoder CLIP
ViT-B/32, fusion dimension 512, dan ensemble tiga head dari implementasi
upstream. Karena tokenizer multilingual tidak memiliki konvensi EOT milik
CLIP, cabang teks menggunakan attention-masked mean pooling bawaan model.

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
- Pemilihan representasi teks menggunakan attention-masked mean pooling. Kode
  asli memakai token EOT melalui `input_ids.argmax()`, tetapi asumsi tersebut
  tidak berlaku untuk tokenizer multilingual DistilBERT.
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
  optional encoder freezing, learning rate terpisah, pemeriksaan VRAM sebelum
  unfreezing, serta pembersihan CUDA cache sebelum test evaluation.
