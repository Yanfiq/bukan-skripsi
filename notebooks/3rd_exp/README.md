# Eksperimen ke Third

TL;DR\
Reimplementasi arsitektur MV-CLIP dari MMSD2.0 dengan struktur training notebook
yang lebih baru. Model memakai text encoder multilingual
`sentence-transformers/clip-ViT-B-32-multilingual-v1`, image encoder CLIP
ViT-B/32, fusion dimension 512, dan ensemble tiga head dari implementasi
upstream. Karena tokenizer multilingual tidak memiliki konvensi EOT milik
CLIP, cabang teks menggunakan attention-masked mean pooling bawaan model.
