# Eksperimen ke Second

TL;DR\
Model adaptasi dari MMSD2.0 dengan beberapa perubahan
- Ganti transformers.models.bert.modeling_bert.BertLayer jadi nn.TransformerEncoderLayer biar modern
- Decouple encodernya, alhasil embeddingnya enggak aligned kayak di model CLIP

Hasil\
- Model ngalamin penurunan performa sekitar 10% (F1-Score)

Next exp\
- Implementasi optimal transport pake wassertain distance di interactive view buat coba ngatasi encoder yang gk aligned