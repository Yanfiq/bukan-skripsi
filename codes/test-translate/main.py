# %%
from transformers import T5ForConditionalGeneration, T5Tokenizer, BitsAndBytesConfig
import torch

# %%
quantization_config = BitsAndBytesConfig(
    # load_in_8bit = True
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",            # Better accuracy than standard fp4
    # bnb_4bit_use_double_quant=True        # Compresses the quantization constants
)

# %%
model_name = 'google/madlad400-10b-mt'
model = T5ForConditionalGeneration.from_pretrained(
    model_name, 
    device_map="cuda", 
    torch_dtype=torch.float16,
    quantization_config=quantization_config
)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# %%
text = "<2id> What a successful toast, it looks so delicious!"
input_ids = tokenizer(text, return_tensors="pt").input_ids.to('cuda')
outputs = model.generate(input_ids=input_ids)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

# %%



