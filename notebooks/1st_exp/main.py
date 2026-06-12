# %% [markdown]
# # fIRST rUN

# %%
# paket-pakettttt
# %pip install pandas numpy matplotlib seaborn scikit-learn transformers torch torchinfo torchvision timm gdown

import gdown
import os
from pathlib import Path

# %%
cwd = Path.cwd()
project_root_dir = cwd.parents[1]
dataset_root_dir = project_root_dir / "datasets"

### modifiable
dataset_name = "MMSD2.0" # ganti dataset lewat sini
dataset_dir = dataset_root_dir / dataset_name
dataset_data_dir = dataset_dir / "data"
###

# symlink
project_dataset_dir = cwd / "dataset"
if not project_dataset_dir.exists():
    project_dataset_dir.symlink_to(dataset_data_dir)

# %%
# donwload MMSD2.0 images
gdown.download(url='https://drive.google.com/uc?id=1mK0Nf-jv_h2bgHUCRM4_EsdTiiitZ_Uj', output=project_dataset_dir.as_posix(), quiet=False)
gdown.download(url='https://drive.google.com/uc?id=1AOWzlOz5hmdO39dEmzhQ4z_nabgzi7Tu', output=project_dataset_dir.as_posix(), quiet=False)
gdown.download(url='https://drive.google.com/uc?id=1dJERrVlp7DlNSXk-uvbbG6Rv7uvqTOKd', output=project_dataset_dir.as_posix(), quiet=False)
gdown.download(url='https://drive.google.com/uc?id=1pODuKC4gP6-QDQonG8XTqI8w8ds68mE3', output=project_dataset_dir.as_posix(), quiet=False)

# pastiin udh install 7z dari package manager
!7z x ./dataset/dataset_image.zip -o./dataset

# %%
# download whitelist
gdown.download("https://drive.google.com/file/d/18yU3HaSvBNYml2EfKn-uG7vUKXGDMt6d/view?usp=drive_link", output=project_dataset_dir.as_posix(), quiet=False)

# %% [markdown]
# # Data Processing

# %%
import pandas as pd
import numpy as np
import random

from transformers import AutoTokenizer, AutoProcessor, AutoModel, AlbertTokenizer, AlbertModel
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight

from torchinfo import summary
from torch import nn
import timm

# %%
df = pd.read_json("./dataset/text_json_id/dataset_translated_fixed.json", orient="records", dtype={"image_id": str, "label": int})
df.head()

# %%
# whitelist itu list gambar yang gk dominan teksnya

with open("./dataset/whitelist.txt", "r") as f:
    whitelist = set(line.strip()[:-4] for line in f)

df = df[df["image_id"].isin(whitelist)]

# %%
print(df.describe())
print(df.info())
print(df.value_counts(['split', 'label']))

# %%
# drop NaN values
# df = df.dropna()

# drop rows with missing image in ./dataset/dataset_image/<image_id>.jpg
def check_image_exists(image_id):
    try:
        img = Image.open(f"./dataset/dataset_image/{image_id}.jpg")
        del img
        return True
    except FileNotFoundError:
        print(f"Image {image_id} not found.")
        return False
df['image_exists'] = df['image_id'].apply(check_image_exists)

# count how many images are missing
print(df['image_exists'].value_counts())

# %%
# # random data
# random_row = df.sample(n=1).iloc[0]
# print(f"Text: {random_row['text_translated']}")
# print(f"Label: {random_row['label']}")
# img = Image.open(f"./dataset/dataset_image/{random_row['image_id']}.jpg")
# img.show()


# %% [markdown]
# # Experiment

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

# %%
torch.cuda.set_device(1)

# %%
class SimpleSarcasmDataset(Dataset):
    def __init__(self, data, tokenizer, processor):
        self.data = data
        self.tokenizer = tokenizer
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        text_inputs = self.tokenizer(item['text_translated'], return_tensors="pt", padding='max_length', truncation=True, max_length=128)
        
        image = Image.open(f"./dataset/dataset_image/{item['image_id']}.jpg").convert("RGB")
        image_inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            'input_ids': text_inputs['input_ids'],
            'pixel_values': image_inputs['pixel_values'],
            'label': torch.tensor(item['label'], dtype=torch.float)
        }

# %%
# Source - https://stackoverflow.com/a/58612961
# Posted by Szymon Maszke, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-02, License - CC BY-SA 4.0

class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return self.dataframe.iloc[index]

# %%
#dataset split
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# Get all indices and labels
indices = list(range(len(df)))
labels = df.label

# Split indices (stratify=labels ensures both sets have the same class proportions)
train_indices, val_test_indices = train_test_split(indices, test_size=0.4, stratify=labels)

labels_test_val = labels.iloc[val_test_indices].tolist()
val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, stratify=labels_test_val)

# Create virtual subsets using these indices
train_dataset_subset = Subset(df, train_indices)
val_dataset_subset = Subset(df, val_indices)
test_dataset_subset = Subset(df, test_indices)

# %%
print(df.iloc[train_dataset_subset.indices].value_counts(['split', 'label']))
print(df.iloc[val_dataset_subset.indices].value_counts(['split', 'label']))
print(df.iloc[test_dataset_subset.indices].value_counts(['split', 'label']))

# %%
# 1. Set global seeds
torch.manual_seed(42)

# 2. Create a local generator with a specific seed
g = torch.Generator()
g.manual_seed(42)

# train_dataset = PandasDataset(df[df["split"] == "train"])
# val_dataset = PandasDataset(df[df["split"] == "valid"])
# test_dataset = PandasDataset(df[df["split"] == "test"])

train_dataset = PandasDataset(df.iloc[train_dataset_subset.indices])
val_dataset = PandasDataset(df.iloc[val_dataset_subset.indices])
test_dataset = PandasDataset(df.iloc[test_dataset_subset.indices])

# %%
text_model_name = "indobenchmark/indobert-base-p2"
# vision_model_name = "WinKawaks/vit-small-patch16-224"
vision_model_name = "google/vit-base-patch16-224"

# %%
tokenizer = AutoTokenizer.from_pretrained(text_model_name)

print(f"Max length: {tokenizer.model_max_length}")
print(f"Vocab size: {tokenizer.vocab_size}")

dummy_text = "Contoh teks untuk tokenisasi."
encoded = tokenizer(dummy_text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)

print(f"Input IDs shape: {encoded['input_ids'].shape}")

# reverse tokenization
decoded_text = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
print(f"Decoded text: {decoded_text}")

# %%
processor = AutoProcessor.from_pretrained(vision_model_name)

# 1. See what the model expects for dimensions
print(f"Expected size: {processor.size}") 

# 2. See normalization values (Mean and Std)
print(f"Mean: {processor.image_mean}")
print(f"Std: {processor.image_std}")

# 3. Check the output shape with a dummy image (e.g., a random tensor)
import torch
dummy_image = torch.randn(3, 500, 500) # Simulating a 500x500 RGB image
pixel_values = processor(dummy_image, return_tensors="pt")["pixel_values"]

print(f"Processed image shape: {pixel_values.shape}")
# Likely [1, 3, 224, 224]

# %%
train_dataloader = DataLoader(SimpleSarcasmDataset(train_dataset, tokenizer, processor), batch_size=32, shuffle=True, generator=g)
valid_dataloader = DataLoader(SimpleSarcasmDataset(val_dataset, tokenizer, processor), batch_size=32, shuffle=False)
test_dataloader = DataLoader(SimpleSarcasmDataset(test_dataset, tokenizer, processor), batch_size=32, shuffle=False)

# %%
class SarcasmModel(nn.Module):
    def __init__(self, text_model, vision_model):
        super().__init__()
        self.text_encoder = text_model
        self.vision_encoder = vision_model
        
        # # Freezing encoders for PoC/Baseline
        # for param in self.text_encoder.parameters(): param.requires_grad = False
        # for param in self.vision_encoder.parameters(): param.requires_grad = False

        # batch_first=True makes it much more intuitive (matches TF/Keras style)
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        
        self.norm = nn.LayerNorm(768)
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, pixel_values):
        # # 1. Feature Extraction (No Gradients for frozen models)
        # with torch.no_grad():
        # Text: [Batch, 768]
        t_feat = self.text_encoder(input_ids).pooler_output
        # Vision Patches: [Batch, 196, 768] (removing the CLS token at index 0)
        v_feat = self.vision_encoder(pixel_values).last_hidden_state[:, 1:, :]
        
        # 2. Shape Alignment: Text needs a sequence dimension for MHA
        t_feat = t_feat.unsqueeze(1) # Shape: [B, 1, 768]
        
        # 3. Cross-Attention: Text queries the Image Patches
        # attn_output shape: [B, 1, 768]
        attn_output, attn_weights = self.cross_attention(query=t_feat, 
                                                         key=v_feat, 
                                                         value=v_feat)
        
        # 4. Residual + Norm
        fused = self.norm(t_feat + attn_output)
        
        # 5. Classification
        # We squeeze(1) to go from [B, 1, 768] -> [B, 768]
        logits = self.classifier(fused.squeeze(1))
        
        return logits, attn_weights

# %%
text_model = AutoModel.from_pretrained(text_model_name)
# vision_model = AutoModel.from_pretrained("google/vit-base-patch16-224")
vision_model = AutoModel.from_pretrained(vision_model_name)
model = SarcasmModel(text_model, vision_model).to(device)

# %%
# Define dummy inputs that match the shapes you found above
batch_size = 32
seq_len = 128 # or tokenizer.model_max_length
img_size = 224

# Create dummy tensors
dummy_input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long).to(device)
dummy_pixel_values = torch.zeros((batch_size, 3, img_size, img_size)).to(device)

# Print the summary
print(summary(model, input_data=[dummy_input_ids, dummy_pixel_values]))

# remove the dummy tensors to free up memory
del dummy_input_ids
del dummy_pixel_values

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

# %%
#train
epochs = 10
for t in range(epochs):
    model.train()
    correct_train = 0
    total_train = 0
    loss_train = 0.0
    for item in train_dataloader:
        model.train()
        input_ids = item['input_ids'].to(device)
        pixel_values = item['pixel_values'].to(device)
        label = item['label'].to(device)

        # print("Input IDs shape:", input_ids.shape) # [B, 1, 128]
        # print("Pixel values shape:", pixel_values.shape) # [B, 1, 3, 224, 224]
        # print("Label shape:", label.shape) # [B]

        fixed_input_ids = input_ids.squeeze(1)          # [B, 128]
        fixed_pixel_values = pixel_values.squeeze(1)    # [B, 3, 224, 224]
        fixed_label = label.unsqueeze(1)     # [B, 1]

        # print("Fixed Input IDs shape:", fixed_input_ids.shape)
        # print("Fixed Pixel values shape:", fixed_pixel_values.shape)
        # print("Fixed Label shape:", fixed_label.shape)

        output_logits, weights = model(fixed_input_ids, fixed_pixel_values)
        output = torch.sigmoid(output_logits)

        pred = (output > 0.5).float()

        # print(f"Output: {output.item():.4f}, Label: {label.item():.0f}")
        # print(f"Output[0]: {output[0].item():.4f}, Label[0]: {label[0].item():.0f}")
        loss = criterion(output, fixed_label.view_as(output))

        correct_train += (pred == fixed_label).sum().item()
        total_train += fixed_label.size(0)
        loss_train += loss.item() * fixed_label.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    correct_valid = 0
    total_valid = 0
    loss_valid = 0.0
    with torch.no_grad():
        for item in valid_dataloader:
            input_ids = item['input_ids'].to(device)
            pixel_values = item['pixel_values'].to(device)
            label = item['label'].to(device)

            fixed_input_ids = input_ids.squeeze(1)          # [B, 128]
            fixed_pixel_values = pixel_values.squeeze(1)    # [B, 3, 224, 224]
            fixed_label = label.unsqueeze(1)     # [B, 1]

            output_logits, weights = model(fixed_input_ids, fixed_pixel_values)
            output = torch.sigmoid(output_logits)

            pred = (output > 0.5).float()
            loss = criterion(output, fixed_label.view_as(output))

            correct_valid += (pred == fixed_label).sum().item()
            total_valid += fixed_label.size(0)
            loss_valid += loss.item() * fixed_label.size(0)


    epoch_train_acc = 100 * correct_train / total_train
    epoch_valid_acc = 100 * correct_valid / total_valid
    epoch_train_loss = loss_train / total_train
    epoch_valid_loss = loss_valid / total_valid
    print(f'Epoch [{t+1}/{epochs}] Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_acc:.2f}%, Valid Loss: {epoch_valid_loss:.4f}, Valid Accuracy: {epoch_valid_acc:.2f}%')

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# %%
def plot_attention_map(image_tensor, attn_weights, title="Attention Map"):
    """
    image_tensor: [3, 224, 224] (The original processed image)
    attn_weights: [1, 196] (The weights for this specific image)
    """
    # 1. Reshape to spatial grid (14x14)
    # Assuming 196 patches (14*14 = 196)
    grid_size = int(np.sqrt(attn_weights.shape[-1])) # Should be 14
    heatmap = attn_weights.view(grid_size, grid_size).detach().cpu().numpy()

    # 2. Normalize for visualization
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # 3. Upscale to image size (224x224)
    # We use bilinear interpolation for a smooth "glow" effect
    heatmap_resized = cv2.resize(heatmap, (224, 224))

    # 4. Prepare the original image
    # Convert from [3, 224, 224] tensor to [224, 224, 3] numpy
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    # Un-normalize if you used ImageProcessor normalization
    img = (img - img.min()) / (img.max() - img.min())

    # 5. Plotting
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    # Overlay heatmap on image
    ax[1].imshow(img)
    ax[1].imshow(heatmap_resized, cmap='jet', alpha=0.5) # alpha controls transparency
    ax[1].set_title(title)
    ax[1].axis('off')

    plt.show()

# Example usage:
# plot_attention_map(pixel_values[0], weights[0])

# %%
#test
model.eval()
correct = 0
total = 0
preview_count = 0
max_previews = 10
with torch.no_grad():
    for item in test_dataloader:
        input_ids = item['input_ids'].to(device)
        pixel_values = item['pixel_values'].to(device)
        label = item['label'].to(device)

        # print("Input IDs shape:", input_ids.shape) # [B, 1, 128]
        # print("Pixel values shape:", pixel_values.shape) # [B, 1, 3, 224, 224]
        # print("Label shape:", label.shape) # [B]

        fixed_input_ids = input_ids.squeeze(1)          # [B, 128]
        fixed_pixel_values = pixel_values.squeeze(1)    # [B, 3, 224, 224]
        fixed_label = label.unsqueeze(1)     # [B, 1]

        output_logits, weights = model(fixed_input_ids, fixed_pixel_values)
        output = torch.sigmoid(output_logits)
        pred = (output > 0.5).float()
        correct += (pred == fixed_label).sum().item()
        total += fixed_label.size(0)


        show_preview = torch.rand(1).item() < 0.5 and preview_count < max_previews
        if show_preview:
                preview_count += 1
                # Pick one random sample from this batch and plot its attention
                batch_size = 8
                rand_idx = torch.randint(0, batch_size, (1,)).item()

                print(f"True: {fixed_label[rand_idx].item():.0f}, Pred: {pred[rand_idx].item():.0f}")
                print(f"Text: {tokenizer.decode(fixed_input_ids[rand_idx], skip_special_tokens=True)}")

                # weights shape: [B, 1, 196] -> take sample and squeeze the seq dim
                attn_sample = weights[rand_idx].squeeze(0)
                plot_attention_map(fixed_pixel_values[rand_idx], attn_sample,
                                title=f"True: {fixed_label[rand_idx].item():.0f}, Pred: {pred[rand_idx].item():.0f}")
                # for i in range(len(fixed_label)):
                #     print(f"True: {fixed_label[i].item():.0f}, Pred: {pred[i].item():.0f}")
                #     # print text
                #     print(f"Text: {tokenizer.decode(fixed_input_ids[i], skip_special_tokens=True)}")
                #     plot_attention_map(fixed_pixel_values[i], weights[i], title=f"True: {fixed_label[i].item():.0f}, Pred: {pred[i].item():.0f}")
epoch_acc = 100 * correct / total
print(f'Test Accuracy: {epoch_acc:.2f}%')

# %%
# try to do some inference from external data
image = Image.open("./messy_room.jpg").convert("RGB")
image_inputs = processor(images=image, return_tensors="pt").to(device)
text_inputs = tokenizer("kamarmu bersih sekali, kamu pasti rajin membersihkannya", return_tensors="pt", padding='max_length', truncation=True, max_length=128).to(device)

model.eval()
with torch.no_grad():
    output_logits, weights = model(text_inputs['input_ids'], image_inputs['pixel_values'])
    output = torch.sigmoid(output_logits)
    print(f"Inference Output: {output}")
    pred = (output > 0.5).float()
    print(f"Inference Prediction: {'Sarcastic' if pred.item() == 1 else 'Not Sarcastic'}")

    attn_sample = weights[0].squeeze(0)

    plot_attention_map(image_inputs['pixel_values'][0], weights[0])

# %%


# %%



