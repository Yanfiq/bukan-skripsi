# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## JALANIN INI CUMA KALO ENVIRONMENTNYA DISPOSABLE (Remote Jupyter Server)

# %%
import shutil
import subprocess
from pathlib import Path
import gc
import os
import random
import time

# Reduce allocator fragmentation for the large encoder activations and Adam
# state tensors. This must be set before PyTorch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# %%
subprocess.run(["uv", "add", "nbformat", "gdown", "torch", "torchvision", "transformers==4.49.0", "sentence-transformers", "pillow", "pandas", "scikit-learn", "tqdm", "torchinfo", "wandb"])

# %%
# Optional: Install 7z if on Linux
subprocess.run(["apt", "install", "-y", "p7zip-full"])

# %%
cwd = Path.cwd().resolve()

# Jupyter normally starts in this notebook's directory, whereas the paired
# Python file is often launched from the repository root. Find the repository
# by its datasets directory instead of relying on a fixed number of parents.
project_root_dir = next(
    (path for path in (cwd, *cwd.parents) if (path / "datasets").is_dir()),
    cwd,
)
dataset_root_dir = project_root_dir / "datasets"

# Symlink or set dataset directory
dataset_dir = cwd / "dataset"
if not dataset_dir.exists():
    upstream_data_dir = dataset_root_dir / "MMSD2.0" / "data"
    if upstream_data_dir.exists():
        dataset_dir = upstream_data_dir
    else:
        # Clone if missing
        subprocess.run(["git", "clone", "--depth", "1", "--no-checkout", "https://github.com/Yanfiq/MMSD2.0.git", "dataset"])
        subprocess.run(["git", "sparse-checkout", "set", "data"], cwd=cwd / "dataset")
        subprocess.run(["git", "checkout", "main"], cwd=cwd / "dataset")
        data_dir = cwd / "dataset" / "data"
        if data_dir.exists():
            for item in data_dir.iterdir():
                shutil.move(str(item), str(cwd / "dataset" / item.name))
            data_dir.rmdir()

# %% [markdown]
# ### Download MMSD2.0 Images & Whitelist (if not already downloaded)

# %%
import gdown
# Avoid downloading and extracting the large image archive on every rerun.
# The metadata repository contains an empty dataset_image placeholder, so the
# directory existing by itself does not mean the archive was extracted.
image_dir = dataset_dir / "dataset_image"
if not image_dir.is_dir() or not any(image_dir.glob("*.jpg")):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    gdown.download(url='https://drive.google.com/uc?id=1mK0Nf-jv_h2bgHUCRM4_EsdTiiitZ_Uj', output=str(dataset_dir) + "/", quiet=False)
    gdown.download(url='https://drive.google.com/uc?id=1AOWzlOz5hmdO39dEmzhQ4z_nabgzi7Tu', output=str(dataset_dir) + "/", quiet=False)
    gdown.download(url='https://drive.google.com/uc?id=1dJERrVlp7DlNSXk-uvbbG6Rv7uvqTOKd', output=str(dataset_dir) + "/", quiet=False)
    gdown.download(url='https://drive.google.com/uc?id=1pODuKC4gP6-QDQonG8XTqI8w8ds68mE3', output=str(dataset_dir) + "/", quiet=False)
    subprocess.run(
        ["7z", "x", str(dataset_dir / "dataset_image.zip"), f"-o{dataset_dir}"],
        check=True,
    )


# %% [markdown]
# ## Hyperparameters & Configuration

# %%
class BlankObject:
    pass

params = BlankObject()
params.text_model_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
params.vision_model_name = "openai/clip-vit-base-patch32"
params.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
params.simple_linear = False
params.fuse_dim = 512        # CLIP ViT-B/32 shared projection dimension
params.text_size = 512
params.image_size = 768
params.layers = 3
params.dropout_rate = 0.1
params.num_train_epochs = 10
params.freeze_encoder_epochs = 0
params.vram_reservation_target_gib = 14.0
params.vram_safety_margin_gib = 0.5
params.vram_poll_interval_seconds = 60
params.train_batch_size = 8
params.dev_batch_size = 8
params.max_len = 77
params.encoder_learning_rate = 1e-6
params.task_learning_rate = 5e-4
params.label_count = 2
params.label_number = 2
params.output_dir = "./saved_models"
params.model = "multilingual_mv_clip_vit_b_32"
params.seed = 42
params.wandb_project = "skripsi_3rd-exp"

# %% [markdown]
# # Dataset Loading & Preprocessing

# %%
import random
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoProcessor
import wandb

# Make data ordering and parameter initialization repeatable. Exact numerical
# equality can still depend on the GPU, CUDA and PyTorch versions in use.
random.seed(params.seed)
np.random.seed(params.seed)
torch.manual_seed(params.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(params.seed)

# %%
json_path = dataset_dir / "text_json_id" / "dataset_translated_fixed.json"
df = pd.read_json(json_path, orient="records", dtype={"image_id": str, "label": int}).set_index("image_id")
df.head()

# %%
# # Filter out text-heavy images using whitelist
# # commented because for now i want to try training with all images, not just the whitelist
# whitelist_path = dataset_dir / "whitelist.txt"
# if whitelist_path.exists():
#     with open(whitelist_path, "r") as f:
#         whitelist = set(line.strip().replace(".jpg", "") for line in f if line.strip())
#     df = df[df.index.isin(whitelist)]

# print(f"Total samples after whitelist filtering: {len(df)}")
# print(df.describe())
# print(df.info())
# print(df.value_counts(['split', 'label']))

# %%
class MMSD2_id_dataset(Dataset):
    def __init__(self, dataframe, dataset_dir):
        self.data = dataframe.to_dict(orient="index")
        self.image_ids = list(self.data.keys())
        self.dataset_dir = Path(dataset_dir)
        for img_id in self.image_ids:
            self.data[img_id]["image_path"] = self.dataset_dir / "dataset_image" / f"{img_id}.jpg"

    def image_loader(self, img_id):
        img_path = self.data[img_id]["image_path"]
        return Image.open(img_path).convert("RGB")

    def text_loader(self, img_id):
        return self.data[img_id]["text"]

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        text = self.text_loader(img_id)
        image = self.image_loader(img_id)
        label = self.data[img_id]["label"]
        return text, image, label, img_id

    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def collate_func(batch_data):
        if len(batch_data) == 0:
            return [], [], [], []

        text_list = []
        image_list = []
        label_list = []
        id_list = []
        for instance in batch_data:
            text_list.append(instance[0])
            image_list.append(instance[1])
            label_list.append(instance[2])
            id_list.append(instance[3])
        return text_list, image_list, label_list, id_list

# %% [markdown]
# ### Split Strategy: Default Upstream Split (Recommended)

# %%
train_dataset = MMSD2_id_dataset(df[df["split"] == "train"], dataset_dir)
val_dataset = MMSD2_id_dataset(df[df["split"] == "valid"], dataset_dir)
test_dataset = MMSD2_id_dataset(df[df["split"] == "test"], dataset_dir)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

# %% [markdown]
# ### (Alternative) Custom Stratified Split

# %%
# indices = list(range(len(df)))
# labels = df.label
# train_indices, val_test_indices = train_test_split(indices, test_size=0.4, stratify=labels, random_state=42)
# labels_test_val = labels.iloc[val_test_indices].tolist()
# val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, stratify=labels_test_val, random_state=42)

# train_dataset = MMSD2_id_dataset(df.iloc[train_indices], dataset_dir)
# val_dataset = MMSD2_id_dataset(df.iloc[val_indices], dataset_dir)
# test_dataset = MMSD2_id_dataset(df.iloc[test_indices], dataset_dir)

# %% [markdown]
# # Model Architecture: Original MMSD2.0 MV-CLIP Fusion

# %%
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, CLIPVisionModelWithProjection
from transformers.models.bert.modeling_bert import BertLayer
from sentence_transformers import SentenceTransformer
from sklearn import metrics
from tqdm import tqdm, trange

# %%
# Reset default device to standard CPU to prevent meta-device errors from previous kernel runs
if hasattr(torch, "set_default_device"):
    torch.set_default_device(None)

device = torch.device(params.device)
print(f"Using device: {device}")

# %%
class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                output_attentions=False,
            )[0]
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class SarcasmModel(nn.Module):
    def __init__(self, args):
        super(SarcasmModel, self).__init__()
        self.args = args
        self.fuse_dim = getattr(args, "fuse_dim", 512)

        # The multilingual checkpoint is a text-only DistilBERT model trained
        # to match the unchanged OpenAI CLIP ViT-B/32 image embedding space.
        self.multilingual_text_model = SentenceTransformer(args.text_model_name)
        self.text_encoder = self.multilingual_text_model[0].auto_model
        self.text_projection = self.multilingual_text_model[2]
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.vision_model_name
        )

        # 3. Multimodal Cross-Attention Fusion
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.config.hidden_size = self.fuse_dim
        self.config.num_attention_heads = 8
        self.trans = MultimodalEncoder(self.config, layer_number=args.layers)

        # 4. Unimodal projection heads
        if args.simple_linear:
            self.text_linear = nn.Linear(args.text_size, args.text_size)
            self.image_linear = nn.Linear(args.image_size, args.image_size)
        else:
            self.text_linear = nn.Sequential(
                nn.Linear(args.text_size, args.text_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
            self.image_linear = nn.Sequential(
                nn.Linear(args.image_size, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )

        # 5. Classifiers & loss
        self.classifier_fuse = nn.Linear(args.text_size, args.label_number)
        self.classifier_text = nn.Linear(args.text_size, args.label_number)
        self.classifier_image = nn.Linear(args.image_size, args.label_number)

        self.loss_fct = nn.CrossEntropyLoss()
        self.att = nn.Linear(args.text_size, 1, bias=False)
        self.encoders_frozen = False

    def set_encoders_trainable(self, trainable):
        """Freeze or unfreeze both pretrained encoders and their projections."""
        encoder_modules = (
            self.multilingual_text_model,
            self.vision_encoder,
        )
        for module in encoder_modules:
            for parameter in module.parameters():
                parameter.requires_grad = trainable

        self.encoders_frozen = not trainable
        for module in encoder_modules:
            module.train(self.training if trainable else False)

    def train(self, mode=True):
        """Keep frozen encoders deterministic while training the new layers."""
        super().train(mode)
        if mode and self.encoders_frozen:
            self.multilingual_text_model.eval()
            self.vision_encoder.eval()
        return self

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None, **kwargs):
        # Support dict input / kwargs
        if isinstance(input_ids, dict):
            kwargs = input_ids
            input_ids = kwargs.get('input_ids')
            attention_mask = kwargs.get('attention_mask')
            pixel_values = kwargs.get('pixel_values')
            if 'labels' in kwargs and labels is None:
                labels = kwargs['labels']

        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_hidden_states = text_output.last_hidden_state

        vision_output = self.vision_encoder(
            pixel_values=pixel_values,
            output_attentions=False,
            return_dict=True,
        )
        image_hidden_states = vision_output.last_hidden_state

        # Reproduce the Sentence-Transformers checkpoint's official masked-mean
        # pooling, followed by its learned 768 -> 512 CLIP-space projection.
        text_mask = attention_mask.unsqueeze(-1).to(text_hidden_states.dtype)
        text_pooled = (text_hidden_states * text_mask).sum(dim=1)
        text_pooled = text_pooled / text_mask.sum(dim=1).clamp(min=1e-9)
        text_feature = self.text_projection(
            {"sentence_embedding": text_pooled}
        )["sentence_embedding"]
        text_feature = self.text_linear(text_feature)

        # ViT-B/32's original image-only branch uses the post-layernorm 768-D
        # CLS representation, as in the upstream MMSD2.0 CLIPModel.
        image_pooled = self.vision_encoder.vision_model.post_layernorm(
            image_hidden_states[:, 0, :]
        )
        image_feature = self.image_linear(image_pooled)

        # The checkpoint's Dense layer is a bias-free linear 768 -> 512 map
        # with Identity activation, so it can also project each text token into
        # the shared space used by the multimodal fusion transformer.
        text_embeds = self.text_projection.activation_function(
            self.text_projection.linear(text_hidden_states)
        )
        image_embeds = self.vision_encoder.visual_projection(image_hidden_states)

        # 3. Multimodal Token Concatenation & Cross-Attention
        num_img_tokens = image_embeds.shape[1]
        batch_size = input_ids.shape[0]

        input_embeds = torch.cat((image_embeds, text_embeds), dim=1)  # [B, L_img + L_text, fuse_dim]

        img_mask = torch.ones((batch_size, num_img_tokens), device=attention_mask.device, dtype=attention_mask.dtype)
        combined_mask = torch.cat((img_mask, attention_mask), dim=-1)  # [B, L_img + L_text]

        extended_mask = combined_mask.unsqueeze(1).unsqueeze(2)
        extended_mask = extended_mask.to(dtype=next(self.parameters()).dtype)
        extended_mask = (1.0 - extended_mask) * -10000.0

        fuse_hiddens = self.trans(input_embeds, extended_mask, output_all_encoded_layers=False)
        fuse_hiddens = fuse_hiddens[-1]  # [B, L_total, fuse_dim]

        # 4. Multimodal pooled representations. The multilingual tokenizer has
        # no CLIP EOT convention, so use its official attention-masked mean.
        new_image_feature = fuse_hiddens[:, 0, :]  # Visual CLS token [B, fuse_dim]
        new_text_tokens = fuse_hiddens[:, num_img_tokens:, :]  # Fused text tokens [B, L_text, fuse_dim]
        new_text_feature = (new_text_tokens * text_mask).sum(dim=1)
        new_text_feature = new_text_feature / text_mask.sum(dim=1).clamp(min=1e-9)

        # 5. Adaptive Attention Fusion
        text_weight = self.att(new_text_feature)
        image_weight = self.att(new_image_feature)
        att = F.softmax(torch.cat((text_weight, image_weight), dim=-1), dim=-1)
        tw = att[:, 0:1]
        iw = att[:, 1:2]
        fuse_feature = tw * new_text_feature + iw * new_image_feature

        # 6. Classification
        logits_fuse = self.classifier_fuse(fuse_feature)
        logits_text = self.classifier_text(text_feature)
        logits_image = self.classifier_image(image_feature)

        fuse_score = F.softmax(logits_fuse, dim=-1)
        text_score = F.softmax(logits_text, dim=-1)
        image_score = F.softmax(logits_image, dim=-1)

        # Retain the upstream ensemble behavior (the values sum to 3, while
        # argmax predictions are identical to an equal-weight average).
        score = fuse_score + text_score + image_score

        outputs = (score,)
        if labels is not None:
            loss_fuse = self.loss_fct(logits_fuse, labels)
            loss_text = self.loss_fct(logits_text, labels)
            loss_image = self.loss_fct(logits_image, labels)
            loss = loss_fuse + loss_text + loss_image
            outputs = (loss,) + outputs

        return outputs

# %% [markdown]
# # Evaluation & Metrics Helpers

# %%
def evaluate_acc_f1(
    params,
    model,
    device,
    data,
    tokenizer,
    processor,
    average='binary',
    pre=None,
    mode='test',
):
    data_loader = DataLoader(
        data,
        batch_size=params.dev_batch_size,
        collate_fn=MMSD2_id_dataset.collate_func,
        shuffle=False
    )
    n_correct, n_total = 0, 0
    t_targets_all, t_outputs_all = None, None

    model.eval()
    sum_loss = 0.0
    sum_step = 0

    with torch.no_grad():
        for i_batch, t_batch in enumerate(data_loader):
            text_list, image_list, label_list, id_list = t_batch
            text_inputs = tokenizer(
                text_list,
                padding='max_length',
                truncation=True,
                max_length=params.max_len,
                return_tensors="pt"
            ).to(device)
            image_inputs = processor(images=image_list, return_tensors="pt").to(device)
            labels = torch.tensor(label_list, dtype=torch.long).to(device)

            loss, t_outputs = model(
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask'],
                pixel_values=image_inputs['pixel_values'],
                labels=labels
            )
            sum_loss += loss.item()
            sum_step += 1

            outputs = torch.argmax(t_outputs, -1)

            n_correct += (outputs == labels).sum().item()
            n_total += len(outputs)

            if t_targets_all is None:
                t_targets_all = labels
                t_outputs_all = outputs
            else:
                t_targets_all = torch.cat((t_targets_all, labels), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, outputs), dim=0)

    avg_loss = sum_loss / max(sum_step, 1)
    if mode == 'test':
        print(f"Test Loss: {avg_loss:.4f}")
    else:
        print(f"Dev Loss: {avg_loss:.4f}")

    if pre is not None:
        with open(pre, 'w', encoding='utf-8') as fout:
            predict = t_outputs_all.cpu().numpy().tolist()
            label = t_targets_all.cpu().numpy().tolist()
            for pred_val, true_val in zip(predict, label):
                fout.write(f"{pred_val}\t{true_val}\n")

    acc = n_correct / max(n_total, 1)
    f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), average=average, zero_division=0)
    precision = metrics.precision_score(t_targets_all.cpu(), t_outputs_all.cpu(), average=average, zero_division=0)
    recall = metrics.recall_score(t_targets_all.cpu(), t_outputs_all.cpu(), average=average, zero_division=0)

    return avg_loss, acc, f1, precision, recall

# %% [markdown]
# # Initialization & Training Setup

# %%
tokenizer = AutoTokenizer.from_pretrained(params.text_model_name)
processor = AutoProcessor.from_pretrained(params.vision_model_name)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=params.train_batch_size,
    collate_fn=MMSD2_id_dataset.collate_func,
    shuffle=True,
    generator=torch.Generator().manual_seed(params.seed),
)

model = SarcasmModel(params).to(device)
if params.freeze_encoder_epochs > 0:
    model.set_encoders_trainable(False)
    print(f"Pretrained encoders frozen for the first {params.freeze_encoder_epochs} epochs")

encoder_parameters = list(model.multilingual_text_model.parameters())
encoder_parameters += list(model.vision_encoder.parameters())
encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
task_parameters = [
    parameter
    for parameter in model.parameters()
    if id(parameter) not in encoder_parameter_ids
]

# Fused Adam avoids the large update temporaries created by the foreach and
# single-tensor implementations for these two large encoders.
optimizer = torch.optim.Adam(
    [
        {
            "params": encoder_parameters,
            "lr": params.encoder_learning_rate,
            "group_name": "pretrained_encoders",
        },
        {
            "params": task_parameters,
            "lr": params.task_learning_rate,
            "group_name": "fusion_and_classifiers",
        },
    ],
    fused=device.type == "cuda",
)
print(
    f"Learning rates -> encoders: {params.encoder_learning_rate:.2e}, "
    f"fusion/classifiers: {params.task_learning_rate:.2e}"
)


def wait_and_reserve_vram(
    device,
    target_reserved_gib,
    safety_margin_gib,
    poll_interval_seconds,
):
    """Wait until this process can reserve the expected unfrozen peak VRAM."""
    if device.type != "cuda":
        return

    gib = 1024**3
    target_bytes = int(target_reserved_gib * gib)
    margin_bytes = int(safety_margin_gib * gib)
    chunk_bytes = 256 * 1024**2

    while True:
        torch.cuda.synchronize(device)
        currently_reserved = torch.cuda.memory_reserved(device)
        additional_needed = max(target_bytes - currently_reserved, 0)
        free_bytes, _ = torch.cuda.mem_get_info(device)

        print(
            f"GPU memory: {free_bytes / gib:.2f} GiB free, "
            f"{currently_reserved / gib:.2f} GiB reserved by this process, "
            f"{additional_needed / gib:.2f} GiB more required"
        )

        if additional_needed == 0:
            print("Required VRAM is already reserved.")
            return

        if free_bytes >= additional_needed + margin_bytes:
            reservation_chunks = []
            try:
                remaining = additional_needed
                while remaining > 0:
                    allocation_size = min(remaining, chunk_bytes)
                    reservation_chunks.append(
                        torch.empty(
                            allocation_size,
                            dtype=torch.uint8,
                            device=device,
                        )
                    )
                    remaining -= allocation_size

                torch.cuda.synchronize(device)
                print(
                    f"Successfully reserved "
                    f"{torch.cuda.memory_reserved(device) / gib:.2f} GiB"
                )

                # Release the tensors but keep their blocks in this process's
                # CUDA cache so the unfrozen training step can reuse them.
                del reservation_chunks
                torch.cuda.synchronize(device)
                return
            except torch.OutOfMemoryError:
                print(
                    "VRAM became unavailable during reservation; "
                    "waiting before retrying."
                )
                del reservation_chunks
                torch.cuda.empty_cache()

        print(f"Retrying in {poll_interval_seconds} seconds...")
        time.sleep(poll_interval_seconds)

# %% [markdown]
# # Training Loop

# %%
max_acc = float('-inf')
run_output_dir = Path(params.output_dir) / params.model
run_output_dir.mkdir(parents=True, exist_ok=True)
wandb.login()
wandb_run = wandb.init(
    project=params.wandb_project,
    name=params.model,
    job_type="train",
    save_code=True,
    config={
        "text_model_name": params.text_model_name,
        "vision_model_name": params.vision_model_name,
        "num_train_epochs": int(params.num_train_epochs),
        "freeze_encoder_epochs": int(params.freeze_encoder_epochs),
        "vram_reservation_target_gib": float(params.vram_reservation_target_gib),
        "vram_safety_margin_gib": float(params.vram_safety_margin_gib),
        "vram_poll_interval_seconds": int(params.vram_poll_interval_seconds),
        "train_batch_size": int(params.train_batch_size),
        "dev_batch_size": int(params.dev_batch_size),
        "max_len": int(params.max_len),
        "encoder_learning_rate": float(params.encoder_learning_rate),
        "task_learning_rate": float(params.task_learning_rate),
        "fusion_layers": int(params.layers),
        "dropout_rate": float(params.dropout_rate),
        "seed": int(params.seed),
    },
)
wandb.define_metric("epoch")
wandb.define_metric("train/*", step_metric="epoch")
wandb.define_metric("validation/*", step_metric="epoch")
wandb.define_metric("optimization/*", step_metric="epoch")

for i_epoch in trange(0, int(params.num_train_epochs), desc="Epoch", disable=False):
    encoders_should_train = i_epoch >= params.freeze_encoder_epochs
    if encoders_should_train == model.encoders_frozen:
        if encoders_should_train:
            print("Waiting for enough VRAM before unfreezing encoders...")
            wait_and_reserve_vram(
                device=device,
                target_reserved_gib=params.vram_reservation_target_gib,
                safety_margin_gib=params.vram_safety_margin_gib,
                poll_interval_seconds=params.vram_poll_interval_seconds,
            )
        model.set_encoders_trainable(encoders_should_train)
        encoder_state = "unfrozen" if encoders_should_train else "frozen"
        print(f"Pretrained encoders are now {encoder_state} at epoch {i_epoch + 1}")

    sum_loss = 0.0
    sum_step = 0
    train_targets_all = []
    train_outputs_all = []

    iter_bar = tqdm(train_loader, desc=f"Epoch {i_epoch} Iter", disable=False)
    model.train()

    for step, batch in enumerate(iter_bar):
        text_list, image_list, label_list, id_list = batch

        text_inputs = tokenizer(
            text_list,
            padding='max_length',
            truncation=True,
            max_length=params.max_len,
            return_tensors="pt"
        ).to(device)
        image_inputs = processor(images=image_list, return_tensors="pt").to(device)
        labels = torch.tensor(label_list, dtype=torch.long).to(device)

        loss, score = model(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            pixel_values=image_inputs['pixel_values'],
            labels=labels
        )

        sum_loss += loss.item()
        sum_step += 1
        train_targets_all.append(labels.detach().cpu())
        train_outputs_all.append(torch.argmax(score.detach(), dim=-1).cpu())

        iter_bar.set_description(f"Epoch {i_epoch} Loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_train_loss = sum_loss / max(sum_step, 1)
    train_targets_all = torch.cat(train_targets_all)
    train_outputs_all = torch.cat(train_outputs_all)
    train_acc = metrics.accuracy_score(train_targets_all, train_outputs_all)
    train_f1 = metrics.f1_score(
        train_targets_all, train_outputs_all, average='binary', zero_division=0
    )
    train_precision = metrics.precision_score(
        train_targets_all, train_outputs_all, average='binary', zero_division=0
    )
    train_recall = metrics.recall_score(
        train_targets_all, train_outputs_all, average='binary', zero_division=0
    )
    print(f"\n--- Epoch {i_epoch} Summary ---")
    print(
        f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Train F1: {train_f1:.4f} | Train Precision: {train_precision:.4f} | "
        f"Train Recall: {train_recall:.4f}"
    )

    dev_loss, dev_acc, dev_f1, dev_precision, dev_recall = evaluate_acc_f1(
        params, model, device, val_dataset, tokenizer, processor, mode='dev'
    )
    print(f"Dev Loss: {dev_loss:.4f} | Dev Acc: {dev_acc:.4f} | Dev F1: {dev_f1:.4f} | Dev Precision: {dev_precision:.4f} | Dev Recall: {dev_recall:.4f}")

    # Select and save checkpoints using validation data only.
    is_best = dev_acc > max_acc
    if is_best:
        max_acc = dev_acc
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), run_output_dir / 'model.pt')
        print(f"[*] New best validation accuracy! Saved model checkpoint to {run_output_dir}")

    wandb_run.log({
        "epoch": i_epoch + 1,
        "train/loss": float(avg_train_loss),
        "train/accuracy": float(train_acc),
        "train/f1": float(train_f1),
        "train/precision": float(train_precision),
        "train/recall": float(train_recall),
        "validation/loss": float(dev_loss),
        "validation/accuracy": float(dev_acc),
        "validation/f1": float(dev_f1),
        "validation/precision": float(dev_precision),
        "validation/recall": float(dev_recall),
        "optimization/encoder_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "optimization/task_learning_rate": float(optimizer.param_groups[1]["lr"]),
        "optimization/encoders_frozen": int(model.encoders_frozen),
        "checkpoint/is_best": int(is_best),
    })

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Training is finished, so release optimizer state and any tensors retained by
# the final batch before loading the best checkpoint for test evaluation.
# In particular, loading a checkpoint directly onto `device` would briefly keep
# a second full copy of the model weights in VRAM.
model.zero_grad(set_to_none=True)
optimizer.zero_grad(set_to_none=True)
del optimizer

for variable_name in (
    "batch",
    "labels",
    "score",
    "loss",
    "text_inputs",
    "image_inputs",
    "train_targets_all",
    "train_outputs_all",
    "iter_bar",
):
    globals().pop(variable_name, None)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(
        "GPU memory after training cleanup: "
        f"{torch.cuda.memory_allocated(device) / 1024**3:.2f} GiB allocated, "
        f"{torch.cuda.memory_reserved(device) / 1024**3:.2f} GiB reserved"
    )

# Evaluate the test set once, using the checkpoint selected on validation data.
best_model_path = run_output_dir / 'model.pt'
if best_model_path.exists():
    checkpoint_state = torch.load(best_model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint_state)
    del checkpoint_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    test_loss, test_acc, test_f1, test_precision, test_recall = evaluate_acc_f1(
        params, model, device, test_dataset, tokenizer, processor, average='macro', mode='test'
    )
    _, _, test_f1_micro, test_precision_micro, test_recall_micro = evaluate_acc_f1(
        params, model, device, test_dataset, tokenizer, processor, average='micro', mode='test'
    )
    print(f"Macro Test -> Acc: {test_acc:.4f} | F1: {test_f1:.4f} | Prec: {test_precision:.4f} | Rec: {test_recall:.4f}")
    print(f"Micro Test -> F1: {test_f1_micro:.4f} | Prec: {test_precision_micro:.4f} | Rec: {test_recall_micro:.4f}")
    wandb_run.log({
        "test/loss": float(test_loss),
        "test/accuracy": float(test_acc),
        "test/macro_f1": float(test_f1),
        "test/macro_precision": float(test_precision),
        "test/macro_recall": float(test_recall),
        "test/micro_f1": float(test_f1_micro),
        "test/micro_precision": float(test_precision_micro),
        "test/micro_recall": float(test_recall_micro),
    })
    wandb_run.summary["best_validation_accuracy"] = float(max_acc)
    wandb_run.summary["test_accuracy"] = float(test_acc)
    wandb_run.summary["test_macro_f1"] = float(test_f1)
    wandb_run.summary["test_micro_f1"] = float(test_f1_micro)

    model_artifact = wandb.Artifact(
        name=f"{params.model}-best",
        type="model",
        description="Best checkpoint selected by validation accuracy",
        metadata={
            "best_validation_accuracy": float(max_acc),
            "text_model_name": params.text_model_name,
            "vision_model_name": params.vision_model_name,
        },
    )
    model_artifact.add_file(str(best_model_path), name="model.pt")
    wandb_run.log_artifact(model_artifact, aliases=["best"])
else:
    print("No best checkpoint was saved; skipping test evaluation.")

wandb_run.finish()

# %%
