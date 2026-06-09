# %% [markdown]
# # MMSD2.0 Training (Adapted from datasets/MMSD2.0/src)

# %%
# %pip install transformers torch tqdm scikit-learn wandb pillow

import os
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import wandb
from PIL import ImageFile
from transformers import CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

# %%
# Path setup
cwd = Path.cwd()
project_root_dir = cwd.parents[1]
dataset_root_dir = project_root_dir / "datasets"

# Modifiable
DATASET_NAME = "MMSD2.0"
DEVICE_ID = "0"  # e.g. "0", "1", or "-1" for CPU
#

dataset_dir = dataset_root_dir / DATASET_NAME
dataset_data_dir = dataset_dir / "data"
dataset_src_dir = dataset_dir / "src"

# Create symlink for notebook convenience
project_dataset_dir = cwd / "dataset"
if not project_dataset_dir.exists():
    project_dataset_dir.symlink_to(dataset_data_dir)

# Allow importing modules from datasets/MMSD2.0/src
if dataset_src_dir.as_posix() not in sys.path:
    sys.path.insert(0, dataset_src_dir.as_posix())

# %%
# Import source modules (adapted from datasets/MMSD2.0/src/main.py)
import importlib

data_set_module = importlib.import_module("data_set")
model_module = importlib.import_module("model")
train_module = importlib.import_module("train")

MyDataset = data_set_module.MyDataset
MV_CLIP = model_module.MV_CLIP
train = train_module.train

# Fix relative path in data_set.py so notebook cwd works
# Source file uses WORKING_PATH="../data" expecting execution from src folder.
data_set_module.WORKING_PATH = dataset_data_dir.as_posix()

# %%
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def make_args() -> Namespace:
    """Notebook-friendly replacement for argparse in src/main.py."""
    return Namespace(
        device=DEVICE_ID,
        model="MV_CLIP",
        text_name="text_json_final",
        simple_linear=False,
        num_train_epochs=10,
        train_batch_size=32,
        dev_batch_size=32,
        label_number=2,
        text_size=512,
        image_size=768,
        adam_epsilon=1e-8,
        optimizer_name="adam",
        learning_rate=5e-4,
        clip_learning_rate=1e-6,
        max_len=77,
        layers=3,
        max_grad_norm=5.0,
        weight_decay=0.05,
        warmup_proportion=0.2,
        dropout_rate=0.1,
        output_dir=(cwd / "output_dir").as_posix(),
        limit=None,
        seed=42,
    )


# %%
# Configure runtime (adapted from src/main.py)
args = make_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
print(f"Using device: {device}")

seed_everything(args.seed)

# %%
# W&B init (disable by default in notebook to avoid auth interruption)
USE_WANDB = False

wandb.init(
    project="MMSD2.0",
    notes="adapted-notebook-run",
    tags=["mm", "notebook", "src-adaptation"],
    config=vars(args),
    mode="online" if USE_WANDB else "disabled",
)
wandb.watch_called = False

# %%
# Dataset loading (adapted from src/main.py + src/data_set.py)
train_data = MyDataset(mode="train", text_name=args.text_name, limit=args.limit)
dev_data = MyDataset(mode="valid", text_name=args.text_name, limit=args.limit)
test_data = MyDataset(mode="test", text_name=args.text_name, limit=args.limit)

print(f"Train size: {len(train_data)}")
print(f"Valid size: {len(dev_data)}")
print(f"Test size:  {len(test_data)}")

# %%
# Model + processor (adapted from src/main.py)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = MV_CLIP(args)
model.to(device)

wandb.watch(model, log="all")

# %%
# Train + evaluate (source implementation from src/train.py)
train(
    args=args,
    model=model,
    device=device,
    train_data=train_data,
    dev_data=dev_data,
    test_data=test_data,
    processor=processor,
)

# %%
print("Training pipeline completed.")
