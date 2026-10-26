# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # W&B Multilingual MV-CLIP Evaluation
#
# Download a training run's history and best model artifact, evaluate it on the
# MMSD2.0 test split, and generate diagnostic plots and prediction files.

# %%
"""Evaluate a multilingual MV-CLIP checkpoint logged to Weights & Biases."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoTokenizer, BertConfig, CLIPVisionModelWithProjection
from transformers.models.bert.modeling_bert import BertLayer


# %% [markdown]
# ## Training-history fields

# %%
HISTORY_KEYS = [
    "epoch",
    "train/loss",
    "train/accuracy",
    "train/f1",
    "train/precision",
    "train/recall",
    "validation/loss",
    "validation/accuracy",
    "validation/f1",
    "validation/precision",
    "validation/recall",
]


# %% [markdown]
# ## Dataset

# %%
class MMSD2Dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, dataset_dir: Path):
        self.rows = dataframe.reset_index().to_dict(orient="records")
        self.image_dir = dataset_dir / "dataset_image"

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_id = str(row["image_id"])
        image = Image.open(self.image_dir / f"{image_id}.jpg").convert("RGB")
        return row["text"], image, int(row["label"]), image_id

    @staticmethod
    def collate(batch):
        texts, images, labels, image_ids = zip(*batch)
        return list(texts), list(images), list(labels), list(image_ids)


# %% [markdown]
# ## Model architecture

# %%
class MultimodalEncoder(nn.Module):
    def __init__(self, config: BertConfig, layer_number: int):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList(copy.deepcopy(layer) for _ in range(layer_number))

    def forward(self, hidden_states, attention_mask):
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask, output_attentions=False)[0]
        return hidden_states


class SarcasmModel(nn.Module):
    """Architecture used by the 3rd experiment's Mean/Mean and Mean/CLS runs."""

    def __init__(self, args: SimpleNamespace, variant: str):
        super().__init__()
        self.variant = variant
        self.multilingual_text_model = SentenceTransformer(args.text_model_name)
        self.text_encoder = self.multilingual_text_model[0].auto_model
        self.text_projection = self.multilingual_text_model[2]
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(args.vision_model_name)

        config = BertConfig.from_pretrained("bert-base-uncased")
        config.hidden_size = args.fuse_dim
        config.num_attention_heads = 8
        self.trans = MultimodalEncoder(config, args.layers)
        self.text_linear = nn.Sequential(
            nn.Linear(args.text_size, args.text_size), nn.Dropout(args.dropout_rate), nn.GELU()
        )
        self.image_linear = nn.Sequential(
            nn.Linear(args.image_size, args.image_size), nn.Dropout(args.dropout_rate), nn.GELU()
        )
        self.classifier_fuse = nn.Linear(args.text_size, args.label_number)
        self.classifier_text = nn.Linear(args.text_size, args.label_number)
        self.classifier_image = nn.Linear(args.image_size, args.label_number)
        self.att = nn.Linear(args.text_size, 1, bias=False)

    def forward(self, input_ids, attention_mask, pixel_values):
        text_hidden = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
        vision_hidden = self.vision_encoder(
            pixel_values=pixel_values, output_attentions=False, return_dict=True
        ).last_hidden_state

        text_mask = attention_mask.unsqueeze(-1).to(text_hidden.dtype)
        text_pooled = (text_hidden * text_mask).sum(1) / text_mask.sum(1).clamp(min=1e-9)
        text_feature = self.text_linear(
            self.text_projection({"sentence_embedding": text_pooled})["sentence_embedding"]
        )
        image_pooled = self.vision_encoder.vision_model.post_layernorm(vision_hidden[:, 0, :])
        image_feature = self.image_linear(image_pooled)

        text_embeds = self.text_projection.activation_function(
            self.text_projection.linear(text_hidden)
        )
        image_embeds = self.vision_encoder.visual_projection(vision_hidden)
        image_token_count = image_embeds.shape[1]
        image_mask = torch.ones(
            (input_ids.shape[0], image_token_count),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        combined_mask = torch.cat((image_mask, attention_mask), dim=1)
        extended_mask = combined_mask[:, None, None, :].to(text_embeds.dtype)
        extended_mask = (1.0 - extended_mask) * -10000.0
        fused = self.trans(torch.cat((image_embeds, text_embeds), dim=1), extended_mask)

        new_image_feature = fused[:, 0, :]
        fused_text = fused[:, image_token_count:, :]
        if self.variant == "mean_cls":
            new_text_feature = fused_text[:, 0, :]
        else:
            new_text_feature = (fused_text * text_mask).sum(1) / text_mask.sum(1).clamp(min=1e-9)

        weights = F.softmax(
            torch.cat((self.att(new_text_feature), self.att(new_image_feature)), dim=-1),
            dim=-1,
        )
        fuse_feature = weights[:, :1] * new_text_feature + weights[:, 1:] * new_image_feature
        # Training sums the three softmax outputs. Divide by three here so the
        # values are true probabilities without changing the predicted class.
        return (
            F.softmax(self.classifier_fuse(fuse_feature), dim=-1)
            + F.softmax(self.classifier_text(text_feature), dim=-1)
            + F.softmax(self.classifier_image(image_feature), dim=-1)
        ) / 3.0


# %% [markdown]
# ## Command-line configuration

# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-path", required=True, help="W&B entity/project/run_id")
    parser.add_argument("--artifact-name", help="Optional artifact name with alias/version")
    parser.add_argument("--variant", choices=("mean_mean", "mean_cls"), default="mean_mean")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser.parse_args()


# %% [markdown]
# ## Dataset discovery

# %%
def find_dataset_dir(explicit: Path | None) -> Path:
    if explicit:
        return explicit.expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "datasets" / "MMSD2.0" / "data"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not find datasets/MMSD2.0/data; pass --dataset-dir")


# %% [markdown]
# ## W&B history retrieval and visualization

# %%
def fetch_history(run, output_dir: Path) -> pd.DataFrame:
    history = run.history(keys=HISTORY_KEYS, pandas=True, samples=100_000)
    history = history.dropna(how="all", subset=[key for key in HISTORY_KEYS if key != "epoch"])
    if "epoch" in history:
        history = history.sort_values("epoch").drop_duplicates("epoch", keep="last")
    history.to_csv(output_dir / "training_history.csv", index=False)
    return history


def plot_history(history: pd.DataFrame, output_path: Path) -> None:
    groups = (("loss", "Loss"), ("accuracy", "Accuracy"), ("f1", "F1"),
              ("precision", "Precision"), ("recall", "Recall"))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    x = history["epoch"] if "epoch" in history else np.arange(1, len(history) + 1)
    for axis, (metric, title) in zip(axes.flat, groups):
        for split, label in (("train", "Train"), ("validation", "Validation")):
            key = f"{split}/{metric}"
            if key in history and history[key].notna().any():
                axis.plot(x, history[key], marker="o", label=label)
        axis.set(title=title, xlabel="Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    axes.flat[-1].axis("off")
    fig.suptitle("Training history", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# %% [markdown]
# ## Model-artifact retrieval

# %%
def select_artifact(run, artifact_name: str | None):
    if artifact_name:
        # A fully-qualified name works across projects; a short name is scoped
        # to the run's entity/project.
        if artifact_name.count("/") < 2:
            artifact_name = f"{run.entity}/{run.project}/{artifact_name}"
        return wandb.Api().artifact(artifact_name, type="model")
    artifacts = [artifact for artifact in run.logged_artifacts() if artifact.type == "model"]
    if not artifacts:
        raise RuntimeError(f"Run {run.path} has no logged model artifact")
    best = [artifact for artifact in artifacts if "best" in artifact.aliases]
    return (best or artifacts)[-1]


# %% [markdown]
# ## Test-set inference

# %%
def infer(model, loader, tokenizer, processor, device, max_len: int) -> pd.DataFrame:
    records = []
    model.eval()
    with torch.inference_mode():
        for texts, images, labels, image_ids in tqdm(loader, desc="Test inference"):
            text_inputs = tokenizer(
                texts, padding="max_length", truncation=True,
                max_length=max_len, return_tensors="pt"
            ).to(device)
            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            probabilities = model(
                text_inputs["input_ids"], text_inputs["attention_mask"], pixel_values
            ).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            for image_id, label, prediction, probability in zip(
                image_ids, labels, predictions, probabilities
            ):
                records.append({
                    "image_id": image_id,
                    "label": label,
                    "prediction": int(prediction),
                    "probability_class_0": float(probability[0]),
                    "probability_class_1": float(probability[1]),
                })
    return pd.DataFrame(records)


# %% [markdown]
# ## Metrics and evaluation plots

# %%
def save_evaluation(predictions: pd.DataFrame, output_dir: Path) -> None:
    y_true = predictions["label"].to_numpy()
    y_pred = predictions["prediction"].to_numpy()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    results = {
        "samples": int(len(predictions)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": report,
    }
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set(xlabel="Predicted label", ylabel="True label", title="Test confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    bins = np.linspace(0, 1, 21)
    for label, axis in enumerate(axes):
        subset = predictions[predictions["label"] == label]
        axis.hist(
            subset["probability_class_1"], bins=bins, alpha=0.8,
            edgecolor="white", color=("#4C78A8" if label == 0 else "#E45756")
        )
        axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set(title=f"True class {label} (n={len(subset)})", xlabel="P(class 1)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Sample count")
    fig.suptitle("Predicted probability distribution by true class", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "probability_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({key: value for key, value in results.items() if key != "classification_report"}, indent=2))


# %% [markdown]
# ## Evaluation pipeline
#
# The function remains callable from Python. When opening this file as a
# notebook, set arguments in the CLI cell or call the individual helpers.

# %%
def main() -> None:
    cli = parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = find_dataset_dir(cli.dataset_dir)
    json_path = dataset_dir / "text_json_id" / "dataset_translated_fixed.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {json_path}")

    print(f"Reading W&B run {cli.run_path}")
    api = wandb.Api()
    run = api.run(cli.run_path)
    history = fetch_history(run, cli.output_dir)
    plot_history(history, cli.output_dir / "training_history.png")

    artifact = select_artifact(run, cli.artifact_name)
    artifact_dir = Path(artifact.download(root=str(cli.output_dir / "artifact")))
    checkpoint_path = artifact_dir / "model.pt"
    if not checkpoint_path.is_file():
        matches = list(artifact_dir.rglob("*.pt"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one .pt checkpoint in {artifact_dir}")
        checkpoint_path = matches[0]

    config = dict(run.config)
    args = SimpleNamespace(
        text_model_name=config.get(
            "text_model_name", "sentence-transformers/clip-ViT-B-32-multilingual-v1"
        ),
        vision_model_name=config.get("vision_model_name", "openai/clip-vit-base-patch32"),
        fuse_dim=int(config.get("fuse_dim", 512)),
        text_size=int(config.get("text_size", 512)),
        image_size=int(config.get("image_size", 768)),
        layers=int(config.get("fusion_layers", config.get("layers", 3))),
        dropout_rate=float(config.get("dropout_rate", 0.1)),
        label_number=int(config.get("label_number", 2)),
    )
    device_name = cli.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Loading {artifact.name} on {device}")
    model = SarcasmModel(args, cli.variant)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)

    dataframe = pd.read_json(
        json_path, orient="records", dtype={"image_id": str, "label": int}
    ).set_index("image_id")
    test_dataset = MMSD2Dataset(dataframe[dataframe["split"] == "test"], dataset_dir)
    loader = DataLoader(
        test_dataset,
        batch_size=cli.batch_size or int(config.get("dev_batch_size", 32)),
        shuffle=False,
        num_workers=cli.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=MMSD2Dataset.collate,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    processor = AutoProcessor.from_pretrained(args.vision_model_name)
    predictions = infer(
        model, loader, tokenizer, processor, device, int(config.get("max_len", 77))
    )
    save_evaluation(predictions, cli.output_dir)
    print(f"Evaluation files written to {cli.output_dir.resolve()}")


# %% [markdown]
# ## Script entry point

# %%
if __name__ == "__main__":
    main()

