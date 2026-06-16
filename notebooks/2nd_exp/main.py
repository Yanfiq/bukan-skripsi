# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: 2nd-exp (3.12.11.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dataset

# %%
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset
import os
from PIL import Image
from pathlib import Path

# %%
cwd = Path.cwd()
dataset_dir = cwd / "dataset"

# %%
df = pd.read_json("./dataset/text_json_id/dataset_translated_fixed.json", orient="records", dtype={"image_id": str, "label": int}).set_index("image_id")
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
class MMSD2_id_dataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.to_dict(orient="index")
        self.image_ids=list(self.data.keys())
        for id in self.data.keys():
            self.data[id]["image_path"] = dataset_dir / "dataset_image" / f"{id}.jpg"

    def image_loader(self,id):
        print(f"Loading image from: {self.data[id]['image_path']}")
        return Image.open(self.data[id]["image_path"])
    
    def text_loader(self,id):
        return self.data[id]["text_translated"]

    def __getitem__(self, index):
        id=self.image_ids[index]
        print(f"Loading data for ID: {id} (index: {index})")
        text = self.text_loader(id)
        image_feature = self.image_loader(id)
        label = self.data[id]["label"]
        return text,image_feature, label, id

    def __len__(self):
        return len(self.image_ids)
    @staticmethod
    def collate_func(batch_data):
        batch_size = len(batch_data)
 
        if batch_size == 0:
            return {}

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
# ### JALANIN INI KALO MAU CUSTOM SPLIT

# %%
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# %%
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
train_dataset = MMSD2_id_dataset(df.iloc[train_dataset_subset.indices])
val_dataset = MMSD2_id_dataset(df.iloc[val_dataset_subset.indices])
test_dataset = MMSD2_id_dataset(df.iloc[test_dataset_subset.indices])

# %% [markdown]
# ### JALANIN INI KALO MAU DEFAULT SPLIT NGIKUT UPSTREAM

# %%
train_dataset = MMSD2_id_dataset(df[df["split"] == "train"])
val_dataset = MMSD2_id_dataset(df[df["split"] == "valid"])
test_dataset = MMSD2_id_dataset(df[df["split"] == "test"])

# %% [markdown]
# # Model

# %%
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoProcessor, AutoModel, AlbertTokenizer, AlbertModel
from torchinfo import summary
from torch import nn
from transformers import CLIPModel,BertConfig
from transformers.models.bert.modeling_bert import BertLayer
import copy

# %%
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")


# %%
# torch.cuda.set_device(1)

# %%
class BlankObject:
    pass


# %%
# text_model_name = "indobenchmark/indobert-base-p2"
# # vision_model_name = "WinKawaks/vit-small-patch16-224"
# vision_model_name = "openai/clip-vit-base-patch32"

params = BlankObject()
params.text_model_name = "indobenchmark/indobert-base-p2"
params.vision_model_name = "openai/clip-vit-base-patch32"
params.device = 1
params.simple_linear = False
params.layers = 3
params.text_size = 512
params.image_size = 768
params.dropout_rate = 0.1
params.label_count = 2
params.num_train_epochs = 10
params.train_batch_size = 1
params.dev_batch_size = 1
params.max_len = 77


# %%
class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_attentions = []
        for layer_module in self.layer:
            hidden_states, attention = layer_module(hidden_states, attention_mask, output_attentions=True)
            all_encoder_attentions.append(attention)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_attentions
    
class SarcasmModel(nn.Module):
    def __init__(self, params):
        super(SarcasmModel, self).__init__()
        self.model = CLIPModel.from_pretrained(params.vision_model_name)
        self.config = BertConfig.from_pretrained(params.text_model_name)
        self.config.hidden_size = 512
        self.config.num_attention_heads = 8
        self.trans = MultimodalEncoder(self.config, layer_number=params.layers)
        if params.simple_linear:
            self.text_linear =  nn.Linear(params.text_size, params.text_size)
            self.image_linear =  nn.Linear(params.image_size, params.image_size)
        else:
            self.text_linear =  nn.Sequential(
                nn.Linear(params.text_size, params.text_size),
                nn.Dropout(params.dropout_rate),
                nn.GELU()
            )
            self.image_linear =  nn.Sequential(
                nn.Linear(params.image_size, params.image_size),
                nn.Dropout(params.dropout_rate),
                nn.GELU()
            )

        self.classifier_fuse = nn.Linear(params.text_size , params.label_count)
        self.classifier_text = nn.Linear(params.text_size, params.label_count)
        self.classifier_image = nn.Linear(params.image_size, params.label_count)

        self.loss_fct = nn.CrossEntropyLoss()
        self.att = nn.Linear(params.text_size, 1, bias=False)

    def forward(self, inputs, labels):
        output = self.model(**inputs,output_attentions=True)
        text_features = output['text_model_output']['last_hidden_state']
        image_features = output['vision_model_output']['last_hidden_state']
        text_feature = output['text_model_output']['pooler_output']
        image_feature = output['vision_model_output']['pooler_output']
        text_feature = self.text_linear(text_feature)
        image_feature = self.image_linear(image_feature)

        text_embeds = self.model.text_projection(text_features)
        image_embeds = self.model.visual_projection(image_features)
        input_embeds = torch.cat((image_embeds, text_embeds), dim=1)
        attention_mask = torch.cat((torch.ones(text_features.shape[0], 50).to(text_features.device), inputs['attention_mask']), dim=-1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        fuse_hiddens, all_attentions = self.trans(input_embeds, extended_attention_mask, output_all_encoded_layers=False)
        fuse_hiddens = fuse_hiddens[-1]
        new_text_features = fuse_hiddens[:, 50:, :]
        new_text_feature = new_text_features[
            torch.arange(new_text_features.shape[0], device=inputs['input_ids'].device), inputs['input_ids'].to(torch.int).argmax(dim=-1)
        ]

        new_image_feature = fuse_hiddens[:, 0, :].squeeze(1)

        text_weight = self.att(new_text_feature)
        image_weight = self.att(new_image_feature)    
        att = nn.functional.softmax(torch.stack((text_weight, image_weight), dim=-1),dim=-1)
        tw, iw = att.split([1,1], dim=-1)
        fuse_feature = tw.squeeze(1) * new_text_feature + iw.squeeze(1) * new_image_feature

        logits_fuse = self.classifier_fuse(fuse_feature)
        logits_text = self.classifier_text(text_feature)
        logits_image = self.classifier_image(image_feature)
   
        fuse_score = nn.functional.softmax(logits_fuse, dim=-1)
        text_score = nn.functional.softmax(logits_text, dim=-1)
        image_score = nn.functional.softmax(logits_image, dim=-1)

        score = fuse_score + text_score + image_score

        outputs = (score,)
        if labels is not None:
            loss_fuse = self.loss_fct(logits_fuse, labels)
            loss_text = self.loss_fct(logits_text, labels)
            loss_image = self.loss_fct(logits_image, labels)
            loss = loss_fuse + loss_text + loss_image

            outputs = (loss,) + outputs
        return outputs  


# %%
def evaluate_acc_f1(params, model, device, data, processor, macro=False,pre = None, mode='test'):
        data_loader = DataLoader(data, batch_size=params.dev_batch_size, collate_fn=MMSD2_id_dataset.collate_func,shuffle=False)
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None

        model.eval()
        sum_loss = 0.
        sum_step = 0
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                text_list, image_list, label_list, id_list = t_batch
                inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=params.max_len, return_tensors="pt").to(device)
                labels = torch.tensor(label_list).to(device)
                
                t_targets = labels
                loss, t_outputs = model(inputs,labels=labels)
                sum_loss += loss.item()
                sum_step += 1
  
                outputs = torch.argmax(t_outputs, -1)

                n_correct += (outputs == t_targets).sum().item()
                n_total += len(outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, outputs), dim=0)
        if mode == 'test':
            # wandb.log({'test_loss': sum_loss/sum_step})
            print(f"Test Loss: {sum_loss/sum_step}")
        else:
            # wandb.log({'dev_loss': sum_loss/sum_step})
            print(f"Dev Loss: {sum_loss/sum_step}")
        if pre != None:
            with open(pre,'w',encoding='utf-8') as fout:
                predict = t_outputs_all.cpu().numpy().tolist()
                label = t_targets_all.cpu().numpy().tolist()
                for x,y,z in zip(predict,label):
                    fout.write(str(x) + str(y) +z+ '\n')
        if not macro:   
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu())
            precision =  metrics.precision_score(t_targets_all.cpu(),t_outputs_all.cpu())
            recall = metrics.recall_score(t_targets_all.cpu(),t_outputs_all.cpu())
        else:
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), labels=[0, 1],average='macro')
            precision =  metrics.precision_score(t_targets_all.cpu(),t_outputs_all.cpu(), labels=[0, 1],average='macro')
            recall = metrics.recall_score(t_targets_all.cpu(),t_outputs_all.cpu(), labels=[0, 1],average='macro')
        return acc, f1 ,precision,recall


# %% [markdown]
# # Train

# %%
from tqdm import tqdm, trange

# %%
train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              collate_fn=MMSD2_id_dataset.collate_func,
                              shuffle=True)

model = SarcasmModel(params).to(device)

# %%
clip_params = list(map(id, model.model.parameters()))
base_params = filter(lambda p: id(p) not in clip_params, model.parameters())
# optimizer = ada([
#         {"params": base_params},
#         {"params": model.model.parameters(),"lr": args.clip_learning_rate}
#         ], lr=args.learning_rate, weight_decay=args.weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
#                                         num_training_steps=total_steps)
processor = AutoProcessor.from_pretrained(params.vision_model_name)

# %%
# 1. See what the model expects for dimensions
print(f"Expected size: {processor.image_processor.size}") 

# 2. See normalization values (Mean and Std)
print(f"Mean: {processor.image_processor.image_mean}")
print(f"Std: {processor.image_processor.image_std}")

# 3. Check the output shape with a dummy image (e.g., a random tensor)
import torch
dummy_image = torch.randn(3, 500, 500) # Simulating a 500x500 RGB image
# Rescale dummy_image to [0,1], convert to PIL, then process
img = dummy_image.clone().cpu()
img = (img - img.min()) / (img.max() - img.min() + 1e-8)
np_img = (img.permute(1, 2, 0).mul(255).to(torch.uint8).numpy())
pil_img = Image.fromarray(np_img)
pixel_values = processor(images=pil_img, return_tensors="pt")["pixel_values"]

print(f"Processed image shape: {pixel_values.shape}")
# Likely [1, 3, 224, 224]

# %%
max_acc = 0.
for i_epoch in trange(0, int(params.num_train_epochs), desc="Epoch", disable=False):
    sum_loss = 0.
    sum_step = 0

    iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
    model.train()

    for step, batch in enumerate(iter_bar):
        text_list, image_list, label_list, id_list = batch
        inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=params.max_len, return_tensors="pt").to(device)
        labels = torch.tensor(label_list).to(device)

        loss, score = model(inputs,labels=labels)
        sum_loss += loss.item()
        sum_step += 1

        iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
        loss.backward()
        optimizer.step()
        if params.optimizer_name == 'adam':
            scheduler.step() 
        optimizer.zero_grad()

    print(f"Epoch {i_epoch} completed. Average Loss: {sum_loss/sum_step:.4f}")
    # wandb.log({'train_loss': sum_loss/sum_step})

    dev_acc, dev_f1 ,dev_precision,dev_recall = evaluate_acc_f1(params, model, device, dev_data, processor, mode='dev')
    print(f"Epoch {i_epoch} completed. Dev Acc: {dev_acc:.4f}, Dev F1: {dev_f1:.4f}, Dev Precision: {dev_precision:.4f}, Dev Recall: {dev_recall:.4f}")
    # wandb.log({'dev_acc': dev_acc, 'dev_f1': dev_f1, 'dev_precision': dev_precision, 'dev_recall': dev_recall})
    # logging.info("i_epoch is {}, dev_acc is {}, dev_f1 is {}, dev_precision is {}, dev_recall is {}".format(i_epoch, dev_acc, dev_f1, dev_precision, dev_recall))

    if dev_acc > max_acc:
        max_acc = dev_acc

        path_to_save = os.path.join(params.output_dir, params.model)
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
        model_to_save = (model.module if hasattr(model, "module") else model)
        torch.save(model_to_save.state_dict(), os.path.join(path_to_save, 'model.pt'))

        test_acc, test_f1,test_precision,test_recall = evaluate_acc_f1(params, model, device, test_data, processor,macro = True, mode='test')
        _, test_f1_,test_precision_,test_recall_ = evaluate_acc_f1(params, model, device, test_data, processor, mode='test')
        # wandb.log({'test_acc': test_acc, 'macro_test_f1': test_f1,
        #             'macro_test_precision': test_precision,'macro_test_recall': test_recall, 'micro_test_f1': test_f1_,
        #             'micro_test_precision': test_precision_,'micro_test_recall': test_recall_})
        print(f"Epoch {i_epoch} completed. Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")
        print(f"Epoch {i_epoch} completed. Micro Test F1: {test_f1_:.4f}, Micro Test Precision: {test_precision_:.4f}, Micro Test Recall: {test_recall_:.4f}")

    torch.cuda.empty_cache()

# %%
