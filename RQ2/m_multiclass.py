#!/usr/bin/env python
# coding: utf-8

import torch
import pandas as pd
import numpy as np
import datasets
import transformers
import torch.distributed as dist
import random
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from IPython.display import display, HTML
from datasets import Dataset, DatasetDict
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedType
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model_checkpoint = "deepseek-ai/deepseek-coder-1.3b-instruct"
shortname = "ds5_1"

train_df = pd.read_json(f"dataset/train_rq3_1.json")
val_df = pd.read_json(f"dataset/test_rq3_1.json")
all_df = pd.read_json(f"dataset/all_rq3_1.json")

train_df = train_df.drop(['label', 'year', 'source','hash', 'length_category'], axis=1)
val_df = val_df.drop(['label', 'year', 'source','hash', 'length_category'], axis=1)
all_df = all_df.drop(['label', 'year', 'source','hash', 'length_category'], axis=1)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
all_df = all_df.reset_index(drop=True)

train_df = train_df.rename(columns={'cwe': 'label'})
val_df = val_df.rename(columns={'cwe': 'label'})
all_df = all_df.rename(columns={'cwe': 'label'})

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(val_df)
all_dataset = Dataset.from_pandas(all_df)

# Get the unique labels and the number of labels
all_labels = np.unique(all_dataset['label'])
num_labels = len(all_labels)
print("Number of unique labels:", num_labels)

# Map labels to integer values
label_map = {label: i for i, label in enumerate(all_labels)}
train_dataset = train_dataset.map(lambda x: {'labels': label_map[x['label']]})
train_dataset = train_dataset.remove_columns('label')
valid_dataset = valid_dataset.map(lambda x: {'labels': label_map[x['label']]})
valid_dataset = valid_dataset.remove_columns('label')


raw_datasets = DatasetDict({
    "train": train_dataset,
    "validation": valid_dataset
})

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    outputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=4020)
    return outputs

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_datasets["train"].features
tokenized_datasets.set_format("torch")
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
model.resize_token_embeddings(len(tokenizer))

def create_dataloaders(train_batch_size=1, eval_batch_size=1):
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=train_batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, batch_size=eval_batch_size
    )
    return train_dataloader, eval_dataloader


train_dataloader, eval_dataloader = create_dataloaders()

hyperparameters = {
    "learning_rate": 2e-5,
    "num_epochs": 10,
    "train_batch_size": 1, 
    "eval_batch_size": 1, 
    "seed": 42,
}

def training_function(model):
    accelerator = Accelerator()

    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    train_dataloader, eval_dataloader = create_dataloaders(
        train_batch_size=hyperparameters["train_batch_size"], eval_batch_size=hyperparameters["eval_batch_size"]
    )
    set_seed(hyperparameters["seed"])
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=hyperparameters["learning_rate"])

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_epochs = hyperparameters["num_epochs"]
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataloader) * num_epochs,
    )

    progress_bar = tqdm(range(num_epochs * len(train_dataloader)), disable=not accelerator.is_main_process)
    # Now we train the model
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        model.eval()
        all_predictions = []
        all_labels = []

        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)

                all_predictions.append(accelerator.gather(predictions))
                all_labels.append(accelerator.gather(batch["labels"]))

        all_predictions = torch.cat(all_predictions)[:len(tokenized_datasets["validation"])]
        all_labels = torch.cat(all_labels)[:len(tokenized_datasets["validation"])]

        print(f"Length of all_predictions: {len(all_predictions)}")
        print(f"Length of all_labels: {len(all_labels)}")
        torch.save(all_predictions, f"predictions_{shortname}_{epoch}.pt")
        torch.save(all_labels, f"labels_{shortname}_{epoch}.pt")

        # Compute evaluation metrics
        accuracy = accuracy_score(all_labels.cpu().numpy(), all_predictions.cpu().numpy())
        precision = precision_score(all_labels.cpu().numpy(), all_predictions.cpu().numpy(), average='macro', zero_division=0)
        recall = recall_score(all_labels.cpu().numpy(), all_predictions.cpu().numpy(), average='macro', zero_division=0)
        f1 = f1_score(all_labels.cpu().numpy(), all_predictions.cpu().numpy(), average='macro')

        # Print evaluation results
        accelerator.print(f"epoch {epoch}:")
        accelerator.print(f"Accuracy: {accuracy:.4f}")
        accelerator.print(f"Precision (macro): {precision:.4f}")
        accelerator.print(f"Recall (macro): {recall:.4f}")
        accelerator.print(f"F1 Score (macro): {f1:.4f}")
        if dist.get_rank() == 0:
            model.module.save_pretrained(f"{shortname}_model_{epoch}")
            tokenizer.save_pretrained(f"{shortname}_model_{epoch}")

training_function(model)
