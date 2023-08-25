# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:17:54 2023

@author: mikkel
"""
import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from pdfminer.high_level import extract_text
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score
import sys
import os
# sys.path.append(os.path.abspath('C:/DIWorkpoint'))

# owned imports
from ocr.extractText import *

run_name = "/runname"
folder = "datafolder"

# Function to extract text from pdf
def extract_text_from_pdf(file_path):
    # text = extract_text(file_path)
    text = extractText(file_path)
    return text

# Assuming the labels are the names of the folders
labels = os.listdir('datafolder/files')
label_to_id = {label: id for id, label in enumerate(labels)}

# Prepare the data
data = []
for label in labels:
    for file_path in glob.glob(f'datafolder/files/{label}/*.pdf'):
        text = extract_text_from_pdf(file_path)
        data.append({'text': text, 'label': label_to_id[label]})

# Convert the data to a DataFrame
df = pd.DataFrame(data)

#%%
# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2)

# Load the tokenizer and the model
model_name = "Maltehb/danish-bert-botxo" # This is a BERT model trained on danish data
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(labels), 
    gradient_checkpointing=True   # enable gradient checkpointing
)

# Tokenize the inputs
train_encodings = tokenizer(train_texts.to_list(), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts.to_list(), truncation=True, padding=True, max_length=512)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

# Convert the inputs to PyTorch format
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create train and val dataset
train_dataset = Dataset(train_encodings, train_labels.tolist())
val_dataset = Dataset(val_encodings, val_labels.tolist())

# Define the training arguments
training_args = TrainingArguments(
    output_dir=folder + '/models'+run_name,
    num_train_epochs=5,
    per_device_train_batch_size=8,   # reduce batch size?
    per_device_eval_batch_size=32,   # reduce batch size?
    gradient_accumulation_steps=2,   # enable gradient accumulation
    warmup_steps=0,
    weight_decay=0.01,
    logging_dir=folder + '/logs'+run_name,
    logging_steps=1,
)

# Create the trainer and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics

)

# Start training
print('Start training...')
trainer.train()
print('Training complete.')

# Compute accuracy on the validation dataset
metrics = trainer.evaluate()
print(f"Accuracy: {metrics['eval_accuracy']}")

# Save the model and tokenizer
trainer.save_model(folder + '/models'+run_name)  # or another directory
tokenizer.save_pretrained(folder + '/models'+run_name)  # or another directory