import wandb
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Initialize wandb
wandb.init(project="Dravida Ulagam - Small", name="distilbert-sick-sentencesimilarity-nov1-run1")

# Load the SICK dataset
dataset = load_dataset("sick", trust_remote_code=True)  # This may require the dataset name in 'datasets'

# Initialize the tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=1)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params}")

# Preprocess the dataset
def preprocess(examples):
    return tokenizer(examples["sentence_A"], examples["sentence_B"], padding="max_length", truncation=True)

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset = encoded_dataset.map(lambda x: {"labels": x["relatedness_score"]}, batched=True)

# Define evaluation metrics
pearson = evaluate.load("pearsonr")
spearman = evaluate.load("spearmanr")

# Define compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    pearson_corr = pearson.compute(predictions=predictions, references=labels)["pearsonr"]
    spearman_corr = spearman.compute(predictions=predictions, references=labels)["spearmanr"]
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "mse": mse,
        "mae": mae,
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir="./distilbert-sick-results",
    evaluation_strategy="steps",
    eval_steps=25,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    logging_dir="./distilbert-sick-logs",
    logging_steps=25,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    weight_decay=0.01,
    save_steps=25,
    load_best_model_at_end=True,
    lr_scheduler_type="cosine",
    report_to="wandb",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics
)

# Train and evaluate
trainer.train()
trainer.evaluate()

wandb.finish()