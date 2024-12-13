import wandb
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments, DebertaTokenizer, DebertaForSequenceClassification

from datasets import load_dataset
import evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


wandb.init(project="Dravida Ulagam - Large", name="dberta-sick-sentencesimilarity-nov28-run1")

# Load the SICK dataset
dataset = load_dataset("sick", trust_remote_code=True)  # This may require the dataset name in 'datasets'

model_name = "microsoft/deberta-xlarge"
tokenizer = DebertaTokenizer.from_pretrained(model_name)
model = DebertaForSequenceClassification.from_pretrained(model_name, num_labels=1)



total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params}")

# Preprocess the dataset
def preprocess(examples):
    return tokenizer(examples["sentence_A"], examples["sentence_B"], padding="max_length", max_length=512,truncation=True)

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset = encoded_dataset.map(lambda x: {"labels": x["relatedness_score"]}, batched=True)

# Define evaluation metrics
pearson = evaluate.load("pearsonr")
spearman = evaluate.load("spearmanr")

# Define compute_metrics function
import wandb

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Convert predictions and labels to flat arrays
    if isinstance(predictions, list):
        predictions = [item for sublist in predictions for item in sublist]  # Flatten nested lists
    if isinstance(labels, list):
        labels = [item for sublist in labels for item in sublist]  # Flatten nested lists

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Convert decoded strings to float for regression comparison
    decoded_preds = np.array([float(pred) for pred in decoded_preds])
    decoded_labels = np.array([float(label) for label in decoded_labels])
    
    # Compute evaluation metrics
    pearson_corr = pearson.compute(predictions=decoded_preds, references=decoded_labels)["pearsonr"]
    spearman_corr = spearman.compute(predictions=decoded_preds, references=decoded_labels)["spearmanr"]
    mse = mean_squared_error(decoded_labels, decoded_preds)
    mae = mean_absolute_error(decoded_labels, decoded_preds)
    
    metrics = {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "mse": mse,
        "mae": mae,
    }
    
    # Log metrics to wandb
    wandb.log(metrics)
    
    return metrics


# Define training arguments
training_args = TrainingArguments(
    output_dir="./dberta-result-run1",
    evaluation_strategy="steps",
    eval_steps=25,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    logging_dir="./dberta-result-run1",
    logging_steps=25,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    weight_decay=0.01,
    save_steps=1000,
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