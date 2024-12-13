import wandb
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Initialize wandb
wandb.init(project="Dravida Ulagam - Small", name="t5small-sick-sentencesimilarity-nov1-run1")

# Load the SICK dataset
dataset = load_dataset("sick")

# Initialize the tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params}")

# Preprocess the dataset
def preprocess(examples):
    # Encode inputs and create a target string from the relatedness score
    inputs = [f"sentence similarity: {s1} </s> {s2}" for s1, s2 in zip(examples["sentence_A"], examples["sentence_B"])]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    model_inputs["labels"] = tokenizer(
        [str(score) for score in examples["relatedness_score"]], padding="max_length", truncation=True, max_length=10
    )["input_ids"]
    return model_inputs

encoded_dataset = dataset.map(preprocess, batched=True)

# Define evaluation metrics
pearson = evaluate.load("pearsonr")
spearman = evaluate.load("spearmanr")

# Define compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode predictions and labels to get the relatedness scores
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Convert predictions and labels to floats for metric computation
    decoded_preds = [float(pred) for pred in decoded_preds]
    decoded_labels = [float(label) for label in decoded_labels]
    
    pearson_corr = pearson.compute(predictions=decoded_preds, references=decoded_labels)["pearsonr"]
    spearman_corr = spearman.compute(predictions=decoded_preds, references=decoded_labels)["spearmanr"]
    mse = mean_squared_error(decoded_labels, decoded_preds)
    mae = mean_absolute_error(decoded_labels, decoded_preds)
    
    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "mse": mse,
        "mae": mae,
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir="./t5small-sick-results",
    evaluation_strategy="steps",
    eval_steps=25,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    logging_dir="./t5small-sick-logs",
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
    #compute_metrics=compute_metrics
)

# Train and evaluate
trainer.train()
trainer.evaluate()

wandb.finish()