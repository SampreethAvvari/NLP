import wandb
import torch
from transformers import (
    DebertaForSequenceClassification,
    DebertaTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import evaluate

wandb.login(key = '19a6c0405fc04811550b523713c570a29af7e7d9')

# Initialize wandb
wandb.init(
    project="Dravida Ulagam - Large",
    name="deberta-xlarge-classification-AGNews"
)

# Load dataset
dataset = load_dataset("ag_news")
tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-xlarge")

# Tokenize dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Prepare data for PyTorch
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format("torch")

train_dataset = encoded_dataset["train"]
test_dataset = encoded_dataset["test"]

# Load the model
model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-xlarge", num_labels=4)

# Define metrics using evaluate
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision_metric.compute(predictions=predictions, references=labels, average="macro")["precision"],
        "recall": recall_metric.compute(predictions=predictions, references=labels, average="macro")["recall"],
        "f1": f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"],
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=2500,
    save_steps=2500,
    logging_dir="./logs",
    logging_steps=2500,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,  # Effectively doubling batch size
    num_train_epochs=3,
    learning_rate=1e-4,
    warmup_steps=500,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="wandb",  # Log metrics to wandb
    fp16=True,  # Mixed precision for faster training
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train and evaluate
trainer.train()
trainer.evaluate()

# Finish wandb
wandb.finish()

