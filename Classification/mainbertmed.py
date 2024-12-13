import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments, AdamW, get_cosine_schedule_with_warmup
from datasets import load_dataset
import evaluate
import wandb

# Initialize Weights & Biases for tracking
wandb.init(
    project="Dravida Ulagam - Medium",
    name="xlm-roberta-classification-AGNews-7",
    config={
        "learning_rate": 1e-4,
        "batch_size": 16,
        "epochs": 5,
        "eval_steps": 2500,
        "checkpoint_steps": 2500,
        "gradient_accumulation_steps": 1,
        "dropout": 0.05,
        "scheduler": "cosine",
        "optimizer": "AdamW"
    }
)

# Load AG News dataset
dataset = load_dataset("ag_news")

# Load the XLM-Roberta tokenizer and model for sequence classification
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=4)

# Tokenize dataset
def preprocess(example):
    return tokenizer(
        example["text"],
        padding=True,
        truncation=True,
        max_length=512
    )

tokenized_dataset = dataset.map(preprocess, batched=True)

# Initialize metrics using evaluate
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

# Custom compute_metrics function for Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"]
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=2500,
    logging_steps=2500,
    save_steps=2500,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_dir='./logs',
    report_to="wandb",
    gradient_accumulation_steps=1,
    fp16=True
)

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=1e-4)
num_training_steps = len(tokenized_dataset["train"]) * training_args.num_train_epochs // training_args.per_device_train_batch_size
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler)
)

# Train the model
trainer.train()

# Close WandB run
wandb.finish()

