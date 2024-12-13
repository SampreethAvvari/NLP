import torch
from transformers import T5Tokenizer, T5ForSequenceClassification, Trainer, TrainingArguments, AdamW, get_cosine_schedule_with_warmup, DataCollatorWithPadding
from datasets import load_dataset
import evaluate
import wandb
import numpy as np

# Initialize Weights & Biases for tracking
wandb.init(
    project="Dravida Ulagam - Small",
    name="t5-small-classification-AGNews-10",
    config={
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 10,
        "eval_steps": 5000,
        "checkpoint_steps": 5000,
        "gradient_accumulation_steps": 1,
        "dropout": 0.05,
        "scheduler": "cosine",
        "optimizer": "AdamW"
    }
)

# Load AG News dataset
dataset = load_dataset("ag_news")

# Load the T5-small tokenizer and model for sequence classification
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForSequenceClassification.from_pretrained(
    "t5-small",
    num_labels=4,
    problem_type="single_label_classification"
)

# T5 uses a different padding token
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Tokenize dataset
def preprocess(examples):
    texts = [f"classify text: {text}" for text in examples["text"]]

    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors=None,
    )

    tokenized["labels"] = examples["label"]
    return tokenized

# Apply preprocessing with batching
tokenized_dataset = dataset.map(
    preprocess,
    batched=True,
    batch_size=1000,
    remove_columns=dataset["train"].column_names
)

# Create data collator for dynamic padding
class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        if "labels" in batch:
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
        return batch

data_collator = CustomDataCollator(tokenizer=tokenizer)

# Initialize metrics using evaluate
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

# Custom compute_metrics function for Trainer
def compute_metrics(eval_pred):
    # Handle the complex output structure
    if isinstance(eval_pred.predictions, tuple):
        # If predictions is a tuple, take the first element (logits)
        logits = eval_pred.predictions[0]
    else:
        logits = eval_pred.predictions

    # Reshape if needed and get predictions
    if len(logits.shape) == 3:
        # If shape is (batch_size, sequence_length, num_labels)
        # Take the prediction for the first token
        logits = logits[:, 0, :]

    predictions = np.argmax(logits, axis=-1)
    labels = eval_pred.label_ids

    # Compute metrics
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
# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=5000,
    logging_steps=5000,
    save_steps=5000,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_dir='./logs',
    report_to="wandb",
    gradient_accumulation_steps=1,
    remove_unused_columns=True,
    include_inputs_for_metrics=True,
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
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler)
)

# Train the model
trainer.train()

# Close WandB run
wandb.finish()
