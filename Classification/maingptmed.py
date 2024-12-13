import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments, AdamW, get_cosine_schedule_with_warmup
from datasets import load_dataset
import evaluate
import wandb
import numpy as np

def main():
    # Initialize Weights & Biases for tracking
    wandb.login(key = '19a6c0405fc04811550b523713c570a29af7e7d9')

    # Start a new Weights & Biases run
    wandb.init(
        project="Dravida Ulagam - Medium",
        name="gpt2-medium-classification-AGNews-2",
        config={
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 5,
            "eval_steps": 25,
            "checkpoint_steps": 25,
            "gradient_accumulation_steps": 1,
            "dropout": 0.05,
            "scheduler": "cosine",
            "optimizer": "AdamW"
        }
    )

    # Load AG News dataset
    dataset = load_dataset("ag_news")

    # Load the GPT-2 Medium tokenizer and model for sequence classification
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2ForSequenceClassification.from_pretrained("gpt2-medium", num_labels=4)
    model.config.pad_token_id = tokenizer.eos_token_id  # Set the padding token

    # Tokenize dataset
    def preprocess(example):
        return tokenizer(
            example["text"],
            padding="max_length",
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
    # Convert logits to predictions
    predictions = torch.argmax(torch.tensor(logits), axis=-1).numpy()
    # Convert labels to NumPy array
    references = np.array(labels)
    # Compute metrics
    accuracy = accuracy_metric.compute(predictions=predictions, references=references)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=references, average="weighted")["f1"]
    precision = precision_metric.compute(predictions=predictions, references=references, average="weighted")["precision"]
    recall = recall_metric.compute(predictions=predictions, references=references, average="weighted")["recall"]
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
        eval_steps=1000,
        logging_steps=1000,
        save_steps=1000,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
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


if __name__ == "__main__":
    main()

