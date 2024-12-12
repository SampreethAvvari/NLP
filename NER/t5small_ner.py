import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
import wandb

# Log in to WandB
wandb.login(key="19a6c0405fc04811550b523713c570a29af7e7d9")

# Initialize WandB logging
wandb.init(project="Dravida Ulagam - Small", name='t5small-ner-oct31-changed_hyp_run2', config={
    "model": "t5-small",
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 5,
    "weight_decay": 0.01,
    "max_length": 128,
    "lr_scheduler": "cosine"
})

# Load the CoNLL-2003 dataset
dataset = load_dataset('conll2003')
label_names = dataset['train'].features['ner_tags'].feature.names

# Define the checkpoint for tokenizer and model
checkpoint = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Helper function to format input for T5
def format_example(example):
    input_text = " ".join(example["tokens"])
    entities = []

    for token, label in zip(example["tokens"], example["ner_tags"]):
        if label != 0:  # Only include entities
            entities.append(f"{token} ({label_names[label]})")

    target_text = ", ".join(entities) if entities else "No entities"
    return {"input_text": input_text, "target_text": target_text}

# Apply the format function
formatted_dataset = dataset.map(format_example, remove_columns=dataset['train'].column_names)

# Tokenization function for T5
def tokenize_fn(batch):
    inputs = tokenizer(batch["input_text"], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(batch["target_text"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

# Tokenize the dataset
tokenized_dataset = formatted_dataset.map(tokenize_fn, batched=True)

# Metric for evaluation
metric = evaluate.load("seqeval")

# Track convergence based on epoch-wise loss stability
convergence_epoch = None
plateau_threshold = 0.001
previous_f1 = None

# Helper function to parse and normalize the output format
def parse_ner_output(output, max_len=128):
    # Split by commas to separate entities and labels
    entities = output.split(", ")
    token_labels = []

    for entity in entities:
        if " (" in entity and entity.endswith(")"):
            token, label = entity.rsplit(" (", 1)
            label = label[:-1]  # Remove closing parenthesis
            token_labels.append(label)
        else:
            token_labels.append("O")  # Use "O" for no entity

    # Ensure the parsed output has a fixed length, padding with "O" if necessary
    if len(token_labels) < max_len:
        token_labels.extend(["O"] * (max_len - len(token_labels)))
    elif len(token_labels) > max_len:
        token_labels = token_labels[:max_len]

    return token_labels

# Updated compute_metrics function
def compute_metrics(eval_pred):
    global convergence_epoch, previous_f1

    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Apply parsing function to predictions and labels with consistent length
    parsed_preds = [parse_ner_output(pred, max_len=128) for pred in decoded_preds]
    parsed_labels = [parse_ner_output(label, max_len=128) for label in decoded_labels]

    # Compute seqeval metrics
    results = metric.compute(predictions=parsed_preds, references=parsed_labels)
    precision = results["overall_precision"]
    recall = results["overall_recall"]
    f1 = results["overall_f1"]
    accuracy = results["overall_accuracy"]

    # Log metrics to WandB
    wandb.log({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    })

    # Track convergence rate
    if previous_f1 is not None and abs(previous_f1 - f1) < plateau_threshold and convergence_epoch is None:
        convergence_epoch = trainer.state.epoch
    previous_f1 = f1

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

   
# Define training arguments with the updated hyperparameters
training_args = Seq2SeqTrainingArguments(
    output_dir="/scratch/ds7395/nlp_proj/t5_small_ner_changed_hyperparameters_run2",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=32,  # Reduce to 8 if memory issues occur
    per_device_eval_batch_size=32,
    optim="paged_adamw_32bit",
    num_train_epochs=5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    predict_with_generate=True,  # Enables generation for evaluation in Seq2Seq tasks
    save_strategy="epoch",  # Save checkpoints every epoch
    logging_strategy="steps",
    logging_steps=1, 
    gradient_accumulation_steps=8
)

# Initialize Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Log Convergence Rate to WandB after training
wandb.log({
    "convergence_epoch": convergence_epoch if convergence_epoch is not None else None
})
wandb.finish()
