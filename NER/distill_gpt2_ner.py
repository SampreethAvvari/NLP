import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    GPT2TokenizerFast,
    GPT2ForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
import evaluate
import wandb


# Log in to WandB
wandb.login(key="")  # You'll be prompted to enter your API key or you can set it via environment variable

# Initialize WandB logging
wandb.init(project="Dravida Ulagam - Small", name='distillgpt2-ner-nov1-run2_changed_lr_scheduler_10epochs', config={
    "model": "distill-gpt2",  # Change to "t5-base" or "t5-large" if resources allow
    "batch_size": 16,
    "learning_rate": 5e-5,
    "epochs": 8,
    "weight_decay": 0.01,
    "max_length": 512,
    "lr_scheduler": "cosine"
})


# Load the CoNLL-2003 dataset
dataset = load_dataset('conll2003')

# Get the list of labels
label_list = dataset["train"].features["ner_tags"].feature.names

# Load the tokenizer and model
checkpoint = "distilgpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(checkpoint, add_prefix_space=True)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token, so we set it to eos_token

model = GPT2ForTokenClassification.from_pretrained(
    checkpoint, num_labels=len(label_list)
)

num_parameters = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_parameters}")

# Align the labels with the tokenized inputs
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        padding="max_length",
        truncation=True,
        max_length=512,
        is_split_into_words=True,
        return_offsets_mapping=True,
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their word IDs
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word_id of None
            if word_idx is None:
                label_ids.append(-100)
            # Only label the first token of a word, and set the rest to -100
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Apply the tokenization and alignment
tokenized_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
)


# Define the data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Load the metric
metric = evaluate.load("seqeval")


# Define the compute_metrics function
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [label_list[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    true_labels = [
        [label_list[lab] for (pred, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = metric.compute(predictions=true_predictions, references=true_labels)
    # Extract overall metrics
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
    


    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir="/scratch/ds7395/nlp_proj/distilgpt2-ner-output_test_newlr_10epochs",
    evaluation_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_strategy="steps",
    logging_steps=25,
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="precision",
    greater_is_better=True,
    seed=42,
    gradient_accumulation_steps=2
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Evaluate the model
trainer.evaluate()
