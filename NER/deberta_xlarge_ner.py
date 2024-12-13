
import numpy as np 
import pandas as pd 
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
import evaluate
import wandb


# Define the checkpoint for tokenizer

wandb.login(key="")

# Initialize WandB logging
wandb.init(project="Dravida Ulagam - Large", name='deberata_xlarge_nov14_run1', config={
    "model": "xlm-roberta-base",
    "batch_size": 8,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-5,
    "epochs": 5,
    "weight_decay": 0.01
})
# Load the CoNLL-2003 dataset
dataset = load_dataset('conll2003')
label_names = dataset['train'].features['ner_tags'].feature.names

# Define the checkpoint for tokenizer
checkpoint = 'microsoft/deberta-xlarge'
tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True)


# Tokenize the first example
token = tokenizer(dataset['train'][0]['tokens'], is_split_into_words=True)

# Align target labels to handle sub-tokens
def align_target(labels, word_ids):
    begin2inside = {1: 2, 3: 4, 5: 6, 7: 8}  # Mapping B- labels to I- labels
    align_labels = []
    last_word = None

    for word in word_ids:
        if word is None:
            label = -100
        elif word != last_word:
            label = labels[word]
        else:
            label = labels[word]
            if label in begin2inside:
                label = begin2inside[label]
        align_labels.append(label)
        last_word = word
    return align_labels

# Tokenization function for the dataset
def tokenize_fn(batch):
    tokenized_inputs = tokenizer(batch['tokens'], truncation=True, is_split_into_words=True)
    labels_batch = batch['ner_tags']
    aligned_targets_batch = []

    for i, labels in enumerate(labels_batch):
        word_ids = tokenized_inputs.word_ids(i)
        aligned_targets_batch.append(align_target(labels, word_ids))

    tokenized_inputs["labels"] = aligned_targets_batch
    return tokenized_inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset['train'].column_names)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Metric for evaluation
metric = evaluate.load("seqeval")


# Track convergence based on epoch-wise loss stability
convergence_epoch = None
plateau_threshold = 0.001
previous_f1 = None

# Compute metrics function with WandB logging
def compute_metrics(logits_and_labels):
    global convergence_epoch, previous_f1

    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis=-1)

    str_labels = [
        [label_names[t] for t in label if t != -100] for label in labels
    ]
    str_preds = [
        [label_names[p] for (p, t) in zip(prediction, label) if t != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=str_preds, references=str_labels)
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

# Label mappings
id2label = {k: v for k, v in enumerate(label_names)}
label2id = {v: k for k, v in enumerate(label_names)}

# Load the model
model = AutoModelForTokenClassification.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id
)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params}")
# Define training arguments
training_args = TrainingArguments(
    output_dir="/scratch/ds7395/nlp_proj/deberta_xlarge_nov14_run1",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Log Convergence Rate to WandB after training
wandb.log({
    "convergence_epoch": convergence_epoch if convergence_epoch is not None else None
})
wandb.finish()

