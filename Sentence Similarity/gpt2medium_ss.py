from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
from torch.utils.data import Dataset
import wandb
wandb.init(project="Dravida Ulagam - medium", name="gpt2medium-sick-sentencesimilarity-nov28-run2-test")

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token
model = GPT2ForSequenceClassification.from_pretrained("openai-community/gpt2-medium", num_labels=6)  # 6 classes (0-5)

# Custom Dataset Class for PyTorch
class SICKDataset(Dataset):
    def _init_(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _len_(self):
        return len(self.dataset)

    def _getitem_(self, idx):
        example = self.dataset[idx]
        sentence_A = example["sentence_A"]
        sentence_B = example["sentence_B"]
        label = round(example["relatedness_score"])  # Round score to integer for classification

        # Tokenize input
        encoded = self.tokenizer(
            f"Sentence 1: {sentence_A}\nSentence 2: {sentence_B}",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# Load the SICK dataset
dataset = load_dataset("sick")  # Automatically downloads and prepares the dataset

# Split into training and validation sets
train_data = SICKDataset(dataset["train"], tokenizer)
val_data = SICKDataset(dataset["validation"], tokenizer)

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./gpt2medium-sick-results-3",
    evaluation_strategy="steps",
    eval_steps=25,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    logging_dir="./gpt2medium-sick-logs-3",
    logging_steps=25,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    weight_decay=0.01,
    save_steps=25,
    load_best_model_at_end=True,
    lr_scheduler_type="cosine",
    report_to="wandb",  # Log metrics to WandB
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Save the model
model.save_pretrained("./gpt2medium-sick-final")
tokenizer.save_pretrained("./gpt2medium-sick-final")