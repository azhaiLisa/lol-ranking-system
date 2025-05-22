# Training from scratch with custom tokenizer using GPTNeoXForCausalLM

from transformers import PreTrainedTokenizerFast, GPTNeoXConfig, GPTNeoXForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import json
import matplotlib.pyplot as plt

# Load your custom tokenizer
hf_tokenizer = PreTrainedTokenizerFast.from_pretrained("./my_tokenizer")
hf_tokenizer.model_max_length = 8192
# hf_tokenizer.pad_token = hf_tokenizer.eos_token  # Set pad token if needed

# Define a new model config for scratch training
config = GPTNeoXConfig(
    vocab_size=len(hf_tokenizer),
    hidden_size=256,  # Small model size for RTX 2060
    intermediate_size=1024,
    num_hidden_layers=6,
    num_attention_heads=8,
    max_position_embeddings=8192,
    rotary_pct=1.0,
    tie_word_embeddings=False,
    pad_token_id=hf_tokenizer.pad_token_id
)

model = GPTNeoXForCausalLM(config)

# Dataset loading (one full match per line)
dataset = load_dataset("text", data_files={"train": "../data/processed_tokens.txt"})
split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Tokenize full matches with truncation to 8192
def tokenize(batch):
    return hf_tokenizer(batch["text"], truncation=True, padding=False, max_length=8192)

tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized_eval = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=hf_tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./model_checkpoints_neox_10epochs",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    eval_strategy="epoch",
    # eval_steps=5000,
    save_strategy="epoch",
    learning_rate=5e-4,
    weight_decay=0.01,
    prediction_loss_only=True,
    fp16=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# Train
trainer.train()
trainer.save_model("./lol-model-neox")
hf_tokenizer.save_pretrained("./lol-model-neox")

# Plot training curve
logs = sorted(trainer.state.log_history, key=lambda x: x.get("step", -1))
train_steps, train_losses = [], []
eval_steps, eval_losses = [], []
for entry in logs:
    if "loss" in entry and "step" in entry:
        train_steps.append(entry["step"])
        train_losses.append(entry["loss"])
    if "eval_loss" in entry and "step" in entry:
        eval_steps.append(entry["step"])
        eval_losses.append(entry["eval_loss"])

plt.figure(figsize=(10,5))
plt.plot(train_steps, train_losses, label="Training Loss")
plt.plot(eval_steps, eval_losses, label="Validation Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.title("Training and Validation Loss 10 Epochs")
plt.tight_layout()
plt.savefig("loss_plot_10epochs.png")
plt.show()

# Save logs
with open("train_logs_scratch.json", "w") as f:
    json.dump(trainer.state.log_history, f)


with open("../data/train_tokens.txt", "w") as f:
    for ex in train_dataset:
        f.write(ex["text"] + "\n")

with open("../data/eval_tokens.txt", "w") as f:
    for ex in eval_dataset:
        f.write(ex["text"] + "\n")