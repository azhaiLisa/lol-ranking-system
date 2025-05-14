import torch
from torch.utils.data import Dataset, DataLoader, random_split
import transformers
print("Transformers version:", transformers.__version__)
from transformers import PreTrainedTokenizerFast, AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from datasets import Dataset as HFDataset
import json
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from tokenizers.processors import TemplateProcessing

# Step 1: Build a Tokenizer
def build_tokenizer(token_sequences: List[List[str]], vocab_size: int = 10000):
    """Build a tokenizer from the token sequences using the tokenizers library."""
    # Initialize a new tokenizer
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    
    # Configure the tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit()
    ])
    
    # Add special tokens
    special_tokens = ["[GAME_START]", "[GAME_END]", "[PAD]", "[UNK]"] + [f"[RANK_{tier}]" for tier in [
        "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"
    ]]

    
    # Train the tokenizer
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    
    # Train on the sequences
    tokenizer.train_from_iterator(token_sequences, trainer=trainer)
    
    # Add post-processing for special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="[GAME_START] $A [GAME_END]",
        special_tokens=[
            ("[GAME_START]", tokenizer.token_to_id("[GAME_START]")),
            ("[GAME_END]", tokenizer.token_to_id("[GAME_END]"))
        ]
    )
    
    # Convert to PreTrainedTokenizerFast
    pretrained_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[GAME_START]",
        eos_token="[GAME_END]",
        pad_token="[PAD]",
        unk_token="[UNK]"
    )
    
    return pretrained_tokenizer

# Step 2: Convert Token Sequences to IDs
def prepare_data(token_sequences: List[List[str]], tokenizer: PreTrainedTokenizerFast):
    """Convert token sequences to input_ids and attention masks."""
    encoded_sequences = []
    
    for sequence in tqdm(token_sequences, desc="Encoding sequences"):
        # Tokenize the sequence
        encoded = tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=1024, # gpt2 limit, may switch to gptj later
            return_tensors="pt"
        )
        encoded_sequences.append({
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0]
        })
    
    return encoded_sequences

# Step 3: Create Dataset
class LeagueDataset(Dataset):
    def __init__(self, encoded_sequences: List[Dict]):
        self.sequences = encoded_sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        data = self.sequences[idx]
        labels = data["input_ids"].clone()
        # Set padding tokens to -100 to ignore them in loss calculation
        labels[data["attention_mask"] == 0] = -100
        return {
            "input_ids": data["input_ids"],
            "attention_mask": data["attention_mask"],
            "labels": labels
        }

# Step 4: Define Transformer Model
def create_model(tokenizer: PreTrainedTokenizerFast, max_length: int = 512):
    """Create a transformer model for sequence prediction."""
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Resize token embeddings to match our vocabulary
    model.resize_token_embeddings(len(tokenizer))
    
    # Set the pad token ID in the model config
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model

def main():
    # Load processed tokens
    print("Loading processed tokens...")
    with open("../data/processed_tokens_updated.txt", "r") as f:
        # Split each line into tokens
        token_sequences = [line.strip().split() for line in f]
    
    # Step 1: Build tokenizer
    print("Building tokenizer...")
    tokenizer = build_tokenizer(token_sequences)
    tokenizer.save_pretrained("league_tokenizer")
    
    # Step 2: Convert sequences to IDs
    print("Converting sequences to IDs...")
    encoded_sequences = prepare_data(token_sequences, tokenizer)
    
    # Step 3: Create dataset and split
    print("Creating dataset...")
    dataset = LeagueDataset(encoded_sequences)
    
    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Step 4: Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False
    )
    
    # Step 5: Create model
    print("Creating model...")
    model = create_model(tokenizer)
    
    # Step 6: Training arguments
    training_args = TrainingArguments(
        output_dir="./league_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print("Saving model...")
    trainer.save_model("./league_model_final")
    tokenizer.save_pretrained("./league_model_final")

if __name__ == "__main__":
    main() 