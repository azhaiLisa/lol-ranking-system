# Updated evaluation script for non-sliding-window match input

import torch
import torch.nn.functional as F
from datasets import load_dataset
import random
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerFast
import matplotlib.pyplot as plt
import numpy as np

# Load dataset and model
eval_dataset = load_dataset("text", data_files={"validation": "../data/eval_tokens.txt"})["validation"]

model = GPTNeoXForCausalLM.from_pretrained("../model/lol-model-neox").cuda().eval()
tokenizer = PreTrainedTokenizerFast.from_pretrained("../model/lol-model-neox")
assert tokenizer.vocab_size == model.config.vocab_size

def extract_single_token_probs(text, model, tokenizer, token_str):
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)["input_ids"].cuda()
    token_id = tokenizer.convert_tokens_to_ids(token_str)
    if token_id is None or token_id >= tokenizer.vocab_size:
        raise ValueError(f"{token_str} not found in vocab.")

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        probs = F.softmax(outputs.logits, dim=-1)[0]  # shape: [seq_len, vocab_size]
        token_probs = probs[:-1, token_id]  # Ignore final token (no next-token prediction)

    return {token_str: token_probs.cpu().tolist()}

def normalize_zscore(prob_list):
    arr = np.array(prob_list)
    return ((arr - arr.mean()) / (arr.std() + 1e-8)).tolist()

def normalize_minmax(prob_list):
    arr = np.array(prob_list)
    return ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8)).tolist()

def plot_event_prob_trends_normalized(event_probs, method="zscore"):
    for event, probs in event_probs.items():
        if method == "zscore":
            norm_probs = normalize_zscore(probs)
        elif method == "minmax":
            norm_probs = normalize_minmax(probs)
        else:
            raise ValueError("Unknown normalization method.")
        plt.plot(norm_probs, label=event)
    plt.title("Normalized Event Probabilities Over Match Timeline")
    plt.xlabel("Token Index")
    plt.ylabel("Normalized Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_event_prob_trends(event_probs, name):
    for event, probs in event_probs.items():
        plt.plot(probs, label=event)
    plt.title("Event Probabilities Over Match Timeline")
    plt.xlabel("Token Index")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.show()

# === Example match ===
sample_text = random.choice(eval_dataset)["text"]
input_ids = tokenizer(sample_text, return_tensors="pt")["input_ids"]

vocab_size = tokenizer.vocab_size
if torch.any(input_ids >= vocab_size):
    raise ValueError(f"Input contains token IDs >= vocab size ({vocab_size})")

# === Extract single-token probabilities ===
game_start_probs = extract_single_token_probs(sample_text, model, tokenizer, "[GAME_START]")
frame_probs = extract_single_token_probs(sample_text, model, tokenizer, "[FRAME]")
game_end_probs = extract_single_token_probs(sample_text, model, tokenizer, "[GAME_END]")
kill_probs = extract_single_token_probs(sample_text, model, tokenizer, "[KILL]")

# === Plot ===
plot_event_prob_trends(game_start_probs, "start")
plot_event_prob_trends(frame_probs, "frame")
plot_event_prob_trends(game_end_probs, "end")
plot_event_prob_trends(kill_probs,"kill")