import torch
import torch.nn.functional as F
from datasets import load_dataset
import random
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import matplotlib.pyplot as plt
from itertools import chain
import numpy as np


eval_dataset = load_dataset("text", data_files={"validation": "../data/eval_tokens.txt"})["validation"]

model = GPT2LMHeadModel.from_pretrained("../model/lol-model-scratch").cuda().eval()
tokenizer = PreTrainedTokenizerFast.from_pretrained("../model/lol-model-scratch")
assert tokenizer.vocab_size == model.config.vocab_size

def split_into_windows(text, tokenizer, block_size=512, stride=256):
    input_ids = tokenizer(text)["input_ids"]
    input_ids = [x for x in input_ids if x != tokenizer.pad_token_id]

    windows = []
    masks = []

    for i in range(0, len(input_ids) - block_size + 1, stride):
        chunk = input_ids[i:i + block_size]
        windows.append(chunk)
        masks.append([1] * len(chunk))

    return windows, masks



def extract_pattern_probs_aligned(text, model, tokenizer, pattern_type, pattern_length=3, block_size=512, stride=256):
    """
    Timeline-aware version of extract_pattern_probs.
    Accumulates probabilities per real token index and averages across overlaps.
    """
    input_ids_full = tokenizer(text)["input_ids"]
    timeline_len = len(input_ids_full)

    pattern_id = tokenizer.convert_tokens_to_ids(pattern_type)
    if pattern_id is None or pattern_id >= tokenizer.vocab_size:
        raise ValueError(f"{pattern_type} not found in tokenizer vocab.")

    event_scores = np.zeros(timeline_len)
    event_counts = np.zeros(timeline_len)

    windows, masks = split_into_windows(text, tokenizer, block_size, stride)

    model.eval()
    with torch.no_grad():
        for w_idx, (input_ids, attention_mask) in enumerate(zip(windows, masks)):
            input_tensor = torch.tensor([input_ids]).to(model.device)
            mask_tensor = torch.tensor([attention_mask]).to(model.device)

            logits = model(input_ids=input_tensor, attention_mask=mask_tensor).logits
            probs = F.softmax(logits, dim=-1)[0]  # [seq_len, vocab_size]

            for t in range(len(probs) - pattern_length + 1):
                if input_ids[t] != pattern_id:
                    continue

                p_total = 0.0
                for p1 in [f"[P{i}]" for i in range(10)]:
                    pid1 = tokenizer.convert_tokens_to_ids(p1)
                    if pid1 is None or pid1 >= tokenizer.vocab_size:
                        continue

                    if pattern_length == 2:
                        p = probs[t][pattern_id] * probs[t + 1][pid1]
                        p_total += p.item()

                    elif pattern_length == 3:
                        for tok_id2 in range(tokenizer.vocab_size):
                            p = probs[t][pattern_id] * probs[t + 1][pid1] * probs[t + 2][tok_id2]
                            p_total += p.item()

                # Align to real token index
                global_idx = w_idx * stride + t
                if global_idx < timeline_len:
                    event_scores[global_idx] += p_total
                    event_counts[global_idx] += 1

    avg_probs = (event_scores / np.maximum(event_counts, 1)).tolist()
    return {f"{pattern_type}[P*]{'[*]' if pattern_length == 3 else ''}": avg_probs}


def extract_single_token_probs_aligned(text, model, tokenizer, token_str, block_size=512, stride=256):
    input_ids_full = tokenizer(text)["input_ids"]
    timeline_len = len(input_ids_full)

    token_id = tokenizer.convert_tokens_to_ids(token_str)
    if token_id is None or token_id >= tokenizer.vocab_size:
        raise ValueError(f"{token_str} not found in vocab.")

    # Initialize timeline
    event_scores = np.zeros(timeline_len)
    event_counts = np.zeros(timeline_len)

    # Sliding windows
    windows, masks = split_into_windows(text, tokenizer, block_size, stride)

    model.eval()
    with torch.no_grad():
        for w_idx, (window_ids, attention_mask) in enumerate(zip(windows, masks)):
            input_tensor = torch.tensor([window_ids]).to(model.device)
            mask_tensor = torch.tensor([attention_mask]).to(model.device)

            outputs = model(input_ids=input_tensor, attention_mask=mask_tensor)
            probs = F.softmax(outputs.logits, dim=-1)[0]  # shape: [seq_len, vocab_size]
            # token_probs = probs[:-1, token_id]  # skip final prediction position
            token_probs = probs[:-1, token_id]

            # Map window to timeline
            start = w_idx * stride
            for i, p in enumerate(token_probs.cpu().tolist()):
                global_idx = start + i
                if global_idx < timeline_len:
                    event_scores[global_idx] += p
                    event_counts[global_idx] += 1

    avg_probs = (event_scores / np.maximum(event_counts, 1)).tolist()
    return {token_str: avg_probs}

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



# === Function: plot probability trends ===
def plot_event_prob_trends(event_probs):
    for event, probs in event_probs.items():
        plt.plot(probs, label=event)
    plt.title("Event Probabilities Over Match Timeline")
    plt.xlabel("Token Index")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Pick one example from the eval set ===
# sample_text = dataset["train"][0]["text"]
sample_text = random.choice(eval_dataset)["text"]
# sample_text = random.choice(dataset["train"])["text"]
input_ids = tokenizer(sample_text, return_tensors="pt")["input_ids"]
# events_to_track = ["[ITEM_BUY]", "[KILL]", "[GAME_END]"]
# event_probs = extract_event_probabilities_sliding(sample_text, model, tokenizer, events_to_track)

# print(input_ids[0])

vocab_size = tokenizer.vocab_size
if torch.any(input_ids >= vocab_size):
    raise ValueError(f"Input contains token IDs >= vocab size ({vocab_size})")

# item_probs = extract_pattern_probs_aligned(sample_text, model, tokenizer, "[ITEM_BUY]", 3)
# kill_probs = extract_pattern_probs_aligned(sample_text, model, tokenizer, "[KILL]", 3)
game_start_probs = extract_single_token_probs_aligned(sample_text, model, tokenizer, "[GAME_START]")
frame_probs = extract_single_token_probs_aligned(sample_text, model, tokenizer, "[FRAME]")
game_end_probs = extract_single_token_probs_aligned(sample_text, model, tokenizer, "[GAME_END]")

# === Plot the trends ===
# plot_event_prob_trends(item_probs)
# plot_event_prob_trends(kill_probs)
plot_event_prob_trends(game_start_probs)
plot_event_prob_trends(frame_probs)
plot_event_prob_trends(game_end_probs)
# plot_normalised(game_end_probs)
# plot_event_prob_trends_normalized(
#     {**item_probs, **kill_probs, **game_end_probs},
#     method="zscore"  # or "zscore"
# )



# true_pos = tokenizer.tokenize(sample_text).index("[GAME_END]")
# print(f"True GAME_END at token index: {true_pos}")
