import torch
import torch.nn.functional as F
import math
from collections import defaultdict
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerFast
from tqdm import tqdm
import re

# Setup
MODEL_PATH = "../model/lol-model-neox"
TOKENS_FILE = "../data/the_match.txt"
MATCH_INDEX = 0  # Change if needed

# Load tokenizer
hf_tokenizer = PreTrainedTokenizerFast.from_pretrained("../model/my_tokenizer")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTNeoXForCausalLM.from_pretrained(MODEL_PATH).to(device).eval()
assert hf_tokenizer.vocab_size == model.config.vocab_size

# Tokens & Config
ROLE_TOKENS = {
    "[TOP_B]", "[JUNGLE_B]", "[MIDDLE_B]", "[BOTTOM_B]", "[UTILITY_B]",
    "[TOP_R]", "[JUNGLE_R]", "[MIDDLE_R]", "[BOTTOM_R]", "[UTILITY_R]"
}

BLUE_ROLES = {
    "[TOP_B]", "[JUNGLE_B]", "[MIDDLE_B]", "[BOTTOM_B]", "[UTILITY_B]",
}
ROLE_TOKEN_IDS = [hf_tokenizer.convert_tokens_to_ids(r) for r in ROLE_TOKENS]

SURPRISE_EVENT_HEADS = {
    "[KILL]", "[ASSIST]",
    "[SPECIAL_DOUBLE_KILL]", "[SPECIAL_TRIPLE_KILL]", "[SPECIAL_QUADRA_KILL]",
    "[SPECIAL_PENTA_KILL]", "[SPECIAL_KILL_ACE]", "[SPECIAL_KILL_FIRST_BLOOD]",
    "[SKILL_UP]", "[LEVEL_UP]",
    "[ITEM_BUY]", "[ITEM_SELL]", "[ITEM_USE]", "[ITEM_UNDO]",
    "[WARD_PLACE]", "[WARD_KILL]",
    "[BUILDING_DESTROY]", "[BUILDING_PLATE]",
    "[MONSTER_DRAGON]", "[MONSTER_BARON_NASHOR]", "[MONSTER_RIFTHERALD]",
    "[MONSTER_HORDE]", "[MONSTER_ATAKHAN]"
}

EVENT_WEIGHTS = {
    "[KILL]": 1.5, "[ASSIST]": 0.8,
    "[SPECIAL_DOUBLE_KILL]": 1.6, "[SPECIAL_TRIPLE_KILL]": 2.2,
    "[SPECIAL_QUADRA_KILL]": 3.0, "[SPECIAL_PENTA_KILL]": 4.0,
    "[SPECIAL_KILL_ACE]": 2.5, "[SPECIAL_KILL_FIRST_BLOOD]": 2.0,
    "[SKILL_UP]": 0.05, "[LEVEL_UP]": 0.05,
    "[ITEM_BUY]": 0.15, "[ITEM_SELL]": 0.05,
    "[ITEM_USE]": 0.1, "[ITEM_UNDO]": 0.05,
    "[WARD_PLACE]": 0.1, "[WARD_KILL]": 0.2,
    "[BUILDING_DESTROY]": 3.0, "[BUILDING_PLATE]": 1.0,
    "[MONSTER_DRAGON]": 3.0, "[MONSTER_BARON_NASHOR]": 4.5,
    "[MONSTER_RIFTHERALD]": 2.5, "[MONSTER_HORDE]": 1.5,
    "[MONSTER_ATAKHAN]": 3.5
}

EVENT_ROLE_RATIO = 0.5

# Surprise Calculation
def compute_event_surprise(logits, token_id):
    prob = F.softmax(logits, dim=-1)[token_id].item()
    return -math.log2(prob + 1e-8)

def compute_role_surprise(logits, token_id):
    if token_id not in ROLE_TOKEN_IDS:
        return 0.0
    role_logits = logits[ROLE_TOKEN_IDS]
    role_probs = F.softmax(role_logits, dim=0)
    index = ROLE_TOKEN_IDS.index(token_id)
    prob = role_probs[index].item()
    return -math.log2(prob + 1e-8)

def parse_event_roles(tokens, start_idx):
    roles = []
    i = start_idx + 1
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("[") and (t in SURPRISE_EVENT_HEADS or t.startswith("[FRAME") or t.startswith("[GAME_")):
            break
        if t in ROLE_TOKENS:
            roles.append((i, t))
        i += 1
    return roles

def evaluate_and_log_events(tokens):
    input_ids = hf_tokenizer.convert_tokens_to_ids(tokens)
    event_logs = defaultdict(list)

    for i in range(1, len(tokens)):
        token = tokens[i]
        if token not in SURPRISE_EVENT_HEADS:
            continue

        input_tensor = torch.tensor([input_ids[:i]], device=device)
        with torch.no_grad():
            logits = model(input_tensor).logits[0, -1]
        event_token_id = input_ids[i]
        event_surprise = compute_event_surprise(logits, event_token_id)

        role_positions = parse_event_roles(tokens, i)
        role_surprises = {}
        for idx, _ in role_positions:
            role_tensor = torch.tensor([input_ids[:idx]], device=device)
            with torch.no_grad():
                role_logits = model(role_tensor).logits[0, -1]
            role_surprises[idx] = compute_role_surprise(role_logits, input_ids[idx])

        weight = EVENT_WEIGHTS.get(token, 1.0)

        def total_surprise(idx):
            return weight * (
                EVENT_ROLE_RATIO * event_surprise + (1 - EVENT_ROLE_RATIO) * role_surprises.get(idx, 0.0)
            )

        if token == "[KILL]" and len(role_positions) >= 2:
            killer_idx, killer = role_positions[0]
            victim_idx, victim = role_positions[1]
            event_logs[killer].append((i, token, round(total_surprise(killer_idx), 2)))
            event_logs[victim].append((i, token, round(-total_surprise(victim_idx), 2)))  # negative
        elif token == "[ASSIST]":
            for idx, role in role_positions:
                score = 0.5 * total_surprise(idx)
                event_logs[role].append((i, token, round(score, 2)))
        else:
            for idx, role in role_positions:
                event_logs[role].append((i, token, round(total_surprise(idx), 2)))

    return event_logs

# Main
if __name__ == "__main__":
    with open(TOKENS_FILE, "r", encoding="utf-8") as f:
        eval_dataset = [line.strip() for line in f]

    tokens = hf_tokenizer.tokenize(eval_dataset[MATCH_INDEX])
    event_logs = evaluate_and_log_events(tokens)

    print(f"\n🔍 Top Surprise Events Per Team Player (Match ID {MATCH_INDEX}):\n")
    for role in sorted(ROLE_TOKENS):
        events = event_logs.get(role, [])
        if not events:
            print(f"{role}: No surprise events found")
            continue

        top_positive = sorted([e for e in events if e[2] > 0], key=lambda x: -x[2])[:10]
        top_negative = sorted([e for e in events if e[2] < 0], key=lambda x: x[2])[:10]

        print(f"{role}:")
        for idx, evt, score in top_positive:
            print(f"   ▲ {evt} at token {idx} → Surprise Score = {score}")
        for idx, evt, score in top_negative:
            print(f"   ▼ {evt} at token {idx} → Surprise Score = {score}")
