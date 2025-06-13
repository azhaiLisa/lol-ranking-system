import re
import csv
from collections import defaultdict

INPUT_FILE = "../eval/generated_matches.txt"
OUTPUT_FILE = "generated_kda.csv"

ROLES = [
    "[BOTTOM_B]", "[BOTTOM_R]", "[JUNGLE_B]", "[JUNGLE_R]",
    "[MIDDLE_B]", "[MIDDLE_R]", "[TOP_B]", "[TOP_R]",
    "[UTILITY_B]", "[UTILITY_R]"
]

PLAYER_RE = re.compile(r"\[([A-Z0-9_]+_[RB])\]")
KILL_PAT = re.compile(r"\[KILL]\[([A-Z0-9_]+_[RB])]\[([A-Z0-9_]+_[RB])]")  # compressed format
ASSIST_PAT = re.compile(r"\[ASSIST]")

def kda_from_tokens(line: str):
    K, D, A = defaultdict(int), defaultdict(int), defaultdict(int)
    tokens = re.findall(r"\[[^\[\]]+\]|\S+", line)  # robust split

    i = 0
    while i < len(tokens):
        t = tokens[i]

        m = KILL_PAT.fullmatch(t)
        if m:
            killer, victim = m.group(1), m.group(2)
            K[killer] += 1
            D[victim] += 1
            i += 1
            if i < len(tokens) and ASSIST_PAT.fullmatch(tokens[i]):
                i += 1
                while i < len(tokens) and PLAYER_RE.fullmatch(tokens[i]):
                    A[PLAYER_RE.fullmatch(tokens[i]).group(1)] += 1
                    i += 1
            continue

        if t == "[KILL]" and i + 2 < len(tokens):
            killer = tokens[i + 1].strip("[]")
            victim = tokens[i + 2].strip("[]")
            K[killer] += 1
            D[victim] += 1
            i += 3
            if i < len(tokens) and tokens[i] == "[ASSIST]":
                i += 1
                while i < len(tokens) and PLAYER_RE.fullmatch(tokens[i]):
                    A[tokens[i].strip("[]")] += 1
                    i += 1
            continue

        i += 1

    out = {}
    for r in ROLES:
        tag = r.strip("[]")
        k, d, a = K[tag], D[tag], A[tag]
        ratio = round((k + a) / max(1, d), 2)
        out[r] = ratio
    return out

# Run on all matches and save to CSV
match_rows = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        kda_scores = kda_from_tokens(line.strip())
        row = {"Match ID": idx}
        for role in ROLES:
            row[role] = kda_scores[role]
        match_rows.append(row)

# Save to CSV
fieldnames = ["Match ID"] + ROLES
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(match_rows)

print(f"KDA summary saved to {OUTPUT_FILE} in compact format")
