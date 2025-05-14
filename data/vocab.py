import re

input_file = "processed_tokens.txt" 
output_vocab = "vocab.txt"

# Extract tokens like [SKILL_Q], [P2], etc.
token_pattern = re.compile(r"\[[^\[\]]+?\]")
token_set = set()

with open(input_file, "r") as f:
    for line in f:
        tokens = token_pattern.findall(line)
        token_set.update(tokens)

# Add special tokens
special_tokens = ["[PAD]", "[UNK]"]
all_tokens = special_tokens + sorted(token_set)

with open(output_vocab, "w") as f:
    for token in all_tokens:
        f.write(token + "\n")

print(f"✅ Extracted {len(all_tokens)} tokens to {output_vocab}")
