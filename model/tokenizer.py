from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Sequence, Split, Whitespace
from tokenizers.processors import BertProcessing
from transformers import PreTrainedTokenizerFast
import json

# Your vocabulary list (must include all tokens exactly as they appear)
# vocab = [
#     "[PAD]",
#     "[UNK]",
#     # ... rest of your vocabulary items ...
#     "[KILL]",
#     "[P1]",
#     "[ASSIST]",
#     # ... etc ...
# ]

with open("../data/vocab.txt", "r") as f:
    vocab = [line.strip() for line in f if line.strip()]  # Removes empty lines and whitespace

# Create vocab dictionary
vocab_dict = {token: i for i, token in enumerate(vocab)}

# Build tokenizer
tokenizer = Tokenizer(WordLevel(vocab=vocab_dict, unk_token="[UNK]"))

# Custom pre-tokenizer to handle bracketed tokens without spaces
# First split on '][' to separate adjacent tokens
split_on_close_open = Split(r"\]\[", behavior="isolated")
# Then split any remaining patterns (standalone bracketed tokens)
split_bracketed = Split(r"(\[[^\[\]]+?\])", behavior="isolated")
tokenizer.pre_tokenizer = Sequence([Whitespace(),split_on_close_open, split_bracketed])

# Post-processor
tokenizer.post_processor = BertProcessing(
    ("[PAD]", vocab_dict["[PAD]"]), 
    ("[UNK]", vocab_dict["[UNK]"])
)

# Wrap in HF tokenizer
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    # Tell HF these are special tokens that should never be split
    additional_special_tokens=[t for t in vocab if t.startswith("[") and t.endswith("]")]
)

# Test with actual format
test_str = " [WARD_PLACE][PNone][YELLOW] [SKILL_UP][BOTTOM_B][SKILL_Q]"
print(hf_tokenizer.tokenize(test_str)) 

# Save
hf_tokenizer.save_pretrained("./my_tokenizer")

# Save config with pre-tokenizer info
with open("./my_tokenizer/tokenizer_config.json", "w") as f:
    json.dump({
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": r"\]\[", "behavior": "isolated"},
                {"type": "Split", "pattern": r"(\[[^\[\]]+?\])", "behavior": "isolated"}
            ]
        }
    }, f, indent=2)