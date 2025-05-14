from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

# Load vocab
with open("../data/vocab.txt", "r") as f:
    vocab = [line.strip() for line in f.readlines()]
vocab_dict = {token: i for i, token in enumerate(vocab)}

# Build WordLevel tokenizer
tokenizer = Tokenizer(WordLevel(vocab=vocab_dict, unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Wrap in HF-compatible tokenizer
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
hf_tokenizer.add_special_tokens({
    "pad_token": "[PAD]",
    "unk_token": "[UNK]"
})

# Save to disk
hf_tokenizer.save_pretrained("./my_tokenizer")
