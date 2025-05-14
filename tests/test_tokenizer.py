from transformers import PreTrainedTokenizerFast

# Loading custom tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("../model/my_tokenizer")

print("[KILL]" in tokenizer.get_vocab()) 
print("[P1]" in tokenizer.get_vocab())   
print(tokenizer.tokenize("[KILL][P1][ASSIST][P2][P4][P6][160]"))

def test_tokenization_preserves_tokens():
    print("Running: test_tokenization_preserves_tokens")
    input_str = "[KILL][P1][ASSIST][P2][P4][P6][160]"
    tokens = tokenizer.tokenize(input_str)
    expected = ['[KILL]', '[P1]', '[ASSIST]', '[P2]', '[P4]', '[P6]', '[160]']
    assert tokens == expected, f"Token mismatch: got {tokens}, expected {expected}"

def test_encoding_and_decoding():
    print("Running: test_encoding_and_decoding")
    input_str = "[FRAME] [SKILL_UP] [P3] [SKILL_Q]"
    enc = tokenizer.encode(input_str, add_special_tokens=False)
    dec = tokenizer.decode(enc)
    for token in ["[FRAME]", "[SKILL_UP]", "[P3]", "[SKILL_Q]"]:
        assert token in dec, f"{token} missing in decoded output: {dec}"

def test_unknown_token():
    print("Running: test_unknown_token")
    input_str = "[NON_EXISTENT_TOKEN]"
    tokens = tokenizer.tokenize(input_str)
    assert "[UNK]" in tokens, f"Expected unknown token fallback, got: {tokens}"

def test_padding_behavior():
    print("Running: test_padding_behavior")
    encoded = tokenizer.encode("[GAME_START]", padding="max_length", max_length=5)
    assert len(encoded) == 5, f"Expected padded length 5, got: {len(encoded)}"

if __name__ == "__main__":
    test_tokenization_preserves_tokens()
    test_encoding_and_decoding()
    test_unknown_token()
    test_padding_behavior()
    print("✅ All tokenizer tests passed.")