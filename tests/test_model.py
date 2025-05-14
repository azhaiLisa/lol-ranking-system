import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
import json
from typing import List, Dict
import argparse
from tqdm import tqdm

class LeagueModelTester:
    def __init__(self, model_path: str, tokenizer_path: str):
        """Initialize the model tester with a trained model and tokenizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def generate_sequence(self, 
                         prompt: str, 
                         min_length: int,
                         max_length: int, 
                         temperature: float = 1.0,
                         top_p: float = 0.9,
                         num_return_sequences: int = 1) -> List[str]:
        """Generate sequences from a prompt."""
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate sequences
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                min_length=min_length,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
        
        # Decode sequences
        generated_sequences = [
            self.tokenizer.decode(output, skip_special_tokens=False)
            for output in outputs
        ]
        
        return generated_sequences
    
    def test_sequence_completion(self, test_cases: List[Dict]):
        """Test the model's ability to complete sequences."""
        print("\nTesting Sequence Completion:")
        for i, test_case in enumerate(test_cases, 1):
            prompt = test_case["prompt"]
            print(f"\nTest Case {i}:")
            print(f"Prompt: {prompt}")
            
            # Generate completions
            completions = self.generate_sequence(
                prompt,
                min_length = test_case.get("min_legth", 510),
                max_length=test_case.get("max_length", 1010),
                temperature=test_case.get("temperature", 1.0),
                num_return_sequences=test_case.get("num_return_sequences", 1)
            )
            
            # Print results
            for j, completion in enumerate(completions, 1):
                print(f"Completion {j}: {completion}")
    
    def test_model_loading(self):
        """Test if the model and tokenizer are loaded correctly."""
        print("\nTesting Model Loading:")
        print(f"Model type: {type(self.model).__name__}")
        print(f"Tokenizer type: {type(self.tokenizer).__name__}")
        print(f"Vocabulary size: {len(self.tokenizer)}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def test_tokenization(self, test_sequences: List[str]):
        """Test the tokenizer on various sequences."""
        print("\nTesting Tokenization:")
        for i, sequence in enumerate(test_sequences, 1):
            print(f"\nSequence {i}: {sequence}")
            
            # Tokenize
            tokens = self.tokenizer.tokenize(sequence)
            token_ids = self.tokenizer.encode(sequence)
            
            print(f"Tokens: {tokens}")
            print(f"Token IDs: {token_ids}")
            
            # Decode back
            decoded = self.tokenizer.decode(token_ids)
            print(f"Decoded: {decoded}")

def main():
    parser = argparse.ArgumentParser(description="Test the trained League of Legends model")
    parser.add_argument("--model_path", default="../model/lol-gpt-medium", help="Path to the trained model")
    parser.add_argument("--tokenizer_path", default="../model/lol-gpt-medium", help="Path to the tokenizer")
    args = parser.parse_args()
    
    # Initialize tester
    tester = LeagueModelTester(args.model_path, args.tokenizer_path)
    
    # Test model loading
    tester.test_model_loading()
    
    # Test tokenization
    test_sequences = [
        "[GAME_START][ITEM_BUY][P0][WORLD_ATLAS]",
        "[SKILL_UP][P1][SKILL_Q]",
        "[KILL_NORMAL][P1][P2][300]"
    ]
    tester.test_tokenization(test_sequences)
    
    # Test sequence completion
    test_cases = [
        {
            "prompt": "[RANK_GOLD][GAME_START][ITEM_BUY][P0][WORLD_ATLAS]",
            "max_length": 512,
            "min_length": 200,
            "temperature": 0.9,
            "top_p": 0.95,
            "num_return_sequences": 2
        },
        {
            "prompt": "[RANK_MASTER][GAME_START][ITEM_BUY][P0][WORLD_ATLAS]",
            "max_length": 50,
            "temperature": 0.7,
            "num_return_sequences": 2
        }
    ]
    tester.test_sequence_completion(test_cases)

if __name__ == "__main__":
    main() 