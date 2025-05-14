import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
import numpy as np
from typing import List, Dict, Tuple
import json
from tqdm import tqdm

class SequenceSurprisal:
    def __init__(self, model_path: str, tokenizer_path: str):
        """Initialize the surprisal evaluator with a trained model and tokenizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def tokenize_sequence(self, sequence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a sequence string into input_ids and attention_mask."""
        # Split the sequence into tokens if it's a string
        if isinstance(sequence, str):
            tokens = sequence.split()
        else:
            tokens = sequence
            
        # Tokenize
        encoded = self.tokenizer(
            tokens,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)
    
    def get_token_surprisal(self, sequence: str) -> List[Dict]:
        """Calculate surprisal for each token in the sequence."""
        input_ids, attention_mask = self.tokenize_sequence(sequence)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
        # Calculate surprisal for each position
        surprisals = []
        for i in range(1, len(input_ids[0])):  # Skip first token as it has no prediction
            # Get logits for the next token
            next_token_logits = logits[0, i-1]
            
            # Convert to probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Get the actual next token
            actual_token_id = input_ids[0, i]
            
            # Calculate surprisal (-log2 of probability)
            token_prob = probs[actual_token_id].item()
            surprisal = -np.log2(token_prob) if token_prob > 0 else float('inf')
            
            # Get the token string
            token_str = self.tokenizer.decode([actual_token_id])
            
            surprisals.append({
                "position": i,
                "token": token_str,
                "surprisal": surprisal,
                "probability": token_prob
            })
        
        return surprisals
    
    def evaluate_sequence(self, sequence: str) -> Dict:
        """Evaluate a complete sequence and return summary statistics."""
        surprisals = self.get_token_surprisal(sequence)
        
        # Calculate statistics
        surprisal_values = [s["surprisal"] for s in surprisals]
        
        return {
            "sequence_length": len(surprisals),
            "mean_surprisal": np.mean(surprisal_values),
            "median_surprisal": np.median(surprisal_values),
            "max_surprisal": np.max(surprisal_values),
            "min_surprisal": np.min(surprisal_values),
            "std_surprisal": np.std(surprisal_values),
            "token_by_token": surprisals
        }

def main():
    # Initialize evaluator
    evaluator = SequenceSurprisal(
        model_path="./league_model_final",
        tokenizer_path="./league_model_final"
    )
    
    # Example sequence
    test_sequence = "[GAME_START] [ITEM_BUY][P1][POTION] [SKILL_UP][P1][SKILL_Q] [GAME_END]"
    
    # Evaluate sequence
    results = evaluator.evaluate_sequence(test_sequence)
    
    # Print summary
    print("\nSequence Evaluation Summary:")
    print(f"Sequence Length: {results['sequence_length']}")
    print(f"Mean Surprisal: {results['mean_surprisal']:.4f}")
    print(f"Median Surprisal: {results['median_surprisal']:.4f}")
    print(f"Max Surprisal: {results['max_surprisal']:.4f}")
    print(f"Min Surprisal: {results['min_surprisal']:.4f}")
    print(f"Std Surprisal: {results['std_surprisal']:.4f}")
    
    # Print token-by-token analysis
    print("\nToken-by-Token Analysis:")
    for token_info in results["token_by_token"]:
        print(f"Position {token_info['position']}: {token_info['token']}")
        print(f"  Surprisal: {token_info['surprisal']:.4f}")
        print(f"  Probability: {token_info['probability']:.4f}")

if __name__ == "__main__":
    main() 