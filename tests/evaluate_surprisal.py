import torch
import numpy as np
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from typing import List, Dict
from collections import defaultdict

class SlidingWindowSurprisalEvaluator:
    def __init__(self, model_path: str, tokenizer_path: str, block_size: int = 512, stride: int = 256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device).eval()
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

        self.block_size = block_size
        self.stride = stride
        self.player_tokens = [f"[P{i+1}]" for i in range(10)]

    def split_into_windows(self, text: str) -> List[List[int]]:
        input_ids = self.tokenizer(text)["input_ids"]
        input_ids = [x for x in input_ids if x != self.tokenizer.pad_token_id]

        windows = []
        for i in range(0, len(input_ids) - self.block_size + 1, self.stride):
            chunk = input_ids[i:i + self.block_size]
            windows.append(chunk)

        return windows, len(input_ids)

    def compute_surprisal(self, text: str) -> Dict:
        windows, total_len = self.split_into_windows(text)

        surprisals = np.zeros(total_len)
        counts = np.zeros(total_len)
        player_surprisals = defaultdict(list)

        with torch.no_grad():
            for w_idx, chunk in enumerate(windows):
                input_tensor = torch.tensor([chunk]).to(self.device)
                attention_mask = torch.ones_like(input_tensor).to(self.device)

                outputs = self.model(input_ids=input_tensor, attention_mask=attention_mask)
                logits = outputs.logits[0]  # shape: [seq_len, vocab_size]
                probs = torch.softmax(logits, dim=-1)

                for i in range(1, len(chunk)):
                    token_id = chunk[i]
                    prob = probs[i - 1][token_id].item()
                    surprisal = -np.log2(prob) if prob > 0 else float('inf')

                    global_pos = w_idx * self.stride + i
                    if global_pos < total_len:
                        surprisals[global_pos] += surprisal
                        counts[global_pos] += 1

                    token_str = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                    for player_id in self.player_tokens:
                        if player_id in token_str:
                            player_surprisals[player_id].append(surprisal)

        avg_surprisal = surprisals / np.maximum(counts, 1)
        player_scores = {
            pid: {
                "mean_surprisal": float(np.mean(s_list)) if s_list else None,
                "count": len(s_list),
                "total_surprisal": float(np.sum(s_list)) if s_list else 0.0
            }
            for pid, s_list in player_surprisals.items()
        }

        return {
            "surprisal_curve": avg_surprisal.tolist(),
            "mean_surprisal": float(np.mean(avg_surprisal)),
            "std_surprisal": float(np.std(avg_surprisal)),
            "sequence_length": total_len,
            "player_surprisal_scores": player_scores
        }

# Example usage
if __name__ == "__main__":
    evaluator = SlidingWindowSurprisalEvaluator("../model/lol-gpt-medium", "../model/lol-gpt-medium")

    with open("../data/processed_tokens.txt") as f:
        sample_text = f.readlines()[0]  # pick first match

    result = evaluator.compute_surprisal(sample_text)

    print(f"Sequence length: {result['sequence_length']}")
    print(f"Mean surprisal: {result['mean_surprisal']:.4f}")
    print(f"Std surprisal: {result['std_surprisal']:.4f}")
    print("\nPer-player surprisal summary:")
    for pid, stats in result["player_surprisal_scores"].items():
        print(f"{pid}: count={stats['count']}, mean={stats['mean_surprisal']:.4f}, total={stats['total_surprisal']:.2f}")
