import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

KDA_FILE = "match_kda.csv"
SURPISE_FILE = "match_surprise.csv"

# Load CSVs
kda_df = pd.read_csv(KDA_FILE)
surprise_df = pd.read_csv(SURPISE_FILE)

# Merge by Match ID
merged = pd.merge(kda_df, surprise_df, on="Match ID", suffixes=("_KDA", "_SURPRISE"))
merged.columns = [col.replace('[', '').replace(']', '') for col in merged.columns]

print("Merged DataFrame shape:", merged.shape)
print(merged.head(2).to_string(index=False))

# Config 
positions = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]
teams = ["B", "R"]

# Mixed-team Spearman correlations
mixed_spearman_scores = []
mixed_match_ids = []

for _, row in merged.iterrows():
    match_id = row["Match ID"]
    kda_vals = []
    surprise_vals = []
    
    for team in teams:
        for pos in positions:
            kda_val = row.get(f"{pos}_{team}_KDA", None)
            surprise_val = row.get(f"{pos}_{team}_SURPRISE", None)
            if pd.notna(kda_val) and pd.notna(surprise_val):
                kda_vals.append(kda_val)
                surprise_vals.append(surprise_val)
    
    if len(set(kda_vals)) > 1 and len(set(surprise_vals)) > 1:
        corr, _ = spearmanr(kda_vals, surprise_vals)
        mixed_spearman_scores.append(corr)
        mixed_match_ids.append(match_id)
    else:
        print(f"Skipped match {match_id} due to uniform scores")

# Plot histogram 
plt.figure(figsize=(8, 5))
plt.hist(mixed_spearman_scores, bins=20, edgecolor='black')
plt.title("Spearman Correlation: KDA vs Surprise Score (Mixed Teams)")
plt.xlabel("Spearman Correlation")
plt.ylabel("Number of Matches")
plt.grid(True)
plt.tight_layout()
plt.savefig("kda_surprise_correlation_mixed.png")
plt.show()

# Save output
pd.DataFrame({
    "Match ID": mixed_match_ids,
    "Spearman": mixed_spearman_scores
}).to_csv("spearman_correlation_mixed.csv", index=False)
