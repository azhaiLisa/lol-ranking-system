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

# Collect Spearman correlations
spearman_scores = []

match_ids = []
teams_list = []
spearman_scores = []

for _, row in merged.iterrows():
    match_id = row["Match ID"]
    for team in teams:
        kda_vals = []
        surprise_vals = []
        for pos in positions:
            kda_val = row.get(f"{pos}_{team}_KDA", None)
            surprise_val = row.get(f"{pos}_{team}_SURPRISE", None)
            if pd.notna(kda_val) and pd.notna(surprise_val):
                kda_vals.append(kda_val)
                surprise_vals.append(surprise_val)
        
        if len(set(kda_vals)) > 1 and len(set(surprise_vals)) > 1:
            corr, _ = spearmanr(kda_vals, surprise_vals)
            spearman_scores.append(corr)
            match_ids.append(match_id)
            teams_list.append(team)
        else:
            print(f"skipped match {match_id} team {team} due to uniform scores")      

# Plot the histogram
plt.figure(figsize=(8, 5))
plt.hist(spearman_scores, bins=15, edgecolor='black')
plt.title("Spearman Correlation: KDA vs Surprise Score (per Team)")
plt.xlabel("Spearman Correlation")
plt.ylabel("Number of Teams")
plt.grid(True)
plt.tight_layout()
plt.savefig("kda_surprise_correlation.png")
plt.show()

# Save correlations 
pd.DataFrame({
    "Match ID": match_ids,
    "Team": teams_list,
    "Spearman": spearman_scores
}).to_csv("spearman_correlation_labeled.csv", index=False)