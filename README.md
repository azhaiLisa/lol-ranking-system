# lol-ranking-system

This project aims to create a better ranking system for League of Legends by leveraging detailed match data, advanced tokenization, and machine learning.

## Data Pipeline Overview

### 1. Data Collection

- **Script:** `data/data.py`
- **Source:** Riot Games API (requires your own API key)
- **Process:**
  - Collects match data from multiple regions, focusing on ranked solo queue games.
  - Fetches player, match, and timeline data.
  - Filters out unranked and non-solo matches.
  - Stores all collected matches in `match_ranked_multiregion.json` in batch processing.
  - Also saves progress in `matches.pkl` and `seen_ids.pkl` for crash recovery.

### 2. Data Structure

Each match entry in the JSON database (e.g., `match_ranked_multiregion.json`, `example_ranked_match.json`) has the following structure:

```json
{
  "match_id": "KR_123456789",
  "rank": "Master I",
  "timeline": { ... },   // Full timeline of in-game events
  "metadata": { ... },   // Match metadata, including participants roles
  "puuid": "...",        // Player unique identifier
  "region": "kr"
}
```

- The `timeline` field contains a sequence of frames, each with events (kills, item purchases, skill-ups, etc.) and participant stats.

### 3. Tokenization

- **Script:** `data/token_processor.py`
- **Process:**
  - Reads match data (default: `matches.json` or any batch file).
  - Converts each match's timeline into a sequence of tokens representing in-game events, actions, and states.
  - Maps participant IDs to roles (e.g., `[P1]` → `[TOP_B]`).
  - Outputs tokenized data to `processed_tokens.txt` (for model training/validation).

### 4. Vocabulary

- **File:** `data/vocab.txt`
- Contains the full set of token vobabulary used for model training, extracted with `data/vocab.py` from `processed_tokens.txt`.

### 5. Model Training

- **Directory:** `model/`
- **Scripts and Purpose:**

  - `tokenizer.py`: Builds a custom tokenizer using the HuggingFace `tokenizers` library, tailored for the bracketed token format. Loads vocabulary from `data/vocab.txt`, applies custom pre-tokenization, and saves the tokenizer to `model/my_tokenizer/`.
  - `train_neo.py`: Trains a GPTNeoX-based language model from scratch using the custom tokenizer and tokenized match data. Handles data loading, splitting, tokenization, model configuration, training, checkpointing, and loss plotting. Saves the trained model to `model/lol-model-neox/`.

- **Directories and Files:**

  - `my_tokenizer/`: Directory containing the saved tokenizer artifacts for use in training and inference.
  - `lol-model-neox/`: Directory containing the trained model checkpoints and tokenizer after running `train_neo.py`.
  - `train_logs_scratch.json`: Training logs (loss, steps, etc.).
  - `loss_plot_10epochs.png`: Visualization of training and validation loss.

- **Usage:**
  - Run `tokenizer.py` after updating your vocabulary to generate a tokenizer compatible with your data and models.
  - Run `train_neo.py` to train a new model from scratch on your processed data. Adjust hyperparameters and paths as needed.

### 6.  Evaluation

- **Directory:** `eval/`
- **Scripts and Purpose:**

  - `eval_surprise.py`: Computes aggregate surprise scores for all roles across a batch of matches, saving results to `match_surprise.csv` for further analysis.
  - `eval_neo.py`: Visualizes the probability assigned by the model to specific tokens (e.g., `[GAME_START]`, `[FRAME]`, `[KILL]`) over the course of a match, saving plots as PNG images.
  - `eval_single.py`: Evaluates a single match, computing "surprise" scores for key in-game events and roles using the trained model. Prints the most surprising positive/negative events for each role.

  - `compute_kda.py`: Computes KDA (Kill/Death/Assist) ratios for each role from generated match data, saving results as `generated_kda.csv`.
  - `compare_kda.py`: Computes the Spearman correlation between KDA and surprise scores for each team in each match, saving results as `spearman_correlation_labeled.csv` and plotting a histogram.
  - `compare_combined.py`: Computes the Spearman correlation between KDA and surprise scores across all roles in both teams (mixed), saving results as `spearman_correlation_mixed.csv` and plotting a histogram.

  - `generate.py`: Uses the trained model to generate new match sequences from a given prompt, saving the generated matches to `generated_matches.txt`.

- **Usage:**
  - Run these scripts to analyze model performance, event surprise, and the relationship between model predictions and in-game statistics.


## How to Use

1. **Collect Data:**  
   Edit `API_KEY` in `data/data.py` and run the script to collect new match data.

2. **Tokenize Data:**  
   Run `data/token_processor.py` to convert raw match data into tokens.

3. **Train Model:**  
   Use scripts in `model/` to train your ranking model on the tokenized data.

4. **Evaluate:**  
   Use scripts in `eval/` to evaluate model performance.

## Notes

- The data files are large; ensure you have sufficient disk space.
- The project is modular: you can swap in new data, change tokenization, or adjust model training as needed.
