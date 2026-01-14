# Team Transcript Performance Prediction

This repository contains a reproducible NLP pipeline for predicting **team performance scores** from **long transcripts** of team interactions (e.g., 1000+ words), using a long-context transformer model (**Longformer**) and strong classical baselines (**TF-IDF**).

The project is designed for **small datasets** where multiple transcripts may belong to the same team, and it evaluates performance in a **leakage-aware** way by keeping all transcripts from the same team together during cross-validation.

---

## What this project does

- Loads transcript `.txt` files from a directory
- Loads a `scores.csv` file containing transcript scores and metadata
- Joins transcripts and scores by filename
- Runs **grouped K-fold cross-validation** (grouped by `team_id` so a team never appears in both train and test)
- Optionally runs **nested hyperparameter search** (Optuna) inside each fold
- Trains a **Longformer regression model** to predict scores
- Evaluates using:
  - MAE, MSE, RMSE
  - Pearson and Spearman correlation
  - R²
- Compares against a **TF-IDF + Ridge regression** baseline evaluated on the same folds
- Saves fold-level results and out-of-fold predictions for analysis

---

## Repository structure (suggested)

.
├── longformer_model_for_transcripts.py
├── data/
│   ├── transcripts/              # .txt transcript files
│   └── scores.csv                # scores + metadata
├── outputs/                      # generated during training
└── README.md

---

## Requirements

- Python 3.10+ recommended
- PyTorch
- Transformers
- scikit-learn
- Optuna
- pandas, numpy, scipy
- torch
- transformers
- optuna
- scikit-learn
- pandas
- numpy
- scipy

---

## Data format

### Transcripts folder

Your transcripts should be plain text files:

- One transcript per `.txt` file
- Files are stored in one directory (passed as `--data_dir`)
- Each file must have a unique filename

Example:

data/transcripts/team_01_easy.txt
data/transcripts/team_01_hard.txt
data/transcripts/team_02_easy.txt
...

---

## scores.csv format

The `scores.csv` file must contain **one row per transcript file**.

### Required columns

- filename (string): Must exactly match a transcript file name in `--data_dir` (including `.txt`).
- score (number): The target score to predict (e.g., performance rating).

### Strongly recommended columns

- team_id (string): Team identifier. All transcripts from the same team must share the same team_id. This is used for grouped CV to avoid leakage.
- difficulty (string): Transcript scenario difficulty, typically easy or hard.
- condition (string): Experimental condition for the team, e.g., increasing or decreasing difficulty order. Should be the same for both transcripts from the same team.

### Example scores.csv

filename,score,team_id,difficulty,condition
team_01_easy.txt,72,team_01,easy,increasing
team_01_hard.txt,61,team_01,hard,increasing
team_02_easy.txt,55,team_02,easy,decreasing
team_02_hard.txt,67,team_02,hard,decreasing

### Common pitfalls

- Blank rows in scores.csv
- filename values that don’t exactly match the transcript filenames
- Duplicate filenames
- A team appearing in multiple conditions (should not happen)
- A team_id not having exactly two transcripts (if your study design is 2 transcripts per team)

---

## Running the script

Your current run command will still work:

python longformer_model_for_transcripts.py \
  --data_dir /path/to/transcripts \
  --score_file /path/to/scores.csv \
  --kfold 3 \
  --do_hyperparam_search

### Key arguments

- --data_dir: path to folder of .txt transcripts
- --score_file: path to scores.csv
- --output_dir: where outputs are saved (default: ./longformer_pragmatic_cv)
- --kfold: number of folds for cross-validation (default: 5)
- --do_hyperparam_search: enable nested Optuna tuning
- --n_trials: number of Optuna trials per outer fold (default: 15)

### Useful optional flags

- --n_trials 5 for faster experimentation
- --no_tfidf_baseline to skip TF-IDF baseline
- --no_stratify_by_condition to disable condition balancing in folds
- --max_length 4096 to set transformer truncation length
- --early_stopping_patience 2 to stop if validation stops improving

---

## Outputs

The script writes results under --output_dir, typically including:

- fold_metrics.csv
  Fold-level metrics for Longformer (and TF-IDF baseline, if enabled)

- oof_predictions.csv
  Out-of-fold predictions (each transcript predicted by a model that didn’t train on that team)

- Per-fold model directories (e.g., fold_1/longformer/)
  Saved transformer checkpoints and logs

---

## Notes on evaluation

This project uses grouped cross-validation by team to estimate how well the model generalizes to unseen teams (rather than memorizing team-specific language patterns). This is important when each team contributes multiple transcripts.
