


import os
import argparse
import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset
from transformers import (
    LongformerTokenizerFast, LongformerForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import optuna


# Set global seed for reproducibility

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# Load transcript texts and scores

def load_data(text_dir, score_file):
    data = []
    try:
        score_df = pd.read_csv(score_file)
        if 'filename' not in score_df.columns or 'score' not in score_df.columns:
            raise ValueError("Score file must contain 'filename' and 'score' columns")
        score_map = score_df.set_index('filename')['score'].to_dict()

        for filename in os.listdir(text_dir):
            if filename.endswith(".txt") and filename in score_map:
                file_path = os.path.join(text_dir, filename)
                if os.path.isfile(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        data.append((content, score_map[filename]))
        if not data:
            raise ValueError("No matching text-score pairs found.")

        return pd.DataFrame(data, columns=["text", "score"])
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


# Dataset Class

class LongformerDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_length=4096):
        self.encodings = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(scores, dtype=torch.float)
        # First token global
        self.global_masks = torch.zeros_like(self.encodings['attention_mask'])
        self.global_masks[:, 0] = 1

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'global_attention_mask': self.global_masks[idx],
            'labels': self.labels[idx]
        }


# Metrics

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions

    if isinstance(preds, tuple):
        preds = preds[0]

    print(f"Predictions type: {type(preds)}, shape: {preds.shape if hasattr(preds, 'shape') else 'no shape'}")
    print(f"Labels type: {type(labels)}, shape: {labels.shape if hasattr(labels, 'shape') else 'no shape'}")

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if len(preds.shape) > 1 and preds.shape[1] == 1:
        preds = preds.squeeze(-1)

    # Initialize all metrics
    mae, mse, r2, pearson = 0.0, 0.0, 0.0, 0.0

    try:
        mae = mean_absolute_error(labels, preds)
    except Exception as e:
        print(f"Error calculating MAE: {e}")

    try:
        mse = mean_squared_error(labels, preds)
    except Exception as e:
        print(f"Error calculating MSE: {e}")

    try:
        if len(np.unique(labels)) > 1:
            r2 = r2_score(labels, preds)
        else:
            r2 = 0.0
    except Exception as e:
        print(f"Error calculating R2: {e}")

    try:
        if len(labels) > 1 and len(np.unique(labels)) > 1 and len(np.unique(preds)) > 1:
            pearson = pearsonr(labels, preds)[0]
        else:
            pearson = 0.0
    except Exception as e:
        print(f"Error calculating Pearson: {e}")

    return {
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "pearson": pearson
    }


# Model Initialization

def model_init():
    model = LongformerForSequenceClassification.from_pretrained(
        "allenai/longformer-base-4096",
        num_labels=1,
        problem_type="regression"
    )
    model.gradient_checkpointing_enable()
    return model


# Hyperparameter Space

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [2, 4]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 6),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 0.1, log=True),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 300),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4])
    }


# Train & Evaluate

def train_and_evaluate(train_texts, train_scores, val_texts, val_scores, hyperparams, output_dir, tokenizer):
    train_dataset = LongformerDataset(train_texts, train_scores, tokenizer)
    val_dataset = LongformerDataset(val_texts, val_scores, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  # evaluation_strategy has been deprecated
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_dir=os.path.join(output_dir, 'logs'),
        report_to=None,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_r2",
        greater_is_better=True,
        **hyperparams
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(output_dir)  # Explicit save
    return trainer.evaluate(), trainer



# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--score_file', required=True)
    parser.add_argument('--output_dir', default='./longformer_pragmatic_cv')
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--do_hyperparam_search', action='store_true')
    parser.add_argument('--n_trials', type=int, default=15)  # Number of Optuna trials
    args = parser.parse_args()

    set_global_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_data(args.data_dir, args.score_file)
    texts = df['text'].tolist()
    scores = df['score'].tolist()
    tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")

    #  1. Hyperparameter Search (Optional)
    if args.do_hyperparam_search:
        print("--- Finding best hyperparameters using a temporary split of the full dataset ---")
        # This split is temporary and only for the purpose of running Optuna efficiently.
        temp_train_texts, temp_val_texts, temp_train_scores, temp_val_scores = train_test_split(
            texts, scores, test_size=0.25, random_state=42  # e.g., 75/25 split
        )

        def objective(trial):
            hyperparams = hp_space(trial)
            trial_output_dir = os.path.join(args.output_dir, f"tuning_trial_{trial.number}")
            os.makedirs(trial_output_dir, exist_ok=True)

            # The train_and_evaluate function trains a model and evaluates it.
            metrics, _ = train_and_evaluate(
                temp_train_texts, temp_train_scores,
                temp_val_texts, temp_val_scores,
                hyperparams, trial_output_dir, tokenizer
            )
            # Optuna will try to maximize the R-squared on the temporary validation set.
            return metrics.get("eval_r2", 0.0)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials)
        best_hyperparams = study.best_trial.params
        print("\n--- Best hyperparameters found ---")
        for k, v in best_hyperparams.items():
            print(f"{k}: {v}")

    else:
        # If not do_hyperparam_search, use predefined hyperparameters
        best_hyperparams = {
            "learning_rate": 5e-5,
            "per_device_train_batch_size": 2,
            "num_train_epochs": 4,
            "weight_decay": 0.01,
            "warmup_steps": 0,
            "gradient_accumulation_steps": 1
        }
        print("\n--- Using default hyperparameters ---")

    # 2. K-Fold Cross-Validation on the FULL dataset, this is because dataset is fairly small
    print(f"\n--- Running {args.kfold}-Fold Cross-Validation on the FULL dataset ---")
    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)
    all_metrics = {"mae": [], "mse": [], "r2": [], "pearson": []}

    # We convert to numpy arrays for easier indexing with KFold splits
    texts_np = np.array(texts)
    scores_np = np.array(scores)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(texts_np), 1):
        print(f"\n==== Fold {fold_idx}/{args.kfold} ====")
        fold_output_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_output_dir, exist_ok=True)

        train_texts_fold = texts_np[train_idx].tolist()
        train_scores_fold = scores_np[train_idx].tolist()
        val_texts_fold = texts_np[val_idx].tolist()
        val_scores_fold = scores_np[val_idx].tolist()

        # The same 'best_hyperparams' are used for every fold
        metrics, _ = train_and_evaluate(
            train_texts_fold, train_scores_fold,
            val_texts_fold, val_scores_fold,
            best_hyperparams, fold_output_dir, tokenizer
        )

        print(
            f"Fold {fold_idx} metrics: MAE={metrics['eval_mae']:.4f}, MSE={metrics['eval_mse']:.4f}, R2={metrics['eval_r2']:.4f}, Pearson={metrics['eval_pearson']:.4f}")
        for k in all_metrics:
            all_metrics[k].append(metrics['eval_' + k])

    #  3. Aggregate and Report Final Results
    print("\n==== K-Fold CV Final Results ====")
    for metric_name, values in all_metrics.items():
        print(f"{metric_name.upper()} Mean: {np.mean(values):.4f} | Std: {np.std(values):.4f}")

    pd.DataFrame(all_metrics).to_csv(os.path.join(args.output_dir, "kfold_results.csv"), index=False)
    print("Final K-Fold results saved.")


if __name__ == "__main__":
    main()


