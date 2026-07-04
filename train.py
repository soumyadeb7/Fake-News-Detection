"""
Fake News Detection - Classification Task
Author: Soumyadeb Bakshi

Approach: TF-IDF (word features) + Logistic Regression
"""

import json
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# I fix the random seed so the results stay the same on repeat runs.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# I use paths relative to this file so it still works from another folder.
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"
results_dir = base_dir / "results"
results_dir.mkdir(exist_ok=True)

# I load the two files and mark them as fake or real.
fake_df = pd.read_csv(data_dir / "Fake.csv")
true_df = pd.read_csv(data_dir / "True.csv")

fake_df["label"] = 1
true_df["label"] = 0

# I combine them into one table so the model can learn from both classes.
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df[["title", "text", "label"]]

# I join the title and article text because the headline can itself be useful.
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)        # keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()     # collapse repeated whitespace
    return text

# This cleaning step is short and easy to explain.
df["clean_content"] = df["content"].apply(clean_text)

X = df["clean_content"]
y = df["label"]

# I split the data in two steps so the 70:10:20 ratio is easy to explain.
# First, I keep 70% for training and 30% for validation plus testing.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=SEED,
    stratify=y,
)

# From that remaining 30%, I keep 10% for validation and 20% for testing.
# This is the same as splitting the remaining part into 1/3 validation and 2/3 test.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=2 / 3,
    random_state=SEED,
    stratify=y_temp,
)

print(f"Train size: {len(X_train)} ({len(X_train) / len(df):.1%})")
print(f"Val size:   {len(X_val)} ({len(X_val) / len(df):.1%})")
print(f"Test size:  {len(X_test)} ({len(X_test) / len(df):.1%})")

# TF-IDF turns text into numbers that a simple classifier can use.
# We fit it only on the training data so the validation and test sets remain unseen.
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2),
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression is a simple linear model that is easy to explain.
# It learns one weight for each important word or word-pair.
model = LogisticRegression(max_iter=1000, random_state=SEED)
model.fit(X_train_vec, y_train)

# We check the validation set before using the test set, so the final evaluation
# is not influenced by repeated model tuning.
val_preds = model.predict(X_val_vec)
val_acc = accuracy_score(y_val, val_preds)
print(f"\nValidation Accuracy: {val_acc:.4f}")

# We evaluate once on the untouched test set to estimate how well the model
# generalises to new examples.
test_preds = model.predict(X_test_vec)
test_probs = model.predict_proba(X_test_vec)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, test_preds),
    "precision": precision_score(y_test, test_preds, zero_division=0),
    "recall": recall_score(y_test, test_preds, zero_division=0),
    "f1_score": f1_score(y_test, test_preds, zero_division=0),
    "auc_roc": roc_auc_score(y_test, test_probs),
}

print("\n===== TEST SET METRICS =====")
for k, v in metrics.items():
    print(f"{k:10s}: {v:.4f}")

print("\nClassification report:\n", classification_report(y_test, test_preds, target_names=["Real", "Fake"]))

# Saving the metrics keeps the output reproducible and easy to report.
with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# A confusion matrix gives a simple visual summary of the model's errors.
cm = confusion_matrix(y_test, test_preds)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Real", "Fake"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["Real", "Fake"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - Test Set")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.colorbar(im)
plt.tight_layout()
plt.savefig(results_dir / "confusion_matrix.png", dpi=150)
print(f"\nSaved: {results_dir / 'confusion_matrix.png'}")
print(f"Saved: {results_dir / 'metrics.json'}")

# These top-weighted words are the clearest way to explain the model's decision.
feature_names = np.array(vectorizer.get_feature_names_out())
coefs = model.coef_[0]

top_fake_idx = np.argsort(coefs)[-15:][::-1]
top_real_idx = np.argsort(coefs)[:15]

print("\nTop words pushing prediction towards FAKE:")
for i in top_fake_idx:
    print(f"  {feature_names[i]:20s} weight={coefs[i]:.3f}")

print("\nTop words pushing prediction towards REAL:")
for i in top_real_idx:
    print(f"  {feature_names[i]:20s} weight={coefs[i]:.3f}")
