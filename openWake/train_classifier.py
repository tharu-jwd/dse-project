"""
train_classifier.py
Plan Section 5-6 / Step 8.3-8.5: train BRAIN 2 (the command classifier) on
the embeddings produced by extract_features.py.

Pick ONE classifier below by uncommenting its block (and commenting the
others out), then run:

    python train_classifier.py

Each run saves everything to results/<ClassifierName>/:
  - model.joblib            the fitted pipeline (StandardScaler + classifier)
  - classification_report.txt
  - confusion_matrix.csv    raw counts, same label order as the plot
  - confusion_matrix.png    heatmap for quick visual comparison
  - metrics.json            accuracy, macro/weighted F1, false-accepts from 'none'

Compare classifiers by diffing metrics.json / confusion_matrix.png across
the result folders.
"""

import json
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score

FEATURES_DIR = "features"
RESULTS_DIR = "results"


def load_split(name):
    data = np.load(f"{FEATURES_DIR}/{name}.npz", allow_pickle=True)
    return data["X"], data["y"], list(data["labels"])


# =====================================================================
# Choose ONE classifier: uncomment its block, leave the other two
# commented out. Only one `clf = ...` should be active at a time.
# =====================================================================

# --- Option 1: Logistic Regression (recommended baseline, plan Section 5) ---
# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression(class_weight="balanced", max_iter=1000)

# --- Option 2: Random Forest (escalate here if minority-class recall is weak) ---
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(class_weight="balanced", n_estimators=300)

# --- Option 3: Neural Network (MLP) ---
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500)
# NOTE: sklearn's MLPClassifier does NOT support class_weight / sample_weight
# (plan Section 5). Class imbalance is unmitigated with this option unless
# you oversample the minority classes (delete/save/stop/submit) manually
# before fit(), or switch to a PyTorch MLP with a weighted loss.


# =====================================================================


def main():
    X_train, y_train, labels = load_split("train")
    X_val, y_val, _ = load_split("val")
    print(f"train: X={X_train.shape}, val: X={X_val.shape}, labels={labels}")

    model_name = clf.__class__.__name__
    out_dir = f"{RESULTS_DIR}/{model_name}"
    import os
    os.makedirs(out_dir, exist_ok=True)

    brain = make_pipeline(StandardScaler(), clf)
    print(f"Training {model_name}...")
    brain.fit(X_train, y_train)

    pred = brain.predict(X_val)

    report_text = classification_report(y_val, pred, labels=labels)
    print("\n--- classification_report (val) ---")
    print(report_text)

    cm = confusion_matrix(y_val, pred, labels=labels)
    print("--- confusion_matrix (val) ---")
    print("labels order:", labels)
    print(cm)

    # False-accepts from 'none' into a real command, vs command-vs-command
    # confusion (plan Section 6) — these have very different costs.
    wrong_from_none = None
    total_none = None
    if "none" in labels:
        wrong_from_none = int(np.sum((y_val == "none") & (pred != "none")))
        total_none = int(np.sum(y_val == "none"))
        print(f"\nFalse-accepts from 'none': {wrong_from_none}/{total_none}")

    # ---- save model ----
    joblib.dump(brain, f"{out_dir}/model.joblib")

    # ---- save classification report ----
    with open(f"{out_dir}/classification_report.txt", "w") as f:
        f.write(report_text)

    # ---- save confusion matrix as csv ----
    with open(f"{out_dir}/confusion_matrix.csv", "w") as f:
        f.write("," + ",".join(labels) + "\n")
        for label, row in zip(labels, cm):
            f.write(label + "," + ",".join(str(v) for v in row) + "\n")

    # ---- save confusion matrix as a heatmap png ----
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {model_name} (val)")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center",
                         color="white" if v > thresh else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ---- save metrics summary ----
    metrics = {
        "model": model_name,
        "params": clf.get_params(),
        "accuracy": float(np.mean(pred == y_val)),
        "macro_f1": float(f1_score(y_val, pred, labels=labels, average="macro")),
        "weighted_f1": float(f1_score(y_val, pred, labels=labels, average="weighted")),
        "false_accepts_from_none": wrong_from_none,
        "total_none": total_none,
        "labels": labels,
    }
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\nSaved model + reports to {out_dir}/")


if __name__ == "__main__":
    main()
