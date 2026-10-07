"""Logistic regression fraud model on credit_card_fraud_10k.csv.

Run: python logistic_model.py  -> prints metrics and writes outputs/ (metrics.json + plots).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("credit_card_fraud_10k.csv")
df["is_night"] = (df["transaction_hour"] <= 5).astype(int)  # fraud risk vs. hour is not linear
X = df.drop(columns=["transaction_id", "is_fraud"])
y = df["is_fraud"]

num_cols = ["amount", "transaction_hour", "device_trust_score", "velocity_last_24h", "cardholder_age"]
bin_cols = ["foreign_transaction", "location_mismatch", "is_night"]
cat_cols = ["merchant_category"]


def make_model(class_weight=None):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("bin", "passthrough", bin_cols),
        ("cat", OneHotEncoder(drop="first"), cat_cols),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(class_weight=class_weight, max_iter=1000))])


def evaluate(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def print_result(name, r):
    c = r["confusion"]
    print(f"\n=== {name} (threshold={r['threshold']:.2f}) ===")
    for k in ["roc_auc", "pr_auc", "recall", "precision", "f1"]:
        print(f"{k:10s}: {r[k]:.4f}")
    print(f"confusion : TN={c['tn']} FP={c['fp']} FN={c['fn']} TP={c['tp']}")


# 80/20 train/test split, stratified so both sets keep the ~1.5% fraud rate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
split = {name: {"rows": int(len(t)), "fraud": int(t.sum()), "fraud_rate": round(float(t.mean()), 4)}
         for name, t in [("train", y_train), ("test", y_test)]}
print("=== Train/test split (80/20, stratified) ===")
for name, s in split.items():
    print(f"{name:5s}: {s['rows']:,} rows ({s['rows'] / len(y):.0%}), {s['fraud']} fraud ({s['fraud_rate']:.2%})")
skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

results = {}
models = {}
for name, cw in [("baseline", None), ("balanced", "balanced")]:
    models[name] = make_model(cw).fit(X_train, y_train)
    results[name] = evaluate(y_test, models[name].predict_proba(X_test)[:, 1], 0.5)

# Threshold chosen on out-of-fold train predictions only -> test set stays untouched
oof = cross_val_predict(make_model("balanced"), X_train, y_train, cv=skf, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y_train, oof)
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
best_t = float(thr[np.argmax(f1s)])
results["balanced_tuned"] = evaluate(y_test, models["balanced"].predict_proba(X_test)[:, 1], best_t)

for name, r in results.items():
    print_result(name, r)

# 5-fold CV on full data, balanced model, threshold 0.5
cv = cross_validate(make_model("balanced"), X, y, cv=skf, scoring=["roc_auc", "recall", "precision", "f1"])
results["cv_balanced_0.5"] = {k: {"mean": round(cv[f"test_{k}"].mean(), 4), "std": round(cv[f"test_{k}"].std(), 4)}
                              for k in ["roc_auc", "recall", "precision", "f1"]}
print("\n=== 5-fold CV, balanced, threshold 0.5 (mean +/- std) ===")
for k, v in results["cv_balanced_0.5"].items():
    print(f"{k:10s}: {v['mean']:.4f} +/- {v['std']:.4f}")

model = models["balanced"]
names = model.named_steps["pre"].get_feature_names_out()
coefs = pd.Series(model.named_steps["clf"].coef_[0], index=names).sort_values(key=abs, ascending=False)
results["coefficients_balanced"] = coefs.round(3).to_dict()
print("\n=== Coefficients (balanced model) ===")
print(coefs.round(3).to_string())

results["split"] = split
(OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

# Plots
fig, ax = plt.subplots(figsize=(6, 5))
for name in ["baseline", "balanced"]:
    p = models[name].predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, p)
    ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "roc_curve.png", dpi=120)

fig, ax = plt.subplots(figsize=(6, 5))
for name in ["baseline", "balanced"]:
    p = models[name].predict_proba(X_test)[:, 1]
    pr, rc, _ = precision_recall_curve(y_test, p)
    ax.plot(rc, pr, label=f"{name} (AP={results[name]['pr_auc']:.3f})")
ax.axhline(y_test.mean(), linestyle="--", color="grey", linewidth=1, label="fraud base rate")
ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve (test set)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "pr_curve.png", dpi=120)

pred = (model.predict_proba(X_test)[:, 1] >= best_t).astype(int)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["Legit", "Fraud"])
disp.plot(cmap="Blues", colorbar=False)
disp.ax_.set_title(f"Confusion matrix (balanced, threshold={best_t:.2f})")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=120)
print(f"\nSaved metrics and plots to {OUT.resolve()}")
