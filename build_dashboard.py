"""Build dashboard.html: embeds test-set predictions + model summary into dashboard_template.html.

Run after logistic_model.py:  python build_dashboard.py
"""
import json
from pathlib import Path

import logistic_model as lm  # re-trains the models (fast) and writes outputs/

data = {
    "y": lm.y_test.tolist(),
    "proba": {name: [round(float(p), 5) for p in lm.models[name].predict_proba(lm.X_test)[:, 1]]
              for name in ["baseline", "balanced"]},
    "results": {k: lm.results[k] for k in ["baseline", "balanced", "balanced_tuned"]},
    "cv": lm.results["cv_balanced_0.5"],
    "coefficients": lm.results["coefficients_balanced"],
    "tuned_threshold": lm.best_t,
    "split": lm.split,
    "n_total": int(len(lm.df)),
    "n_fraud": int(lm.y.sum()),
}

template = Path("dashboard_template.html").read_text(encoding="utf-8")
html = template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
Path("dashboard.html").write_text(html, encoding="utf-8")
print(f"Wrote dashboard.html ({len(html) / 1024:.0f} KB)")
