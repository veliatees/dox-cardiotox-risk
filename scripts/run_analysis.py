"""Run the full analysis from raw data and write the results to results/.

    python scripts/run_analysis.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from doxrisk import data, model  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Inputs and target, one row per individual.
logcpm = data.log_cpm(data.load_counts())
response, baseline = data.response_features(logcpm, dose=0.625)
trop = data.troponin_table(data.load_troponin()).loc[response.index]

y = (trop[5.0] - trop[0.0]).rename("log2_troponin_increase_5uM")
troponin = pd.DataFrame({"troponin_0": trop[0.0], "troponin_0.625": trop[0.625]})
both = pd.concat([troponin, response], axis=1)
print(f"{len(y)} individuals, {response.shape[1]} genes before filtering")

# Outer loop: leave one individual out. The gene models choose their penalty
# in an inner 5-fold CV on the remaining 43 (nested cross-validation).
models = {
    "mean-only": (model.mean_model(), troponin),
    "troponin-only": (model.troponin_model(), troponin),
    "genes-only": (model.gene_model(baseline), response),
    "troponin + genes": (model.gene_model(baseline), both),
}
scores, preds = {}, {"observed": y}
for name, (estimator, X) in models.items():
    preds[name], scores[name] = model.evaluate(estimator, X, y)
    print(f"{name:18s} RMSE {scores[name]['rmse']:.3f}  "
          f"Spearman {scores[name]['spearman']:.2f}")

pd.DataFrame(scores).T.round(3).to_csv(RESULTS / "scores.csv")
pd.DataFrame(preds).round(4).to_csv(RESULTS / "predictions.csv")

# Stability selection: 1,000 elastic-net fits on random halves of the data.
freq, params = model.selection_frequency(baseline, response, y, n_fits=1000)
freq = freq.sort_values(ascending=False).rename("selection_frequency")
freq[freq > 0].round(3).to_csv(RESULTS / "selection_frequency.csv")
print(f"penalty: alpha {params['alpha']:.4f}, l1_ratio {params['l1_ratio']}")
print(f"genes selected in more than half of the fits: {(freq > 0.5).sum()}")
print(freq.head(15).round(3).to_string())
