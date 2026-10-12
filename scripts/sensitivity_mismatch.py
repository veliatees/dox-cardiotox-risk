"""Repeat the genes-only evaluation without the 5 cell lines flagged `mismatch`.

    python scripts/sensitivity_mismatch.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from doxrisk import data, model  # noqa: E402

logcpm = data.log_cpm(data.load_counts())
response, baseline = data.response_features(logcpm, dose=0.625)
trop_long = data.load_troponin()
flagged = set(trop_long.loc[trop_long["mismatch"], "individual"])
keep = [p for p in response.index if p not in flagged]

trop = data.troponin_table(trop_long).loc[keep]
y = trop[5.0] - trop[0.0]
_, genes = model.evaluate(model.gene_model(baseline), response.loc[keep], y)
_, mean = model.evaluate(model.mean_model(), response.loc[keep], y)

summary = (f"without mismatch-flagged lines ({len(keep)} individuals):\n"
           f"mean-only RMSE {mean['rmse']:.3f}\n"
           f"genes-only RMSE {genes['rmse']:.3f}, Spearman {genes['spearman']:.2f}\n")
(ROOT / "results" / "sensitivity_mismatch.txt").write_text(summary)
print(summary)
