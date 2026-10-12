"""Is the genes-only prediction just predicting the ELISA plate?

Troponin was measured on 7 ELISA plates. This compares a plate-only baseline
with the genes-only predictions from run_analysis.py, and checks whether the
predictions still track the target within plates.

    python scripts/plate_check.py   (after run_analysis.py)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from doxrisk import data  # noqa: E402

pred = pd.read_csv(ROOT / "results" / "predictions.csv", index_col=0)
y, genes = pred["observed"], pred["genes-only"]
plate = data.load_troponin().groupby("individual")["elisa_batch"].first().loc[y.index]

onehot = pd.get_dummies(plate).astype(float)
plate_pred = cross_val_predict(LinearRegression(), onehot, y, cv=LeaveOneOut())
plate_rmse = np.sqrt(np.mean((y - plate_pred) ** 2))


def centred(s):
    return s - s.groupby(plate).transform("mean")


summary = (f"plate-only baseline (leave-one-out): RMSE {plate_rmse:.3f}, "
           f"Spearman {spearmanr(y, plate_pred).statistic:.2f}\n"
           f"share of target variance explained by plate: "
           f"{1 - centred(y).var() / y.var():.2f}\n"
           f"genes-only prediction vs target within plates: Spearman "
           f"{spearmanr(centred(y), centred(genes)).statistic:.2f}\n")
(ROOT / "results" / "plate_check.txt").write_text(summary)
print(summary)
