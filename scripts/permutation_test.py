"""Check the genes-only result against models trained on shuffled targets.

Shuffling y across individuals breaks any real link between gene response and
troponin increase while keeping everything else (the pipeline, the data, the
number of genes). If the cross-validation leaked information, shuffled targets
would still score well. Outer 5-fold CV is used here instead of leave-one-out
to keep the run time manageable; the observed score is recomputed with the
same scheme so that the comparison is like for like.

    python scripts/permutation_test.py [n_permutations]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from doxrisk import data, model  # noqa: E402

n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 50

logcpm = data.log_cpm(data.load_counts())
response, baseline = data.response_features(logcpm, dose=0.625)
trop = data.troponin_table(data.load_troponin()).loc[response.index]
y = trop[5.0] - trop[0.0]

outer = KFold(5, shuffle=True, random_state=0)


def rmse(target):
    pred = cross_val_predict(model.gene_model(baseline), response, target, cv=outer)
    return np.sqrt(np.mean((target - pred) ** 2))


observed = rmse(y)
rng = np.random.default_rng(0)
null = []
for i in range(n_perm):
    shuffled = pd.Series(rng.permutation(y.values), index=y.index)
    null.append(rmse(shuffled))
    print(f"permutation {i + 1}/{n_perm}: RMSE {null[-1]:.3f}", flush=True)

null = np.array(null)
p = (1 + np.sum(null <= observed)) / (1 + n_perm)
pd.DataFrame({"rmse": null}).round(4).to_csv(ROOT / "results" / "permutation_null.csv",
                                            index=False)
summary = (f"observed RMSE (outer 5-fold): {observed:.3f}\n"
           f"shuffled targets: median {np.median(null):.3f}, "
           f"min {null.min():.3f} over {n_perm} permutations\n"
           f"permutation p-value: {p:.3f}\n")
(ROOT / "results" / "permutation_test.txt").write_text(summary)
print(summary)
