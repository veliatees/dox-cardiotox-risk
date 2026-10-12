"""Models, cross-validation and stability selection.

Every step that looks across individuals (gene filtering, scaling, choosing
the penalty) sits inside a scikit-learn Pipeline, so cross-validation refits
it on the training individuals of each fold and the held-out individual never
influences it.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

L1_RATIOS = [0.1, 0.5, 0.9, 1.0]


class ExpressedGenes(BaseEstimator, TransformerMixin):
    """Keep genes expressed in most of the training individuals.

    A gene is kept if its untreated log2-CPM is above `min_logcpm` (1 means
    CPM > 1) in more than `min_fraction` of the individuals seen in `fit`.
    `baseline` holds untreated expression for every individual; only the rows
    of the individuals passed to `fit` are read. Columns that are not genes
    (e.g. troponin) are always kept.
    """

    def __init__(self, baseline, min_logcpm=1.0, min_fraction=0.5):
        self.baseline = baseline
        self.min_logcpm = min_logcpm
        self.min_fraction = min_fraction

    def fit(self, X, y=None):
        train = self.baseline.loc[X.index]
        expressed = (train > self.min_logcpm).mean() > self.min_fraction
        genes = set(expressed.index[expressed])
        self.columns_ = [c for c in X.columns
                         if c in genes or c not in self.baseline.columns]
        return self

    def transform(self, X):
        return X[self.columns_]


def gene_model(baseline, inner_folds=5, seed=0):
    """Gene filter -> scaling -> elastic net with penalty chosen by inner CV."""
    return make_pipeline(
        ExpressedGenes(baseline),
        StandardScaler(),
        ElasticNetCV(l1_ratio=L1_RATIOS, cv=KFold(inner_folds, shuffle=True,
                                                  random_state=seed),
                     max_iter=50_000, n_jobs=-1),
    )


def troponin_model():
    return make_pipeline(StandardScaler(), LinearRegression())


def mean_model():
    return DummyRegressor(strategy="mean")


def evaluate(model, X, y):
    """Leave-one-out predictions and their scores (outer loop of nested CV)."""
    pred = pd.Series(cross_val_predict(model, X, y, cv=LeaveOneOut()),
                     index=y.index)
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    # A constant model has no ranking; leave-one-out means only differ by the
    # held-out value itself, which would give a meaningless rho of -1.
    constant = isinstance(model, DummyRegressor)
    rho = np.nan if constant else spearmanr(y, pred).statistic
    return pred, {"rmse": rmse, "spearman": rho}


def selection_frequency(baseline, X, y, n_fits=1000, fraction=0.5, seed=0):
    """Fraction of subsample fits in which each gene gets a non-zero weight.

    The penalty is chosen once by cross-validation on all individuals; each
    fit then uses a random half of the individuals (without replacement).
    """
    tuned = gene_model(baseline).fit(X, y)[-1]
    fixed = make_pipeline(ExpressedGenes(baseline), StandardScaler(),
                          ElasticNet(alpha=tuned.alpha_, l1_ratio=tuned.l1_ratio_,
                                     max_iter=50_000))
    rng = np.random.default_rng(seed)
    n = int(round(fraction * len(y)))
    counts = pd.Series(0, index=X.columns, dtype=float)
    for _ in range(n_fits):
        rows = rng.choice(len(y), size=n, replace=False)
        fit = clone(fixed).fit(X.iloc[rows], y.iloc[rows])
        kept = fit[0].columns_
        counts[np.array(kept)[fit[-1].coef_ != 0]] += 1
    return counts / n_fits, {"alpha": tuned.alpha_, "l1_ratio": tuned.l1_ratio_}
