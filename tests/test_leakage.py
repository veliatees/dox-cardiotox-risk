"""The held-out individual must not influence which genes are kept."""

import numpy as np
import pandas as pd

from doxrisk.model import ExpressedGenes


def make_data(n_people=10, n_genes=6, seed=0):
    rng = np.random.default_rng(seed)
    people = [f"p{i}" for i in range(n_people)]
    genes = [f"g{j}" for j in range(n_genes)]
    baseline = pd.DataFrame(rng.uniform(0, 3, (n_people, n_genes)),
                            index=people, columns=genes)
    response = pd.DataFrame(rng.normal(size=(n_people, n_genes)),
                            index=people, columns=genes)
    return baseline, response


def test_filter_ignores_individuals_outside_training_set():
    baseline, response = make_data()
    train = response.index[:8]

    kept = ExpressedGenes(baseline).fit(response.loc[train]).columns_

    # Change the held-out individuals' expression as much as possible.
    changed = baseline.copy()
    changed.loc[response.index[8:]] = 100.0
    kept_changed = ExpressedGenes(changed).fit(response.loc[train]).columns_

    assert kept == kept_changed


def test_filter_uses_training_individuals():
    baseline, response = make_data()
    baseline["g0"] = 0.0  # not expressed anywhere: always dropped
    baseline["g1"] = 5.0  # expressed everywhere: always kept

    kept = ExpressedGenes(baseline).fit(response).columns_

    assert "g0" not in kept
    assert "g1" in kept


def test_non_gene_columns_are_kept():
    baseline, response = make_data()
    X = response.assign(troponin_0=1.0)

    kept = ExpressedGenes(baseline).fit(X).columns_

    assert "troponin_0" in kept
