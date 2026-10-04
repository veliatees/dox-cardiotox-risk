"""Load the Knowles et al. (eLife 2018) data and build per-individual tables.

Everything here is computed per sample or per individual, so it is safe to do
before cross-validation: no step looks at other individuals. Steps that do
depend on other individuals (gene filtering, scaling, model fitting) live in
the modelling code and are fitted inside each training fold.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
DOSES = [0.0, 0.625, 1.25, 2.5, 5.0]

# Library sizes have a clear gap between 0.68M and 6.3M reads (median 24.6M).
# Any threshold inside that gap removes the same 7 failed libraries.
MIN_DEPTH = 2_000_000


def load_troponin():
    """One row per sample: individual, dose, troponin, ELISA batch, mismatch flag."""
    trop = pd.read_csv(DATA / "troponin.txt", sep="\t")
    trop = trop.rename(columns={"cell_line": "individual", "dosage": "dose",
                                "experiment": "elisa_batch"})
    trop["sample"] = trop["id"].str[1:]  # "s001-c64-0.000" -> "001-c64-0.000"
    return trop.set_index("sample")[["individual", "dose", "troponin",
                                     "elisa_batch", "mismatch"]]


def load_counts():
    """Raw gene counts, samples x genes, with the two sequencing runs summed."""
    counts = pd.read_csv(DATA / "gene-counts.txt.gz", sep="\t")
    genes = [c for c in counts.columns if c.startswith("ENSG")]
    sample = counts["filename"].str.split("-").str[:3].str.join("-")
    return counts[genes].groupby(sample).sum()


def sample_info(samples):
    """Individual and dose parsed from sample IDs like '001-c64-0.000'."""
    parts = pd.Index(samples).str.split("-")
    return pd.DataFrame({"individual": parts.str[1],
                         "dose": parts.str[2].astype(float)},
                        index=samples)


def log_cpm(counts, min_depth=MIN_DEPTH):
    """Drop failed libraries, then log2(counts per million + 1) for every gene."""
    depth = counts.sum(axis=1)
    counts = counts[depth >= min_depth]
    cpm = counts.div(counts.sum(axis=1), axis=0) * 1_000_000
    return np.log2(cpm + 1)


def response_features(logcpm, dose=0.625):
    """Per individual: expression at `dose` minus expression at dose 0.

    Returns (response, baseline), both individuals x genes. `baseline` is the
    untreated expression, kept so that expression filtering can be fitted on
    training individuals only. Individuals missing either sample are dropped.
    """
    info = sample_info(logcpm.index)
    untreated = logcpm[info["dose"] == 0.0].set_axis(
        info.loc[info["dose"] == 0.0, "individual"])
    treated = logcpm[info["dose"] == dose].set_axis(
        info.loc[info["dose"] == dose, "individual"])
    both = untreated.index.intersection(treated.index).sort_values()
    return treated.loc[both] - untreated.loc[both], untreated.loc[both]


def troponin_table(trop):
    """Individuals x doses, log troponin."""
    wide = trop.pivot_table(index="individual", columns="dose", values="troponin")
    return np.log(wide)
