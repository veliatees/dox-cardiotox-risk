# dox-cardiotox-risk

Does the transcriptomic response of a person's heart muscle cells to a low dose of doxorubicin predict how much damage the cells take at a high dose, and does it add information beyond troponin, the biomarker already used in the clinic?

## Result in brief

In 44 individuals, the gene response at 0.625 µM predicted the rise in troponin at 5 µM better than a mean-only and a troponin-only baseline (leave-one-out RMSE 0.54 against 0.65 and 0.67, in log2 units; Spearman 0.55). Troponin at the low dose carried no information about the later rise. All 50 models trained on shuffled targets scored worse than the real one (p = 0.02). The signal is modest and spread over many genes: no small, stable gene panel emerged, and the advantage shrinks when the five cell lines flagged `mismatch` in the source data are removed.

## Data

iPSC-derived cardiomyocytes, each line exposed to five doxorubicin concentrations (0, 0.625, 1.25, 2.5 and 5 µM) for 24 hours, with RNA-seq gene counts and cardiac troponin I measured in the culture medium. The released files contain 46 cell lines. Source: Knowles et al., eLife 2018, doi:10.7554/eLife.33480, and the repository github.com/davidaknowles/dox. The data are not included here; download them with:

```bash
mkdir data
curl -L -o data/gene-counts.txt.gz https://raw.githubusercontent.com/davidaknowles/dox/master/data/gene-counts-round-two.txt.gz
curl -L -o data/troponin.txt https://raw.githubusercontent.com/davidaknowles/dox/master/data/troponin-processed.txt
```

## Design

- **Unit**: one individual. 7 of the 230 libraries failed (under 2M reads; the median is 24.6M), which removes the untreated sample of two individuals and leaves 44 with both an untreated and a 0.625 µM library.
- **Input**: per gene, log2-CPM at 0.625 µM minus log2-CPM untreated.
- **Target**: log2 of troponin at 5 µM divided by the same individual's untreated troponin. Raw troponin at 5 µM mostly reflects each individual's untreated level (Spearman 0.71 between the two), so the target is the rise rather than the level.
- **Models**, all with the same target:
  - mean-only: predicts the training mean
  - troponin-only: linear regression on log2 troponin untreated and at 0.625 µM
  - genes-only: expression filter (untreated CPM > 1 in more than half of the training individuals), standardisation, elastic net with `l1_ratio` in {0.1, 0.5, 0.9, 1} and the penalty chosen by 5-fold cross-validation inside the training individuals
  - troponin + genes: the same, with both troponin values added as inputs
- **Evaluation**: nested cross-validation with leave-one-out as the outer loop. Every step that looks across individuals (gene filter, scaling, penalty) sits in a scikit-learn `Pipeline` and is refitted on the 43 training individuals of each fold. `tests/test_leakage.py` checks that the held-out individual cannot change which genes pass the filter.

## Results

| Model | RMSE (log2) | Spearman |
|---|---|---|
| Mean-only | 0.652 | – |
| Troponin-only | 0.667 | 0.11 |
| Genes-only | 0.540 | 0.55 |
| Troponin + genes | 0.540 | 0.54 |

An RMSE of 0.54 in log2 units is a typical error of a factor of 1.45 in the predicted troponin rise. Per-individual predictions are in `results/predictions.csv`.

**Checks**

- *Permutation test* (`results/permutation_test.txt`): with outer 5-fold CV to keep the run time down, the real targets give RMSE 0.570; 50 shuffled targets give a median of 0.676 and a minimum of 0.599. All 50 are worse, so p = 0.02, the smallest value 50 permutations can give.
- *ELISA plates* (`results/plate_check.txt`): troponin was measured on 7 plates, which explain 18% of the target's variance. A plate-only baseline scores RMSE 0.694, and within plates the gene predictions still track the target (Spearman 0.60), so the gene model is not just predicting the plate.
- *Lines flagged `mismatch`* (`results/sensitivity_mismatch.txt`): without these 5 lines (39 individuals), genes-only reaches RMSE 0.608 against 0.644 for mean-only (Spearman 0.42). The advantage is smaller; two of the five lines are among the five most strongly affected individuals.

**Selection frequency** (`results/selection_frequency.csv`): with the penalty chosen on all 44 individuals (`alpha` 0.0048, `l1_ratio` 0.1, close to ridge regression), the elastic net was refitted 1,000 times on random halves of the individuals. 5,734 genes were selected at least once, 27 in at least 30% of the fits, and one (ENSG00000206530) in more than half. The prediction rests on many genes with small weights, not on a small panel; the ranking is a list of candidates to follow up, not a validated panel.

## Limitations

- **Cell culture rather than patients.** iPSC-derived cardiomyocytes are less mature than adult heart cells, lack the surrounding tissue, and were exposed for 24 hours only; troponin was measured in culture medium, not in blood.
- **RNA-seq rather than mass spectrometry.** Changes in transcripts need not match changes in proteins, and clinical biomarkers are proteins.
- **Small sample.** 44 individuals with one measurement per dose and no external validation set, so the estimates above are uncertain.
- **Noisy target.** 4 of the 44 individuals have lower troponin at 5 µM than untreated, which is measurement noise rather than protection.
- The exploratory notebook `explore.ipynb` filters genes on all samples for plotting; the analysis in `scripts/` filters inside each training fold.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn matplotlib jupyter ipykernel pytest

pytest                                  # leakage tests
python scripts/run_analysis.py          # models, predictions, selection frequency (~40 min)
python scripts/permutation_test.py 50   # shuffled-target check (~40 min)
python scripts/plate_check.py           # ELISA plate check (seconds)
python scripts/sensitivity_mismatch.py  # without the mismatch-flagged lines (~5 min)
```

## Layout

- `doxrisk/data.py`: loading, library-size filter, log-CPM, response features, troponin table
- `doxrisk/model.py`: in-fold gene filter, models, leave-one-out evaluation, selection frequency
- `scripts/`: the analyses above, writing to `results/`
- `tests/`: leakage tests
- `explore.ipynb`: walk-through of the data and the mean-only baseline
