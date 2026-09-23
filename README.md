# dox-cardiotox-risk

Can the early transcriptomic response of a person's heart muscle cells to doxorubicin predict how much damage those cells take at higher doses, and does it add information beyond troponin, the biomarker already used in the clinic?

## Data

iPSC-derived cardiomyocytes from about 45 individuals, each exposed to five doxorubicin concentrations (0 to 5 µM) for 24 hours, with RNA-seq gene counts and cardiac troponin I measurements. Source: Knowles et al., eLife 2018, doi:10.7554/eLife.33480, and the repository github.com/davidaknowles/dox. The data are not included here; download them with:

```bash
mkdir data
curl -L -o data/gene-counts.txt.gz https://raw.githubusercontent.com/davidaknowles/dox/master/data/gene-counts-round-two.txt.gz
curl -L -o data/troponin.txt https://raw.githubusercontent.com/davidaknowles/dox/master/data/troponin-processed.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib jupyter ipykernel
```

## Status

Work in progress.