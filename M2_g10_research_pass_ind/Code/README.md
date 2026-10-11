# Code

This folder has the code behind the M2 report. It is Python 3 with NumPy, SciPy and Matplotlib.

## Setup
```
pip install -r requirements.txt
```
Run every command from this `Code` folder.

## Layout
```
Code/
├── config/params.yaml        all physical and simulation values, in one place
├── src/
│   ├── config.py             loads params.yaml and derives eta, D_S, delta
│   ├── channel.py            free-space gain and LoS blockage probability
│   ├── geometry.py           antenna position and random user drops
│   ├── analytic.py           closed forms: leakage bound, ergodic rate, K*, Gamma(K), TDMA rates
│   └── montecarlo.py         Monte Carlo samplers: leakage ratio, ergodic rate, EDMA system rate
├── tests/                    72 automated tests (pytest)
└── experiments/
    ├── m2_baseline_edma.py       reproduces EDMA Table I
    ├── m2_leakage_ratio.py       far to adjacent leakage ratio for K = 1, 5, 20, 50
    ├── m2_ergodic_rate.py        exact rate against Monte Carlo
    ├── m2_loading_benchmark.py   K*, Gamma(K) and the comparison with orthogonal access
    └── m2_common.py              small shared helpers
```

## Run the tests
```
python -m pytest tests -q
```
All 72 tests pass. They check the closed forms, the samplers and the published baseline numbers.

## Run the experiments
```
python experiments/m2_baseline_edma.py        # about 1 minute (use --quick for a fast run)
python experiments/m2_leakage_ratio.py        # several minutes, uses all CPU cores (use --quick for a fast run)
python experiments/m2_ergodic_rate.py         # about 1 minute (use --quick for a fast run)
python experiments/m2_loading_benchmark.py    # about 1 minute
```
Each script prints its table and writes a CSV and a PNG figure to `../Results`.
Every random run uses a fixed seed, so the numbers repeat.

## Notes
- `config/params.yaml` marks which values come from the EDMA paper and which are my choices.
- In the code, "NOMA" in older docstrings means the overloaded scheme that the report calls EPMA.
- Some docstrings cite "CLAUDE.md" and "GT3". These are section references in the notes file I keep for my AI coding assistant. That notes file is not part of this submission. Every formula the code uses is written out in the M2 report.
- `tdma_rate_eq9` reproduces EDMA Table I. `tdma_system_rate` is the system-wide orthogonal rate that I use for the comparison in the report. It equals `tdma_rate_eq9` multiplied by M times K.
