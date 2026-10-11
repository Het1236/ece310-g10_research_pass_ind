# Results

All files here are produced by the scripts in `Code/experiments`. Run a script again to recreate its files.

| File | What it shows | Script | Trials and seed |
|---|---|---|---|
| `baseline_edma_table1.csv` and `.png` | EDMA Table I against my reproduction (K = 1, M = 3, 5, 7, 9, 11) | `m2_baseline_edma.py` | 150000 trials per M, seed 2 |
| `leakage_ratio.csv` and `leakage_ratio_vs_phiDs2.png` | Far to adjacent leakage ratio for K = 1, 5, 20, 50 against the bound | `m2_leakage_ratio.py` | 50000 trials below x = 2.5 and 400000 trials from x = 2.5, seed 42 |
| `ergodic_rate_vs_K.csv` and `.png` | Exact rate integral against Monte Carlo, and the rate against K | `m2_ergodic_rate.py` | 50000 trials per K, seed 42 plus K |
| `kstar_vs_snr.csv` | Crossover load K* for 80, 100, 110 and 120 dB | `m2_loading_benchmark.py` | no randomness |
| `loading_vs_orthogonal.csv` and `.png` | System sum rate of the proposed scheme against two orthogonal benchmarks, and the gain of loading, for K = 1 to 64 | `m2_loading_benchmark.py` | no randomness |

Notes:
- All rates are in bps per Hz.
- The geometry is M = 3, D_L = 40 m, 28 GHz. The SNR is 120 dB unless the file says otherwise.
- "EPMA" in the report is the overloaded scheme. In the CSV files its columns are called `epma_*`.
