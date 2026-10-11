# Milestone 2: Two-Layer Uplink Multiple Access in Segmented Pinching-Antenna Systems

This folder is my Milestone 2 (M2) submission for ECE 310, Wireless Communications, Monsoon Semester 2026.

- Group: g10_research_pass_ind
- Student: Het Patel (AU2440266)
- Category: Research Project

## What M2 covers
M1 defined the problem and picked EDMA as the baseline. M2 moves to technical work.

- A literature synthesis and a clear gap statement.
- The problem formulation: system model, assumptions, metrics and research questions.
- The proposed two-layer scheme (I call it EPMA) and the method to study it.
- A reproduction of Table I of the EDMA paper with my own code.
- Three first experiments: far-segment leakage, the exact rate of one loaded segment, and a comparison with orthogonal access.
- The SOTA position of the work.

## Main findings so far
- My code reproduces EDMA Table I within 0.42 percent.
- The far to adjacent leakage ratio does not depend on the number of users per segment, and it stays below the bound.
- The exact rate integral agrees with Monte Carlo simulation for K = 1, 2, 5, 10, 20 and 50.
- At millimetre-wave SNR, adding users to a segment does not raise the sum rate. The gain is more users served on the same hardware.
- The sum rate lead over system-wide time division at low load is mostly a blockage windfall. It disappears near K = 16 at 120 dB.

## Folder contents
- `Report/` the M2 report (PDF).
- `Video/` the link to the 20 minute walkthrough and the slides used in it.
- `Code/` source code, configuration, tests and the experiment scripts. See `Code/README.md`.
- `Data/` notes on the data. This project uses simulated data only. See `Data/README.md`.
- `Results/` the tables and figures produced by the code. See `Results/README.md`.

## How to reproduce
```
cd Code
pip install -r requirements.txt
python -m pytest tests -q
python experiments/m2_baseline_edma.py
python experiments/m2_leakage_ratio.py
python experiments/m2_ergodic_rate.py
python experiments/m2_loading_benchmark.py
```
Each script writes its tables and figures to `Results/`.

## Limits
- SIC is assumed ideal.
- Links inside a region are treated as line of sight.
- The numbers are for one geometry (M = 3 at 28 GHz).
- Fairness, antenna placement and power allocation are planned for M3.

## Note on the individual Google Form
Each team member submits the individual Google Form separately. It is not part of this folder.
