# Milestone 3: Two-Layer Uplink Multiple Access in Segmented Pinching-Antenna Systems

This folder is my Milestone 3 (M3) submission for ECE 310, Wireless Communications, Monsoon Semester 2026.

- Group: g10_research_pass_ind
- Student: Het Patel (AU2440266)
- Category: Research Project

## What M3 covers
M1 stated the problem and the literature. M2 built the system model and checked the baseline. M3 gives the final design method and tests it.

- The final method: antenna placement and power allocation inside a segment, with max-min fairness.
- The experimental setup: parameters, assumptions, metrics and baselines.
- Results on placement, search methods, fairness, blocked links and the number of segments.
- A comparison with baselines, including system-wide orthogonal access.
- A critical analysis of what worked, what failed and what the limits are.
- The revised state-of-the-art position.

## Main findings
- Placing the antenna above the strongest user loses 14.6 percent of the segment rate at K = 1 and 28.3 percent at K = 16. Keeping it at the segment centre loses 2.7 percent at K = 1 and 0.15 percent at K = 16.
- The sum-rate position problem has one peak once the interference floor moves with the antenna. Plain golden-section search is enough there.
- The max-min problem has several peaks in 21.5 percent (K = 2) and 42 percent (K = 4) of random drops. Plain golden-section search loses up to 7.4 percent. The certified global search stays within its 1e-3 bps/Hz tolerance.
- At equal total power NOMA and boosted time sharing have the same sum rate. I do not claim a sum-rate gain from NOMA at equal power.
- EPMA gives 2.31 times the system rate of orthogonal access (average power) at K = 1. The lead is gone at K = 16. It is mostly a blockage windfall and it is not robust.
- Max-min power control keeps the floor flat when one user moves off axis. Orthogonal access does not.
- With real random interference the minimum rate misses the design target in 23 percent (K = 2) and 38 percent (K = 4) of draws.

## Folder contents
- `Report/` the M3 report.
- `Code/` source code, configuration, tests and experiment scripts. See `Code/README.md`.
- `Results/` the tables, figures and run logs produced by the code. See `Results/README.md`.

This project uses simulated data only. There is no separate data folder.

## How to reproduce
```
cd Code
pip install -r requirements.txt
python -m pytest tests -q
python experiments/m3_placement_rules.py
python experiments/m3_search_check.py
python experiments/m3_fairness.py
python experiments/m3_blockage_robustness.py
python experiments/m3_segments.py
```
Each script writes its tables and figures to `Results/`. Add `--quick` to any script for a fast test run with smaller samples. The full runs take about 20 minutes in total.

## Limits
- SIC is ideal and the decoding order is fixed.
- Own links are line of sight in the design. If they were blocked too, the rate would be lower by 14 to 39 percent.
- The certified search is exact for the mean interference floor. It is not a guarantee for each random channel draw.
- Only the uplink is studied.
- All results are simulations of my own model. There is no hardware test.

## Use of AI tools
I used an AI assistant (Claude) to help write and check code and to help edit the report. I ran the experiments and reviewed the results myself.

## Note on the individual Google Form
Each team member submits the individual Google Form separately. It is not part of this folder.
