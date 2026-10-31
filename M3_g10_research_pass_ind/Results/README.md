# Results for Milestone 3

All files are written by the scripts in `Code/experiments/`. Rates are in bps/Hz.

| File | Script | Content |
|---|---|---|
| `placement_rules.csv`, `placement_rules.png` | `m3_placement_rules.py` | Sum rate for the best position, above the strongest user and at the centre |
| `search_cost.csv` | `m3_placement_rules.py` | Search iterations against tolerance |
| `mac_tie.csv` | `m3_placement_rules.py` | Largest gap between NOMA and boosted time sharing |
| `search_check.csv`, `search_check.png` | `m3_search_check.py` | Peak counts and misses of plain and certified search |
| `fairness.csv`, `fairness.png` | `m3_fairness.py` | Max-min floor against the off-axis offset of one user |
| `own_link_blockage.csv` | `m3_blockage_robustness.py` | Rate lost when own links are blocked |
| `realised_interference.csv` | `m3_blockage_robustness.py` | Share of draws below the design target |
| `nlos_residual.csv`, `blockage_robustness.png` | `m3_blockage_robustness.py` | System rate when blocked links leak |
| `segments.csv`, `epma_vs_orthogonal.csv`, `segments.png` | `m3_segments.py` | Number of segments and comparison with orthogonal access |
| `logs/` | all | Console output of the full runs |

The numbers in the report come from these files.
