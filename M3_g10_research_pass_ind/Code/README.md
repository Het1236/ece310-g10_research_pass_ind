# Code for Milestone 3

The code is Python. Every simulation reads its values from `config/params.yaml`.

## Layout
- `src/` the model and the solvers.
  - `channel.py`, `geometry.py` gains and blockage.
  - `analytic.py` leakage bound, exact ergodic rate, orthogonal access benchmark.
  - `montecarlo.py` Monte Carlo checks.
  - `placement.py` golden-section and K-start search, grid search used as a check.
  - `power_alloc.py` power allocation by linear program, max-min by bisection, SIC order.
  - `floor_position.py` interference floor that moves with the antenna, and its slope bound.
  - `maxmin_closed.py` closed-form max-min powers and the Lipschitz constant.
  - `maxmin_global.py` certified global search (Piyavskii-Shubert) for max-min placement.
  - `altopt.py` naive alternating optimisation, used only as a baseline.
- `tests/` unit tests. Run with `python -m pytest tests -q`.
- `config/params.yaml` all parameters.
- `experiments/` one script per result.

## Experiment scripts
| Script | Result |
|---|---|
| `m3_placement_rules.py` | Placement rules, search cost, MAC tie |
| `m3_search_check.py` | Peak counts, plain against certified search |
| `m3_fairness.py` | Max-min fairness under off-axis disparity |
| `m3_blockage_robustness.py` | Own-link blockage, real interference, NLoS residual |
| `m3_segments.py` | Number of segments, EPMA against orthogonal access |

`m3_common.py` holds the plot style and the output folder.

## Run
```
pip install -r requirements.txt
python -m pytest tests -q
python experiments/m3_placement_rules.py
```
Use `--quick` for a fast run. The seed is 42, so full runs repeat exactly.
