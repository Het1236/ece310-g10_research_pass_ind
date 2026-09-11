# Code

At this stage our code covers the early system and channel model only. This is the starting point the rest of the project will build on.

## What is here
The system model and channel functions live in the main repository, at the repository root:
- `src/channel.py` — channel gain and line-of-sight blockage probability functions.
- `src/geometry.py` — segment and user geometry helper functions.
- `config/params.yaml` — physical and simulation parameters, so results can be reproduced.

## How to run it
1. Install Python 3.10 or later.
2. From the repository root, install the requirements:
   `pip install -r requirements.txt`
3. Run the early model tests to check the functions work correctly:
   `pytest tests/test_analytic.py`

## What is not here yet
The feasibility analysis, rate derivation, and optimization code are still being developed. They will be added in later milestones as the project moves forward.
