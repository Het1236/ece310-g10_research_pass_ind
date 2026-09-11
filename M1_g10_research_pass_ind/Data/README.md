# Data

This project does not use an external dataset. All results come from numerical simulation and Monte Carlo experiments written in Python, using NumPy and SciPy.

The physical parameters we simulate with, such as waveguide length, carrier frequency, blockage probability, and the signal-to-noise ratio range, are taken from published pinching-antenna research and stored in `config/params.yaml` at the repository root. This keeps every run reproducible. Anyone can re-run our code with the same parameters and get the same results.
