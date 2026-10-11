# Data

This project does not use an external dataset. All data come from simulation and from closed-form calculation in Python.

- The physical values (carrier frequency, region size, waveguide height, blockage parameter, SNR) are in `Code/config/params.yaml`. Most of them come from the EDMA paper.
- The numbers I reproduce from the EDMA paper (Table I) are typed into `Code/experiments/m2_baseline_edma.py` as the published reference values.
- The CSV files in `Results/` are generated data. Running the scripts in `Code/experiments` creates them again.
- Every Monte Carlo run uses a fixed seed. The number of trials is printed by each script and listed in `Results/README.md`.
