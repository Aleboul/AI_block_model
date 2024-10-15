# Supplementary code for "High dimensional variable clustering based on maxima of a weakly dependent random process"

This repository contains implementation of the ECO algorithm described in High "dimensional variable clustering based on maxima of a weakly dependent random process", Section 4 and codes for generating all figures and table therein. 

## File descriptions:

* `model_1.py`: generates results for E2 and Framework F2.
* `model_1.R`: contains code to run muscle algorithm and sKmeans called by model_1.py.
* `model_2.py`: generates results for E2 and Framework F1.
* `model_2.R`: contains code to run muscle algorithm and sKmeans called by model_2.py.
* `model_3.py`: generates results for E1 and Framework F2.
* `model_3.R`: contains code to run muscle algorithm and sKmeans called by model_3.py.
* `model_4.py`: generates results for E1 and Framework F1.
* `model_4.R`: contains code to run muscle algorithm and sKmeans called by model_4.py.
* `clef.py`: contains functions to run clef algorithm, kindly taken from Chiapino, M., Sabourin, A., and Segers, J. (2019). Identifying groups of variables with the
potential of being large simultaneously. Extremes, 22:193–222.
* `damex.py`: contains functions to run damex algorithm, taken from Chiapino, M., Sabourin, A., and Segers, J. (2019). Identifying groups of variables with the
potential of being large simultaneously. Extremes, 22:193–222.
* `eco_alg.py`: contains functions to run ECO algorithm.
* `ut_eco.py` and `utilities.py`: contain additional function to run eco algorithm clef and damex algorithms, respectively.
* `muscle.R`: functions related to the article "MUltivariate Sparse CLustering for Extremes" by Nicolas Meyer and Olivier Wintenberger.

## Dependencies:
The following packages are required to run the code:

- Python packages:
  - **numpy** (version >= 1.24.2)
  - **Clayton** (version >= 0.0.3)
  - **pandas** (version >= 1.5.3)
  - **matplotlib** (version >= 3.7.1)
  - **sklearn** (version >= 1.2.2)
  - **rpy** (version > 3.5.13)

- R packages:
  - **rlang** (version >= 1.1.1)
  - **MASS** (version > 7.3.55)
  - **skmeans** (version >= 0.2.16)