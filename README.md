# Overview

This repository contains the code and data for creating the plots and tables in "A Tutorial on Bayesian Analysis of Linear Shock Compression Data," published in the *Journal of Applied Physics*.

Features:

- The BALSCD package (Bayesian Analysis of Linear Shock Compression Data), which implements Bayesian linear regression and bootstrapping methods for analyzing shock wave-particle velocity Hugoniot data

- Datasets for argon, copper, and nickel used in the paper, plus additional datasets for pyrolusite, serpentine, and toluene to demonstrate the package's use on new materials

- A script for applying these methods to new datasets

## Requirements

- Python 3.10 or higher
- Dependencies listed in `pyproject.toml` (automatically installed)

## Datasets

The datasets are from Marsh, S. P. (1980), LASL Shock Hugoniot Data, University of California Press.

## Reproducing Paper Results

**Note:** These instructions are for macOS/Linux.
Windows users should use `python` instead of `python3` and `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

1. **Clone the repository**
```bash
    git clone https://github.com/llnl/BALSCD.git
    cd BALSCD
```

2. **Create a virtual environment**
```bash
    python3 -m venv .venv
```

3. **Activate the virtual environment**
```bash
    source .venv/bin/activate
```

4. **Install package in editable mode with dependencies**
```bash
    pip install -e .
```

Optional: Install development dependencies (pytest and ruff)
```bash
    pip install -e ".[dev]"
```

5. **Create paper plots and summary statistics for tables**
```bash
    python3 scripts/reproduce_paper_results.py
```
Plots are saved to `images/` and tables to `summary_statistics/` (both created automatically).

## Running the Analysis on a New Dataset

The `run_analysis.py` script in the `scripts` directory performs the Bayesian and bootstrap analysis discussed in the paper on a new dataset.
The code requires new datasets to have the same format as the datasets in Marsh (1980).
See the script documentation for details and example usage.

## Citation

If you use this code or data in your research, please cite:

Bernstein, J., Myint, P. C., Lindquist, B. A., & Brown, J. L. (2026). A Tutorial on Bayesian analysis of linear shock compression data. *Journal of Applied Physics*, *140*(3), 031101. https://doi.org/10.1063/5.0334353

BibTeX:
```bibtex
@article{bernstein2026tutorial,
  title={A Tutorial on Bayesian analysis of linear shock compression data},
  author={Bernstein, Jason and Myint, Philip C. and Lindquist, Beth A. and Brown, Justin Lee},
  journal={Journal of Applied Physics},
  volume={140},
  number={3},
  pages={031101},
  year={2026},
  doi={10.1063/5.0334353}
}
```

## Issues and Questions

Please report bugs or ask questions by opening an issue on GitHub.
You can also contact Jason Bernstein at bernstein8@llnl.gov.

## License

This software is distributed under the terms of the MIT license.
All new contributions must be made under the MIT license.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

## Release

LLNL-CODE-2010853
