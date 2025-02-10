# RSRC Calculator
[![DOI](https://zenodo.org/badge/930413682.svg)](https://doi.org/10.5281/zenodo.14846489)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)
[![pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)


[**Read the Paper: Recursive Self-Referential Compression (RSRC): AI's Survival Map in the Post-Scaling Era**](https://doi.org/10.5281/zenodo.14846507)

This repository provides a practical tool to calculate the **Recursive Self-Referential Compression (RSRC)** metric, as introduced in the paper [*Recursive Self-Referential Compression (RSRC): AI's Survival Map in the Post-Scaling Era*](https://doi.org/10.5281/zenodo.14846507). RSRC is a dual-metric framework designed to evaluate the efficiency of AI models by distinguishing between **training efficiency (RSRC<sub>t</sub>)** and **inference efficiency (RSRC<sub>i</sub>)**.

As compute-centric AI scaling faces thermodynamic and economic limits, RSRC offers a crucial lens for understanding and optimizing AI model development for a sustainable future. This calculator focuses on the **training efficiency RSRC<sub>t</sub> metric**, providing a score that reflects resource utilization during the training phase.

---

## Repository Structure

```plaintext
RSRC-Calculator/
├── README.md               # This README file
├── LICENSE                 # GNU GPLv3 License
├── rsrc_calculator.py      # Python script for RSRC calculation
├── data/                  
│   └── model_leaderboard.csv  # CSV dataset of AI models and metrics
└── code_snippets/          # Directory for code examples from the paper
    ├── moe_sparsity/           # Code snippets for MoE implementation (Strategy 1)
    ├── energy_monitoring/      # Code snippets for energy monitoring (Strategy 2)
    └── recursive_regularization/ # Code snippets for recursive regularization (Strategy 3)
```

---

## Dataset: Model Leaderboard (`data/model_leaderboard.csv`)

The `data/model_leaderboard.csv` file contains a curated leaderboard of various AI models along with key metrics used for RSRC calculation. The metrics include:

- **Model Name**: Name of the AI model.
- **Architecture**: Model architecture type (Dense, MoE, etc.).
- **Parameters (B)**: Total number of parameters (in billions). For MoE models, this often represents the total parameters, with the active parameters being a subset.
- **Layers**: Number of layers in the model.
- **Training FLOPs**: Estimated total floating-point operations for training (in PetaFLOPs).
- **MMLU (5-shot)**: Performance on the MMLU benchmark (5-shot accuracy).
- **Energy/pFLOP**: Estimated energy cost per PetaFLOP of computation (in picojoules).
- **Year**: Year of model release or training completion.

### Important Note on Data Approximation

Many data points in `model_leaderboard.csv` are **approximations and extrapolations**—especially for cutting-edge or proprietary models where official data is limited. The estimates are derived from:

- **Analyst Estimates and Reports**: Expert opinions, technical analyses, and blog posts.
- **Inference from Publicly Available Data**: Estimations from reported training costs, GPU benchmarks, etc.
- **Model Cards and Technical Documentation**: Information released by model developers, which may sometimes be incomplete.
- **Scaling Law Extrapolations**: Applying known scaling trends to estimate metrics when direct data is absent.

While we strive for the best estimates, the data should be considered indicative rather than perfectly accurate. The RSRC calculator is intended as a **comparative tool** using the best available information. Community contributions to refine and improve the dataset are highly encouraged.

---

## RSRC Calculator Usage (`rsrc_calculator.py`)

The `rsrc_calculator.py` script calculates the RSRC training efficiency score (RSRC<sub>t</sub>) for AI models.

### Requirements

- Python 3.x
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)

### How to Run

1. **Install Dependencies:**

    ```bash
    pip install pandas numpy
    ```

2. **Run the Script:**

    ```bash
    python rsrc_calculator.py
    ```

The script will:

- Load model data from `data/model_leaderboard.csv`
- Calculate the RSRC score for each model based on the paper's formula and provided parameters
- Output the updated leaderboard (including RSRC scores) to the console
- Optionally, output the leaderboard as a new CSV file (currently, the results are printed to the console)

### Customization and Extension

- **Update the Dataset:** Modify `model_leaderboard.csv` with new models or refined metrics.
- **Adjust Parameters:** Experiment with parameters (such as `alpha`, `beta`, `gamma`, `age decay`, etc.) in `rsrc_calculator.py` to explore different weighting schemes.
- **Expand Code Examples:** Contribute additional code snippets to the `code_snippets/` directory that illustrate further RSRC-related concepts.

---

## Code Snippets (`code_snippets/`)

This directory contains illustrative code snippets related to RSRC optimizations:

- **MoE Sparsity (`moe_sparsity/`):**  
  *Demonstrates the integration of sparse Mixture-of-Experts layers to reduce active parameter count and promote recursive processing.*

- **Energy Monitoring (`energy_monitoring/`):**  
  *Provides examples for integrating energy profiling tools (such as NVIDIA SMI) to measure energy consumption during training.*

- **Recursive Regularization (`recursive_regularization/`):**  
  *Illustrates approaches for incorporating entropy-based penalties to foster compressed internal representations.*

[**We encourage contributions of further code snippets and examples to expand this resource.**]

---

## Contributing

We welcome contributions to improve the RSRC calculator, refine the dataset, and expand the code examples. Please feel free to fork the repository and submit pull requests.

---

## License

This project is licensed under the **GNU General Public License v3**. See the [LICENSE](./LICENSE) file for complete details.

---

## Disclaimer

The RSRC metric and this calculator are experimental tools for evaluating AI model efficiency based on approximated and extrapolated data. The results should be interpreted within this context. This repository is not an official benchmark, and neither the authors of the RSRC paper nor the repository contributors are responsible for decisions based on these calculations.

---

[**MURST Research Initiative**](https://murst.org/)
