# RSRC Calculator: Evaluating AI Model Efficiency in the Post-Scaling Era

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[**Read the Paper: Recursive Self-Referential Compression (RSRC): AI's Survival Map in the Post-Scaling Era**](LINK_TO_YOUR_PAPER_HERE)

This repository provides a practical tool to calculate the **Recursive Self-Referential Compression (RSRC)** metric, as introduced in the paper "[Recursive Self-Referential Compression (RSRC): AI's Survival Map in the Post-Scaling Era](LINK_TO_YOUR_PAPER_HERE)".  RSRC is a dual-metric framework designed to evaluate the efficiency of AI models, distinguishing between **training efficiency (RSRC<sub>t</sub>)** and **inference efficiency (RSRC<sub>i</sub>)**.

As compute-centric AI scaling faces thermodynamic and economic limits, RSRC offers a crucial lens for understanding and optimizing AI model development for a sustainable future. This calculator focuses on the **training efficiency RSRC<sub>t</sub> metric**, providing a score that reflects resource utilization during the training phase.

## Repository Structure

```
RSRC-Calculator/
├── README.md           # This README file
├── LICENSE             # GNU GPLv3 License
├── rsrc_calculator.py  # Python script for RSRC calculation
├── data/               # Directory for datasets
│   └── model_leaderboard.csv  # CSV dataset of AI models and metrics
└── code_snippets/      # Directory for code examples from the paper
    ├── moe_sparsity/           # Code snippets for MoE implementation (Strategy 1)
    ├── energy_monitoring/      # Code snippets for energy monitoring (Strategy 2)
    └── recursive_regularization/ # Code snippets for recursive regularization (Strategy 3)
```

## Dataset: Model Leaderboard (`data/model_leaderboard.csv`)

The `data/model_leaderboard.csv` file contains a curated leaderboard of various AI models along with key metrics used for RSRC calculation. These metrics include:

*   **Model Name**: Name of the AI model.
*   **Architecture**:  Model architecture type (Dense, MoE, etc.).
*   **Parameters (B)**: Total number of parameters in billions. For MoE models, this often represents the total parameters, with active parameters being a subset.
*   **Layers**: Number of layers in the model.
*   **Training FLOPs**: Estimated total floating-point operations for training (in PetaFLOPs).
*   **MMLU (5-shot)**:  Performance on the MMLU benchmark (5-shot accuracy).
*   **Energy/pFLOP**: Estimated energy cost per PetaFLOP of computation (in picojoules).
*   **Year**: Year of model release or training completion.
*   **RSRC**: Calculated RSRC training efficiency score.

**Important Note on Data Approximation:**

It is crucial to understand that many data points within `model_leaderboard.csv` are **approximations and extrapolations**, especially for cutting-edge and proprietary models where official data is often limited. These approximations are derived from a variety of sources, including:

*   **Analyst Estimates and Reports**: Industry expert opinions, technical analyses, and blog posts.
*   **Inference from Publicly Available Data**: Estimating FLOPs from reported training costs, energy consumption from GPU benchmarks, and so on.
*   **Model Cards and Technical Documentation**: Utilizing information released by model developers, which may sometimes be incomplete or lack specific details on training metrics.
*   **Scaling Law Extrapolations**: Applying known scaling trends to estimate metrics where direct data is absent.

**As such, while we strive for the best possible estimates, the data should be considered as indicative and not perfectly accurate.** The RSRC calculator is intended as a **comparative tool** using the best available information to evaluate relative efficiency.  We encourage community contributions to refine and improve the accuracy of the dataset.

## RSRC Calculator Usage (`rsrc_calculator.py`)

The `rsrc_calculator.py` script is a Python tool to calculate the RSRC training efficiency score (RSRC<sub>t</sub>) for AI models.

**Requirements:**

*   Python 3.x
*   pandas
*   numpy

**Running the Calculator:**

1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy
    ```

2.  **Run the script:**
    ```bash
    python rsrc_calculator.py
    ```

The script will:

*   Load model data from `data/model_leaderboard.csv`.
*   Calculate the RSRC score for each model based on the formula presented in the paper and the provided parameters.
*   Output the updated leaderboard to the console, including the calculated RSRC scores.
*   Optionally, it can output the leaderboard as a new CSV file (currently configured to print to console).

**Customizing and Extending:**

*   **Modify `model_leaderboard.csv`**:  Update the dataset with new models or refined metrics.
*   **Adjust Parameters in `rsrc_calculator.py`**:  Experiment with `alpha`, `beta`, `gamma`, `age decay`, and other parameters within the script to explore different weighting schemes and sensitivities of the RSRC metric.
*   **Contribute Code Snippets**:  Add more code examples to the `code_snippets/` directory illustrating RSRC-related concepts.

## Code Snippets (`code_snippets/`)

This directory contains illustrative code snippets related to the RSRC framework and optimization strategies discussed in the paper.  Currently, it includes examples for:

*   **MoE Sparsity (`moe_sparsity/`)**: [*(Add a brief description of what the MoE snippet demonstrates)*]
*   **Energy Monitoring (`energy_monitoring/`)**: [*(Add a brief description of what the energy monitoring snippet demonstrates)*]
*   **Recursive Regularization (`recursive_regularization/`)**: [*(Add a brief description of what the recursive regularization snippet demonstrates)*]

[**We encourage contributions of further code snippets and examples to expand this resource.**]

## Contributing

We welcome contributions to improve the RSRC calculator, refine the dataset, and expand the code examples.  

## License

This project is licensed under the terms of the **GNU General Public License v3**. See the `LICENSE` file for complete license details.

## Disclaimer

The RSRC metric and this calculator are tools for evaluating AI model efficiency within a developing research framework.  The results should be interpreted in the context of the approximated and extrapolated data used.  This is not an official benchmark, and the authors of the RSRC paper and this repository are not responsible for decisions made based on these calculations.

---

[**MURST Research Initiative**](https://murst.org/)
