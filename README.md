
# RSRC-calculator: Recursive Self-Referential Compression Calculator

[![Python](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository contains a Python code snippet for calculating the **Recursive Self-Referential Compression (RSRC)** score for AI models. The RSRC metric aims to provide a way to evaluate the reasoning capability of AI models relative to the computational resources they consume during training.

**[➡️  Explore the full RSRC Calculator code here!](link-to-your-RSRC-calculator-repository)**

## About the RSRC Metric

The Recursive Self-Referential Compression (RSRC) is a metric designed to assess the efficiency of AI models by balancing their recursive reasoning & compression with the resources required to train them. It takes into account:

*   **Model Size (Parameters):**  Larger models are generally considered more capable, but size also impacts resource consumption.
*   **Training Compute (FLOPs):** The amount of computation used to train the model.
*   **Energy Efficiency:** The energy consumed per FLOP during training.
*   **Model Architecture (MoE vs. Dense):**  Different architectures have inherent efficiency trade-offs.
*   **Model Age:** Newer models often benefit from algorithmic and hardware improvements.

The RSRC score is calculated using a simplified formula that incorporates these factors, providing a single metric to compare the resource efficiency of different AI models.

## Code Snippet: `rsrc_calculator.py`

This repository provides a Python code snippet (`rsrc_calculator.py`) that implements the RSRC calculation. You can use this snippet to:

*   Calculate RSRC scores for your own datasets of AI models.
*   Experiment with different parameters and adjustments in the RSRC formula.
*   Contribute to the ongoing development and refinement of the RSRC metric.

### Prerequisites

Before running the code, ensure you have Python 3.6 or later installed, along with the following libraries:

*   [Pandas](https://pandas.pydata.org/): For data manipulation and DataFrame creation. Install using: `pip install pandas`
*   [NumPy](https://numpy.org/): For numerical operations. Install using: `pip install numpy`

### Usage

1.  **Download `rsrc_calculator.py`:** Download the `rsrc_calculator.py` file from this repository.
2.  **Populate `model_data` Dictionary:** Open the `rsrc_calculator.py` file in a text editor.  Locate the `model_data` dictionary within the `if __name__ == '__main__':` block.  **Replace the empty lists** in this dictionary with your AI model data.

    The `model_data` dictionary should be structured as follows:

    ```python
    model_data = {
        "Model": [],  # List of model names (strings)
        "Architecture": [],  # List of architectures ("MoE" or "Dense" strings)
        "Parameters (B)": [],  # List of parameter counts in billions (floats)
        "Layers": [],  # List of layer counts (integers or None if unknown)
        "Training FLOPs": [],  # List of training FLOPs (floats) - in PetaFLOPs
        "MMLU (5-shot)": [],  # List of MMLU 5-shot scores (floats) - in percentage
        "Energy/pFLOP": [],  # List of Energy/pFLOP values (floats) - in pJ
        "Year": []   # List of training/release years (integers)
    }
    ```

    **Example Data Entry:**

    ```python
    model_data = {
        "Model": ["GPT-4", "Claude 3 Opus"],
        "Architecture": ["MoE", "Dense"],
        "Parameters (B)": [1800, 130],
        "Layers": [120, 96],
        "Training FLOPs": [21500, 45000],
        "MMLU (5-shot)": [86.4, 86.8],
        "Energy/pFLOP": [62.1, 90.0],
        "Year": [2023, 2024]
    }
    ```

3.  **Run the Script:** Open your terminal or command prompt, navigate to the directory where you saved `rsrc_calculator.py`, and execute the script using:

    ```bash
    python rsrc_calculator.py
    ```

4.  **View Output:**

    *   **Console Output:** The script will print a Pandas DataFrame to your console, displaying the input model data along with the calculated "New RSRC" scores.
    *   **CSV Output (Optional):** To save the results to a CSV file, uncomment the lines in the `if __name__ == '__main__':` block that are related to `rsrc_leaderboard_df.to_csv(...)`. This will save the leaderboard data to a file named `rsrc_leaderboard.csv` in the same directory.

### Understanding the Formula and Parameters

The `calculate_rsrc` function implements the following simplified RSRC formula:

```
RSRC = ln(Active Parameters) / (FLOPs Efficiency Score ^ gamma) * Age Decay
```

Where:

*   **FLOPs Efficiency Score:**  Calculated as `(Training FLOPs / Parameters^alpha) * (Energy/pFLOP / beta)`. This score penalizes models that are inefficient in their use of compute and energy relative to their size.
*   **Active Parameters:** Represents the effective parameter count, especially important for MoE models where only a fraction of parameters are active during inference. For MoE models, `Active Parameters` are scaled down by `moe_param_scale_factor`.
*   **gamma:**  Exponent applied to the `FLOPs Efficiency Score`. Different `gamma` values are used for MoE (`moe_exponent`) and Dense (`dense_exponent`) architectures to fine-tune the FLOPs penalty.
*   **Age Decay:** A factor that reduces the RSRC score for older models, acknowledging the advancements in model efficiency over time.

**Key Parameters for Customization (defined at the beginning of the script):**

*   `alpha`: Controls the influence of parameter count in the FLOPs Efficiency Score.
*   `beta`: Normalization factor for energy per FLOP.
*   `moe_param_scale_factor`: Scales down parameters for MoE models.
*   `legacy_penalty_factor`: Penalizes older dense models.
*   `moe_exponent`, `dense_exponent`: Gamma exponents for MoE and Dense models respectively.
*   `current_year`:  Used for age decay calculation; update this to the current year or a relevant benchmark year.

Feel free to experiment with these parameters to adjust the RSRC calculation to better fit your specific evaluation needs or to explore different weighting schemes for resource efficiency.

## Contributing

Contributions to improve the code, refine the RSRC formula, or expand the model dataset are welcome! Please feel free to fork this repository and submit pull requests with your enhancements.

## License

This code is released under the [MIT License](LICENSE).
