# Data Approximation Explanation

Many of the data points in the `model_leaderboard.csv` are approximations and extrapolations. The following sections provide a detailed explanation of how these approximations were made.

---

## 1. Training FLOPs

Training FLOPs is often the most opaque metric. For many models—especially those from closed-source labs—the exact training FLOPs are not publicly disclosed. Approximations are made based on several factors:

### a. Reported Training Costs
- **Approach:** If a training budget is announced (e.g., "$100 million"), analysts may reverse-engineer an estimated FLOP count.
- **Assumptions:** Estimates are based on known cloud GPU prices and utilization assumptions. This method is inherently rough.

### b. Model Size and Architecture
- **Approach:** Use scaling laws and trends (such as those described in Kaplan et al.'s work) to provide a rough estimate based on the model's parameter count and type (e.g., dense vs. Mixture-of-Experts (MoE)).
- **Assumptions:** MoE models are often assumed to be more FLOP-efficient during training.

### c. Comparisons to Similar Models
- **Approach:** When FLOPs are known for a similar model, estimates may be scaled based on parameter count or perceived complexity.

### d. Token Count and Inferred FLOPs per Token
- **Approach:** If the training dataset size (in tokens) is known, combining this information with assumptions about FLOPs per token can yield an overall FLOP estimate.
- **Assumptions:** This method is highly dependent on the assumptions made about token processing.

### e. "Gen" Estimates
- **Approach:** For models like GPT-4, analysts sometimes speculate on "Gen 2/3" model FLOP ranges based on observed capabilities and scaling trends.
- **Example Range:** For instance, "Gen 2 models: 1e25–1e26 FLOPs" represents a very broad range.

---

## 2. Energy per pFLOP (pJ/FLOP)

Energy consumption approximations are derived using several methods:

### a. GPU Benchmarks
- **Approach:** Utilize energy efficiency benchmarks (e.g., FLOPS/Watt) of GPUs such as NVIDIA H100 or A100.
- **Conversion:** These benchmarks help estimate pJ/FLOP, though actual training workloads can vary.

### b. Power Consumption Reports
- **Approach:** Use power consumption data from data centers or specific training runs.
- **Conversion:** When combined with FLOP estimates, these data can be converted into pJ/FLOP estimates.

### c. Comparisons to Similar Architectures
- **Assumption:** MoE models are generally assumed to be more energy-efficient per FLOP for inference—and sometimes for training, although this is less clear-cut.

### d. "Inferred" Values
- **Note:** Terms like "inferred" or "estimated" indicate that these values are reasoned guesses based on limited data rather than direct measurements.

---

## 3. MMLU (5-shot)

While MMLU scores are more directly measurable, they can still involve approximations:

- **Rounding:** A tilde (`~`) indicates that the score is rounded or derived from a less precise source.
- **Source Variability:** Scores might come from internal benchmarks as opposed to publicly reported ones.
- **Benchmark Variations:** Slight differences in benchmarking setups can lead to small variations in scores.

---

## 4. Parameters (B) and Layers

For many publicly released models, parameter counts and layer numbers are known with greater precision:

- **General Precision:** These values are usually more accurately reported.
- **MoE Models Specifics:**
  - **Active Parameters:** The "active parameters" count is often an average or estimate per token, as it can vary dynamically.
  - **Approximate Counts:** A tilde (`~`) may be used to denote approximate parameter counts, especially when exact details are not fully disclosed.

---

## 5. Summary

- **Synthesis of Data:** The information in the CSV is a synthesis of the best available estimates, involving both extrapolation and inference.
- **Accuracy:** While not a perfectly accurate reflection of ground truth, these approximations provide a reasonable basis for comparative RSRC analysis in a rapidly evolving and often opaque field.
- **Evolution and Refinement:** The RSRC framework is designed to be refined and improved as better data becomes available. Community contributions are essential to this process.

---

*Note: The approximations described here are subject to revision as new data and methodologies emerge.*
