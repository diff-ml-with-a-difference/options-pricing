# Differential ML with a Difference

This repository contains the replication code and experiment notebook for the paper:

**Differential ML with a Difference**  
by *Paul Glasserman and Siddharth Hemant Karmarkar*  
Columbia University — November 2025

The paper investigates alternative differential labeling via the Likelihood Ratio Method (LRM) and hybrid derivatives to improve Differential ML for discontinuous payoffs (e.g., digital and barrier options). This repository contains the exact experiment setup used to generate all key figures and results.

---

## 🚀 How to Run Experiments

### 🔹 Option 1 — Using Google Colab *(recommended for fast execution & GPU support)*

Click below to directly run the experiment notebook:

🔗 **[Open Full Experiment Notebook in Colab](https://colab.research.google.com/drive/1pfGNqhGBxKNRAFbRPowd53jZqsNcxarW#scrollTo=575-0q1L3nrG)**

or use the GitHub-hosted version:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/option-ml-research/differential-ml-pricing/blob/main/notebooks/DifferentialML_Experiments.ipynb)

The notebook walks through:
- Digital option experiments (Figure 1)
- Barrier option experiments (Figures 3–4)
- Gamma regularization via PW–LR method (Figure 5)
- Reproducibility using Monte Carlo simulated samples

---

### 🔹 Option 2 — Run via Python Script (local or cloud)

1. Clone the repository:
```bash
git clone https://github.com/options-pricing/src/differential_ml_with_a_difference.py
cd differential-ml-pricing

	2.	Install dependencies:

pip install -r requirements.txt

	3.	Run the core experiment module:

python src/collective_experiments.py

💡 This method is preferred if you want to integrate results inside a training pipeline or modify network architecture.

⸻

📁 Repository Structure

differential-ml-pricing/
│── notebooks/
│   └── DifferentialML_Experiments.ipynb
│── src/
│   └── collective_experiments.py
│── paper/
│   └── OptionsPricingResearch.pdf
│── requirements.txt
│── README.md
│── LICENSE


⸻

Versioning for Reproducibility

The exact version used for the paper is tagged as:

v1.0.0-preprint

The main branch may contain improvements or extended experiments not included in the paper.


⸻

🧪 Experimental Setup
	•	Python 3.x (Google Colab default environment)
	•	Monte Carlo simulation via collective_experiments.py
	•	Neural network with 4 hidden layers, 20 units, softplus activation (see paper)
	•	Optimizer: Adam with cosine decay (10⁻³ → 10⁻⁶)
	•	Training batch size: 256–512
	•	Simulation replication factor: k = 10 (averaged)
	•	Network accuracies matched against closed-form benchmarks

⸻

License

Released under the MIT License for transparency and replicability.

⸻

🔁 Notes
	•	All figures from the paper can be replicated using the notebook.
	•	Use Colab if GPU acceleration is preferred.
	•	Local .py execution is supported for integration into quant research pipelines.

⸻

Thanks for exploring this research repository!

---
