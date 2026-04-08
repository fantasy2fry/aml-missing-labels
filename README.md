# AML Project 1: Logistic Regression with Missing Labels

## Authors
* [**Norbert Frydrysiak**](https://github.com/fantasy2fry)
* [**Michał Kukla**](https://github.com/mickuk)
* [**Piotr Bartosiewicz**](https://github.com/PiotrDS)

## 📌 Project Overview
This repository contains the solution for Project 1 of the Advanced Machine Learning course. The aim of this project is to analyze the performance of a logistic regression model in binary classification scenarios where the training dataset contains observations with missing labels ($Y=-1$). 

The project includes:
1. **Missing Data Generation:** Implementation of four missing data mechanisms: MCAR, MAR1, MAR2, and MNAR.
2. **Logistic Lasso Regression (FISTA):** A custom implementation of Logistic Regression optimized using the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) with an L1 penalty.
3. **Semi-supervised variants:** Two approaches that use unlabeled points (`novel` and `saute`), compared against a **labeled-only** baseline and an **oracle** reference.
4. **Benchmarking:** Repeated experiments across datasets, missingness schemes, rates, and seeds, with automatic metric aggregation.

## 🗂 Repository Structure
The project is structured into two main directories: `code/` and `report/`, adhering to the submission guidelines.

* `code/data/`: Contains the 4 real-world datasets used for experiments (raw and processed).
* `code/src/`: Core Python modules.
  * `data_prep.py`: Dataset loading/preprocessing and missing-label generation schemes.
  * `fista.py`: FISTA-based logistic regression implementation.
  * `novel_logreg.py`: Iterative method for learning with missing labels.
  * `pl_logreg.py`: SAUTE-based pseudo-labeling logistic regression utilities.
  * `evaluation.py`: Metric computation (Recall, Precision, F1, ROC AUC, PR AUC, etc.).
  * `data_download.py`: One-time downloader for all raw datasets from UCI.
* `code/notebooks/`: Contains `demo.ipynb`, a Jupyter notebook for newcomers who do not know the project yet and want a guided way to run the pipeline on new data.
* `code/main.py`: Main experiment runner (full experiment grid).

## 🚀 How to Run the Code

### 1. Prerequisites
Ensure you have Python 3.12 installed. It is recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. (Optional) Download raw data

If `code/data/raw/` is empty in your clone, run:

```bash
python3 code/src/data_download.py
```

### 3. Run all experiments (parallel by default)

From project root, run:

```bash
python3 code/main.py
```

This runs the full grid defined in `code/main.py` (datasets, schemes, missing rates, seeds, and methods) and uses process-level parallelization by default.

> Note: the current script does not expose CLI flags for partial runs. To change the scope, edit the `DEFAULT_*` constants in `code/main.py`.

### 4. Notebook quick start for new users

If someone is new to this project and wants to quickly try our approach on new data, start with:

- `code/notebooks/demo.ipynb`

The notebook is a guided entry point that explains the workflow step by step and is the easiest way to run and adapt the pipeline interactively.
