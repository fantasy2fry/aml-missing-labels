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
3. **UnlabeledLogReg:** A custom classifier that utilizes both labeled and unlabeled data, incorporating two different algorithms for completing the missing $Y$ labels.
4. **Benchmarking:** Comparison of our `UnlabeledLogReg` method against a **Naive method** (using only labeled data) and an **Oracle method** (using fully labeled, completely observed data).

## 🗂 Repository Structure
The project is structured into two main directories: `code/` and `report/`, adhering to the submission guidelines.

* `code/data/`: Contains the 4 real-world datasets used for experiments (raw and processed).
* `code/src/`: Core Python modules.
  * `data_prep.py`: Data cleaning, missing value imputation, and $Y$-missing data generation schemes.
  * `fista.py`: Contains the `fit`, `predict_proba`, and `validate` methods for the FISTA algorithm.
  * `unlabeled_logreg.py`: Contains the `UnlabeledLogReg` class and $Y$ completion algorithms.
  * `evaluation.py`: Functions for calculating metrics (Accuracy, F1, ROC AUC, etc.) and plotting validation curves/coefficients.
* `code/notebooks/`: Jupyter notebooks used for data exploration and running the comprehensive experiments required for the report.
* `code/main.py`: The main script to easily run the algorithm pipeline on new datasets.
* `report/`: Contains the final 6-page project report detailing implementation correctness, methodology, and experiment analysis.

## 🚀 How to Run the Code

### 1. Prerequisites
Ensure you have Python 3.12 installed. It is recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
