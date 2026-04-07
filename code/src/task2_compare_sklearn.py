import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import os

from src.fista import FistaLogisticRegression

def load_clean_data(dataset_name="spambase"):
    """Loads dataset without missing values (as requested in Task 2)"""
    base_dir = os.path.dirname(__file__)
    X_path = os.path.join(base_dir, "..", "data", "raw", f"{dataset_name}_X.csv")
    y_path = os.path.join(base_dir, "..", "data", "raw", f"{dataset_name}_y.csv")
    X = pd.read_csv(X_path).values
    y = pd.read_csv(y_path).values.ravel()
    
    # Binarize targets if needed
    unique_classes = np.unique(y)
    if len(unique_classes) != 2:
        raise ValueError(
            f"Expected exactly 2 target classes in dataset '{dataset_name}', "
            f"but found {len(unique_classes)}: {unique_classes.tolist()}"
        )
    y = np.where(y == unique_classes[1], 1, 0)
    
    return X, y

def main():
    print("=== Task 2: FISTA vs Sklearn Comparison ===\n")
    
    # 1. Load and prepare data (without missing values)
    X, y = load_clean_data("spambase")
    
    # Split into train, validation, and test sets (60 / 20 / 20)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Standardize features using training data only to avoid leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    
    print(f"Data shapes - Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}\n")

    # 2. Train Our FISTA Implementation (computes the full regularization path)
    print("--- Training Own FISTA Logistic Regression Path ---")
    lambdas = np.logspace(-4, 1, 50)
    fista_model = FistaLogisticRegression(lambdas=lambdas, max_iter=1000, tol=1e-4)
    
    start_path_time = time.time()
    fista_model.fit(X_train, y_train)
    fista_path_time = time.time() - start_path_time
    print(f"FISTA Path Training Time (50 lambdas): {fista_path_time:.3f} s\n")

    # 3. Iterate through all required measures
    measures = ["recall", "precision", "f-measure", "balanced_accuracy", "roc_auc", "pr_auc"]
    
    comparison_results = []
    plots_dir = os.path.join(os.path.dirname(__file__), "..", "..", "plots_and_results")
    os.makedirs(plots_dir, exist_ok=True)
    
    for measure in measures:
        print(f"==================================================")
        print(f"Evaluating and comparing for measure: '{measure}'")
        print(f"==================================================")
        
        # Validate FISTA on the specific measure
        best_lambda, best_score = fista_model.validate(X_valid, y_valid, measure=measure)
        print(f"[FISTA] Best lambda found for {measure}: {best_lambda:.5f}")
        
        # --- MEASURE FISTA TIME FOR SINGLE BEST LAMBDA (Fair comparison) ---
        single_fista = FistaLogisticRegression(lambdas=[best_lambda], max_iter=1000, tol=1e-4)
        start_fista_single = time.time()
        single_fista.fit(X_train, y_train)
        fista_single_time = time.time() - start_fista_single
        # --------------------------------------------------------------------

        # Predict on Test set
        fista_probas = fista_model.predict_proba(X_test)[:, 1]
        fista_preds = (fista_probas >= 0.5).astype(int)
        
        fista_roc = roc_auc_score(y_test, fista_probas)
        fista_f1 = f1_score(y_test, fista_preds)
        # Usingcoef_ from the main model which is already set to the best one after validate()
        fista_sparsity = np.mean(np.abs(fista_model.coef_) < 1e-5) 
        
        print(f"[FISTA] Test ROC AUC: {fista_roc:.4f}")
        print(f"[FISTA] Test F1 Score: {fista_f1:.4f}")
        print(f"[FISTA] Single Fit Time: {fista_single_time:.3f} s")
        print(f"[FISTA] Sparsity (zeroed features): {fista_sparsity * 100:.1f}%\n")
        
        # Generate Plots for this measure
        plot1 = os.path.join(plots_dir, f"fista_measure_{measure}.png")
        if os.path.exists(plot1): os.remove(plot1)
        fista_model.plot()
        plt.savefig(plot1, bbox_inches='tight')
        plt.close()
        
        print(f"-> Plot saved in 'plots_and_results' folder as 'fista_measure_{measure}.png'.\n")
        
        # 4. Train Scikit-Learn with the equivalent penalty 
        n_samples = len(X_train)
        C_val = 1.0 / (best_lambda * n_samples) if best_lambda > 0 else 1.0
        # Sklearn minimizes the sum of errors, while our FISTA minimizes the average error.
        # Therefore, C must be scaled by n_samples: C = 1 / (lambda * N)
        
        sklearn_model = LogisticRegression(solver='saga', C=C_val, l1_ratio=1.0, max_iter=5000, random_state=42)
        # above ideally lasso
        start_sk_time = time.time()
        sklearn_model.fit(X_train, y_train)
        sklearn_time = time.time() - start_sk_time
        
        sk_probas = sklearn_model.predict_proba(X_test)[:, 1]
        sk_preds = (sk_probas >= 0.5).astype(int)
        
        sk_roc = roc_auc_score(y_test, sk_probas)
        sk_f1 = f1_score(y_test, sk_preds)
        sk_sparsity = np.mean(np.abs(sklearn_model.coef_[0]) < 1e-5)
        
        print(f"[Sklearn] Time: {sklearn_time:.3f} s")
        print(f"[Sklearn] Test ROC AUC: {sk_roc:.4f}")
        print(f"[Sklearn] Test F1 Score: {sk_f1:.4f}")
        print(f"[Sklearn] Sparsity: {sk_sparsity * 100:.1f}%\n")
        
        # Save results to list for table
        comparison_results.append({
            "Optimized_Measure": measure,
            "Best_Lambda": round(best_lambda, 5),
            "FISTA_ROC_AUC": round(fista_roc, 4),
            "Sklearn_ROC_AUC": round(sk_roc, 4),
            "FISTA_F1_Score": round(fista_f1, 4),
            "Sklearn_F1_Score": round(sk_f1, 4),
            "FISTA_Sparsity_%": round(fista_sparsity * 100, 1),
            "Sklearn_Sparsity_%": round(sk_sparsity * 100, 1),
            "FISTA_Time_s": round(fista_single_time, 3),
            "Sklearn_Time_s": round(sklearn_time, 3)
        })

    # Save to CSV
    df_results = pd.DataFrame(comparison_results)
    csv_path = os.path.join(plots_dir, "comparison_metrics.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"-> Full comparison table saved to '{csv_path}'\n")

    # Generate Bar Chart Comparison
    metrics_to_plot = ["FISTA_ROC_AUC", "Sklearn_ROC_AUC"]
    df_results.plot(x="Optimized_Measure", y=metrics_to_plot, kind="bar", figsize=(10, 6))
    plt.title("Performance Comparison: FISTA vs Scikit-Learn (ROC AUC)")
    plt.ylabel("ROC AUC Score")
    
    min_val = df_results[metrics_to_plot].min().min()
    plt.ylim(max(0, min_val - 0.05), 1.0)
    
    plt.xticks(rotation=45)
    plt.legend(["FISTA (Own)", "Scikit-Learn"])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    bar_chart_path = os.path.join(plots_dir, "fista_vs_sklearn_barchart.png")
    if os.path.exists(bar_chart_path): os.remove(bar_chart_path)
    plt.savefig(bar_chart_path, bbox_inches='tight')
    plt.close()
    print(f"-> Bar chart comparison saved as 'fista_vs_sklearn_barchart.png'.\n")

    # Generate ONE plot for coefficients
    print(f"==================================================")
    print("Generating ONE common coefficient plot...")
    fista_model.validate(X_valid, y_valid, measure="roc_auc")
    
    plot_coefs_path = os.path.join(plots_dir, "fista_coefficients.png")
    if os.path.exists(plot_coefs_path): os.remove(plot_coefs_path)
    fista_model.plot_coefficients()
    plt.savefig(plot_coefs_path, bbox_inches='tight')
    plt.close()
    
    print(f"-> Overall Lasso Paths plot saved as 'fista_coefficients.png'.")

if __name__ == "__main__":
    main()