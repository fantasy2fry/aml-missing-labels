import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import os

from src.fista import FistaLogisticRegression
from src.data_prep import get_dataset_with_missing

def main():
    print("=== Task 2: FISTA vs Sklearn Comparison ===\n")
    
    datasets = ["spambase", "sonar", "breast_cancer", "phishing"]
    measures = ["recall", "precision", "f-measure", "balanced_accuracy", "roc_auc", "pr_auc"]
    
    plots_dir = os.path.join(os.path.dirname(__file__), "..", "..", "plots_and_results")
    os.makedirs(plots_dir, exist_ok=True)
    all_comparison_results = []

    for dataset_name in datasets:
        print(f"\n{'#'*60}")
        print(f"      STARTING ANALYSIS FOR DATASET: {dataset_name.upper()}")
        print(f"{'#'*60}\n")
        
        # 1. Load data from data_prep using get_dataset_with_missing 
        # (We use MCAR 0 as a dummy parameter just to extract pristine X and y)
        data_dict = get_dataset_with_missing(dataset_name, scheme="MCAR", missing_rate=0.0)
        X, y = data_dict['X'], data_dict['y']
        
        # Split into train, validation, and test sets (60 / 20 / 20)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # Standardize features using training data only to avoid leakage
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)
        
        print(f"Data shapes - Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}\n")

        # 2. Train Our FISTA Implementation
        print(f"--- Training FISTA Path for {dataset_name} ---")
        lambdas = np.logspace(-4, 1, 50)
        fista_model = FistaLogisticRegression(lambdas=lambdas, max_iter=1000, tol=1e-4)
        
        start_path_time = time.time()
        fista_model.fit(X_train, y_train)
        fista_path_time = time.time() - start_path_time
        print(f"FISTA Path Training Time (50 lambdas): {fista_path_time:.3f} s\n")
        
        dataset_results = []
        
        for measure in measures:
            print(f"--- Measure: '{measure}' ---")
            
            # Validate FISTA regularization path to select the best lambda
            best_lambda, best_score = fista_model.validate(X_valid, y_valid, measure=measure)
            print(f"[FISTA] Best lambda: {best_lambda:.5f}")
            
            # Measure strictly fair single-fit time using only the selected lambda
            single_fista = FistaLogisticRegression(lambdas=[best_lambda], max_iter=1000, tol=1e-4)
            start_fista_single = time.time()
            single_fista.fit(X_train, y_train)
            fista_single_time = time.time() - start_fista_single
            single_fista.validate(X_valid, y_valid, measure=measure)

            # Predict on Test set with the same single-fit model used for timing
            fista_probas = single_fista.predict_proba(X_test)[:, 1]
            fista_preds = (fista_probas >= 0.5).astype(int)
            
            fista_roc = roc_auc_score(y_test, fista_probas)
            fista_f1 = f1_score(y_test, fista_preds)
            fista_sparsity = np.mean(np.abs(single_fista.coef_) < 1e-5) 
            
            # Generate Plots
            plot1 = os.path.join(plots_dir, f"fista_measure_{dataset_name}_{measure}.png")
            if os.path.exists(plot1): os.remove(plot1)
            fista_model.plot()
            plt.savefig(plot1, bbox_inches='tight')
            plt.close()
            
            # 4. Train Scikit-Learn
            n_samples = len(X_train)
            C_val = 1.0 / (best_lambda * n_samples) if best_lambda > 0 else 1.0
            
            sklearn_model = LogisticRegression(solver='saga', C=C_val, l1_ratio=1.0, max_iter=5000, random_state=42)
            start_sk_time = time.time()
            sklearn_model.fit(X_train, y_train)
            sklearn_time = time.time() - start_sk_time
            
            sk_probas = sklearn_model.predict_proba(X_test)[:, 1]
            sk_preds = (sk_probas >= 0.5).astype(int)
            
            sk_roc = roc_auc_score(y_test, sk_probas)
            sk_f1 = f1_score(y_test, sk_preds)
            sk_sparsity = np.mean(np.abs(sklearn_model.coef_[0]) < 1e-5)
            
            print(f"-> FISTA  [AUC: {fista_roc:.4f}, Sparsity: {fista_sparsity*100:.1f}%, Time: {fista_single_time:.3f}s]")
            print(f"-> Sklearn [AUC: {sk_roc:.4f}, Sparsity: {sk_sparsity*100:.1f}%, Time: {sklearn_time:.3f}s]\n")
            
            # Compute the actual test score for this respective measure on both models
            if measure == "recall":
                f_val = recall_score(y_test, fista_preds, zero_division=0)
                sk_val = recall_score(y_test, sk_preds, zero_division=0)
            elif measure == "precision":
                f_val = precision_score(y_test, fista_preds, zero_division=0)
                sk_val = precision_score(y_test, sk_preds, zero_division=0)
            elif measure == "f-measure":
                f_val = f1_score(y_test, fista_preds, zero_division=0)
                sk_val = f1_score(y_test, sk_preds, zero_division=0)
            elif measure == "balanced_accuracy":
                f_val = balanced_accuracy_score(y_test, fista_preds)
                sk_val = balanced_accuracy_score(y_test, sk_preds)
            elif measure == "roc_auc":
                f_val = roc_auc_score(y_test, fista_probas)
                sk_val = roc_auc_score(y_test, sk_probas)
            elif measure == "pr_auc":
                f_val = average_precision_score(y_test, fista_probas)
                sk_val = average_precision_score(y_test, sk_probas)
            else:
                f_val = sk_val = 0.0

            # Save results
            dataset_results.append({
                "Dataset": dataset_name,
                "Optimized_Measure": measure,
                "Best_Lambda": round(best_lambda, 5),
                "FISTA_Test_Score": round(f_val, 4),
                "Sklearn_Test_Score": round(sk_val, 4),
                "FISTA_ROC_AUC": round(fista_roc, 4),
                "Sklearn_ROC_AUC": round(sk_roc, 4),
                "FISTA_F1_Score": round(fista_f1, 4),
                "Sklearn_F1_Score": round(sk_f1, 4),
                "FISTA_Sparsity_%": round(fista_sparsity * 100, 1),
                "Sklearn_Sparsity_%": round(sk_sparsity * 100, 1),
                "FISTA_Time_s": round(fista_single_time, 3),
                "Sklearn_Time_s": round(sklearn_time, 3)
            })
            all_comparison_results.append(dataset_results[-1])

        # Generate Bar Chart Comparison for this dataset
        df_dataset = pd.DataFrame(dataset_results)
        
        # Plotted height is the actual score achieved for the optimized metric
        metrics_to_plot = ["FISTA_Test_Score", "Sklearn_Test_Score"] 
        df_dataset.plot(x="Optimized_Measure", y=metrics_to_plot, kind="bar", figsize=(10, 6))
        plt.title(f"Performance Comparison ({dataset_name.capitalize()})")
        plt.ylabel("Testing Score\n(Value of the respective optimized measure)")
        plt.xlabel("Tested Measure")
        
        # Start axis strictly from 0 to 1.2 (preventing truncation)
        plt.ylim(0.0, 1.2)
        
        plt.xticks(rotation=45)
        plt.legend(["FISTA (Own)", "Scikit-Learn"])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        bar_chart_path = os.path.join(plots_dir, f"fista_vs_sklearn_barchart_{dataset_name}.png")
        if os.path.exists(bar_chart_path): os.remove(bar_chart_path)
        plt.savefig(bar_chart_path, bbox_inches='tight')
        plt.close()

        # Generate ONE plot for coefficients per dataset
        fista_model.validate(X_valid, y_valid, measure="roc_auc")
        plot_coefs_path = os.path.join(plots_dir, f"fista_coefficients_{dataset_name}.png")
        if os.path.exists(plot_coefs_path): os.remove(plot_coefs_path)
        fista_model.plot_coefficients()
        plt.savefig(plot_coefs_path, bbox_inches='tight')
        plt.close()

    # Save final combined CSV
    df_all = pd.DataFrame(all_comparison_results)
    csv_path = os.path.join(plots_dir, "comparison_metrics_all.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\n==================================================")
    print(f"-> Full multisets table saved to '{csv_path}'")
    print(f"==================================================")

if __name__ == "__main__":
    main()