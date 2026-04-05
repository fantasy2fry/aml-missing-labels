import pandas as pd
from src.data_prep import get_dataset_with_missing, get_all_schemes_for_dataset, AVAILABLE_DATASETS

if __name__ == "__main__":

    # Example 1: single dataset + single scheme
    print("\n=== Example 1: breast_cancer + MNAR ===")
    result = get_dataset_with_missing(
        dataset_name="breast_cancer",
        scheme="MNAR",
        missing_rate=0.30,
    )
    print(result["summary"].to_string())
    X, y, y_obs = result["X"], result["y"], result["y_obs"]
    print(f"Labeled   : {(y_obs != -1).sum()}")
    print(f"Unlabeled : {(y_obs == -1).sum()}")

    # Example 2: single dataset, all four schemes
    print("\n=== Example 2: spambase - all four schemes ===")
    all_schemes = get_all_schemes_for_dataset("spambase", missing_rate=0.25)
    print(all_schemes["summary"].to_string())

    # Example 3: all datasets x all schemes
    print("\n=== Example 3: overview - all datasets x all schemes ===")
    for ds in AVAILABLE_DATASETS:
        r = get_all_schemes_for_dataset(ds, missing_rate=0.30)
        print(f"\n{ds.upper()} - X shape: {r['X'].shape}")
        print(r["summary"].to_string())
        # show y head and X head
        print("y head:")
        print(pd.DataFrame(r["y"]).head())
        print("y_obs head:")
        print(pd.DataFrame(r["y_obs"]).head())
        print("X head:")
        print(pd.DataFrame(r["X"]).head())