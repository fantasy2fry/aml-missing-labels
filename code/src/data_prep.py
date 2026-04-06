"""
AML Project 1 - Task 1
=======================
Data preparation and missing-data generation schemes for logistic regression
with partially observed labels.

Prerequisites
-------------
Run data_download.py once to populate data/raw/ before using this module:

    python data_download.py

Datasets available
------------------
* 'spambase'      - Spambase
* 'sonar'         - Sonar: Mines vs Rocks
* 'breast_cancer' - Breast Cancer Wisconsin Diagnostic
* 'phishing'      - Phishing Websites

Missing-data mechanisms
-----------------------
* 'MCAR' - Missing Completely At Random:   P(S=1 | X, Y) = c
* 'MAR1' - Missing At Random (1 feature):  P(S=1 | X, Y) = P(S=1 | X_j)
* 'MAR2' - Missing At Random (all feats):  P(S=1 | X, Y) = P(S=1 | X)
* 'MNAR' - Missing Not At Random:          P(S=1 | X, Y) depends on Y

Quick start
-----------
    from task1 import get_dataset_with_missing, get_all_schemes_for_dataset

    # Single dataset + single scheme
    result  = get_dataset_with_missing('breast_cancer', 'MAR2', missing_rate=0.30)
    X       = result['X']        # np.ndarray - features (unscaled)
    y       = result['y']        # np.ndarray - true labels {0, 1}
    y_obs   = result['y_obs']    # np.ndarray - labels with -1 for missing
    summary = result['summary']  # pd.DataFrame - missing-rate breakdown

    # Single dataset + all four schemes at once
    all_schemes = get_all_schemes_for_dataset('spambase', missing_rate=0.25)
    y_obs_mcar  = all_schemes['y_obs']['MCAR']
    print(all_schemes['summary'])
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Directory where data_download.py saves raw CSV files
# Always resolve data/raw/ relative to this file (src/task1.py),
# so it points to <project_root>/data/raw/ regardless of cwd.
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AVAILABLE_DATASETS = ("spambase", "sonar", "breast_cancer", "phishing")
AVAILABLE_SCHEMES  = ("MCAR", "MAR1", "MAR2", "MNAR")

DatasetName   = Literal["spambase", "sonar", "breast_cancer", "phishing"]
MissingScheme = Literal["MCAR", "MAR1", "MAR2", "MNAR"]

# In-memory cache so repeated calls within a session skip disk I/O
_CACHE: dict[str, dict] = {}


# ===========================================================================
# 1. Data Loading  (reads from data/raw/*.csv)
# ===========================================================================

def load_dataset(dataset_name: str) -> dict[str, pd.DataFrame]:
    """Load a dataset from the local data/raw/ directory.

    Reads the two CSV files written by data_download.py:
        data/raw/<name>_X.csv
        data/raw/<name>_y.csv

    Results are cached in memory; subsequent calls in the same session
    are instant and require no disk access.

    Parameters
    ----------
    dataset_name : str
        One of: 'spambase', 'sonar', 'breast_cancer', 'phishing'.

    Returns
    -------
    dict with keys:
        'X' - pd.DataFrame of features (raw, unprocessed)
        'y' - pd.DataFrame of targets  (raw, not yet binarised)

    Raises
    ------
    FileNotFoundError
        If the CSV files do not exist. Run data_download.py first.
    """
    _validate_dataset_name(dataset_name)

    if dataset_name in _CACHE:
        return _CACHE[dataset_name]

    x_path = RAW_DIR / f"{dataset_name}_X.csv"
    y_path = RAW_DIR / f"{dataset_name}_y.csv"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Raw data files for '{dataset_name}' not found.\n"
            f"Expected:\n  {x_path}\n  {y_path}\n"
            "Run 'python data_download.py' first to download the datasets."
        )

    print(f"[load_dataset] Reading '{dataset_name}' from {RAW_DIR}/...")
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    print(f"[load_dataset] Done - X={X.shape}, y={y.shape}")

    _CACHE[dataset_name] = {"X": X, "y": y}
    return _CACHE[dataset_name]


def load_all_datasets() -> dict[str, dict]:
    """Load all four datasets from data/raw/ and return them as a nested dict.

    Returns
    -------
    dict mapping dataset_name -> {'X': DataFrame, 'y': DataFrame}
    """
    return {name: load_dataset(name) for name in AVAILABLE_DATASETS}


# ===========================================================================
# 2. Target Binarisation
# ===========================================================================

def _binarise_target(
    y: pd.Series | pd.DataFrame,
    dataset_name: str,
) -> np.ndarray:
    """Convert a raw target column to a binary {0, 1} numpy array.

    Dataset-specific rules
    ----------------------
    spambase      - already {0, 1}, no change needed.
    sonar         - 'R' (Rock) -> 0, 'M' (Mine) -> 1.
    breast_cancer - 'B' (Benign) -> 0, 'M' (Malignant) -> 1.
    phishing      - remap {-1 -> 0, 0 -> 0, 1 -> 1}.
    fallback      - majority class -> 0, all others -> 1.

    Parameters
    ----------
    y : pd.Series or single-column pd.DataFrame
    dataset_name : str

    Returns
    -------
    np.ndarray of int, values in {0, 1}
    """
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = y.copy()

    if dataset_name == "sonar":
        y = y.map({"R": 0, "M": 1})

    elif dataset_name == "breast_cancer":
        y = y.map({"B": 0, "M": 1})

    elif dataset_name == "phishing":
        y = y.apply(lambda v: 0 if int(v) == -1 else int(v))

    else:
        unique = set(y.unique())
        if unique != {0, 1}:
            majority = y.value_counts().idxmax()
            y = (y != majority).astype(int)

    return y.values.astype(int)


# ===========================================================================
# 3. Data Preparation
# ===========================================================================

class DatasetPreparator:
    """Prepare a raw dataset for logistic regression experiments.

    The sole responsibility of this class is binarising the target
    using dataset-specific rules and converting DataFrames to numpy arrays.
    Feature scaling is intentionally excluded and must be applied
    at a later pipeline stage.

    Attributes
    ----------
    feature_names_ : list[str]
        Column names of the input feature matrix (set after fit_transform).
    """

    def __init__(self) -> None:
        self.feature_names_: list[str] = []

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        dataset_name: str = "",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert (X, y) to numpy arrays and binarise the target.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (already cleaned - no missing values, numeric only).
        y : pd.Series or pd.DataFrame
            Raw targets to be binarised.
        dataset_name : str
            Used to select the dataset-specific binarisation rule for y.

        Returns
        -------
        X_out : np.ndarray, shape (n_samples, n_features)
            Feature matrix as a plain float array (unscaled).
        y_out : np.ndarray, shape (n_samples,), values in {0, 1}
        """
        self.feature_names_ = list(X.columns)
        X_out = X.values.astype(float)
        y_out = _binarise_target(y, dataset_name)
        return X_out, y_out

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Convert a new feature DataFrame to a float array.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain all columns listed in feature_names_.

        Returns
        -------
        np.ndarray (unscaled)
        """
        return X[self.feature_names_].values.astype(float)


# ===========================================================================
# 4. Missing-Label Generation
# ===========================================================================

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: s(z) = 1 / (1 + exp(-z))."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


def generate_missing_labels(
    X: np.ndarray,
    y: np.ndarray,
    scheme: str,
    missing_rate: float = 0.30,
    feature_idx: int = 0,
    random_state: int | None = 42,
) -> np.ndarray:
    """Introduce missing labels into a fully-labelled dataset.

    The returned array y_obs equals y for observed samples and -1 for
    samples whose label has been masked (S = 1).

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Feature matrix.
    y : np.ndarray, shape (n,)
        True binary labels {0, 1}.
    scheme : str
        One of: 'MCAR', 'MAR1', 'MAR2', 'MNAR'.
    missing_rate : float, default 0.30
        Target marginal probability P(S=1).
    feature_idx : int, default 0
        Index of the single feature used by MAR1.
    random_state : int or None, default 42
        Seed for reproducibility.

    Returns
    -------
    y_obs : np.ndarray, shape (n,)
        Observed labels: original value or -1 (masked).

    Mechanism details
    -----------------
    MCAR:
        P(S=1 | X, Y) = missing_rate for every observation (constant).

    MAR1:
        P(S=1 | X, Y) = sigmoid(x_j + bias), where x_j is feature
        feature_idx standardised to zero mean / unit variance. The bias
        is chosen so that mean probability ~= missing_rate.

    MAR2:
        P(S=1 | X, Y) = sigmoid(X @ w + bias), where w ~ N(0,1)^p
        (normalised). The entire feature vector drives missingness.

    MNAR:
        Positive-class observations (y=1) are masked more often than
        negative-class ones. Both class-conditional probabilities are
        calibrated so the marginal rate equals missing_rate.
    """
    _validate_scheme(scheme)
    rng = np.random.default_rng(random_state)
    n   = len(y)

    if scheme == "MCAR":
        prob = np.full(n, missing_rate)

    elif scheme == "MAR1":
        x_j   = X[:, feature_idx]
        x_std = (x_j - x_j.mean()) / (x_j.std() + 1e-8)
        bias  = np.log(missing_rate / (1.0 - missing_rate + 1e-8))
        prob  = _sigmoid(x_std + bias)

    elif scheme == "MAR2":
        w     = rng.standard_normal(X.shape[1])
        w    /= np.linalg.norm(w)
        z     = X @ w
        z_std = (z - z.mean()) / (z.std() + 1e-8)
        bias  = np.log(missing_rate / (1.0 - missing_rate + 1e-8))
        prob  = _sigmoid(z_std + bias)

    elif scheme == "MNAR":
        p_pos = min(missing_rate * 1.5, 0.95)
        p_neg = max(missing_rate * 0.5, 0.05)
        prob  = np.where(y == 1, p_pos, p_neg)
        prob  = np.clip(prob * (missing_rate / (prob.mean() + 1e-8)), 0.01, 0.99)

    s     = (rng.uniform(size=n) < prob).astype(int)
    y_obs = np.where(s == 1, -1, y)

    actual = (y_obs == -1).mean()
    print(
        f"[generate_missing_labels] scheme={scheme:<5}  "
        f"target={missing_rate:.2f}  actual={actual:.3f}"
    )
    return y_obs


# ===========================================================================
# 5. Summary Helper
# ===========================================================================

def summarise_missing(
    y: np.ndarray,
    y_obs_dict: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Return a DataFrame with overall and class-conditional missing rates.

    Parameters
    ----------
    y : np.ndarray
        True binary labels {0, 1}.
    y_obs_dict : dict
        Mapping scheme name -> y_obs array.

    Returns
    -------
    pd.DataFrame  (index = scheme name)
        Columns: missing_rate | miss_rate_y0 | miss_rate_y1
    """
    rows = []
    for scheme, y_obs in y_obs_dict.items():
        mask = y_obs == -1
        rows.append({
            "scheme":       scheme,
            "missing_rate": mask.mean(),
            "miss_rate_y0": mask[y == 0].mean() if (y == 0).any() else np.nan,
            "miss_rate_y1": mask[y == 1].mean() if (y == 1).any() else np.nan,
        })
    return pd.DataFrame(rows).set_index("scheme").round(3)


# ===========================================================================
# 6. Main Public API
# ===========================================================================

def get_dataset_with_missing(
    dataset_name: str,
    scheme: str,
    missing_rate: float = 0.30,
    feature_idx: int = 0,
    random_state: int | None = 42,
) -> dict:
    """Load, prepare, and apply a missing-label scheme in one call.

    Parameters
    ----------
    dataset_name : str
        One of: 'spambase', 'sonar', 'breast_cancer', 'phishing'.
    scheme : str
        One of: 'MCAR', 'MAR1', 'MAR2', 'MNAR'.
    missing_rate : float, default 0.30
        Target proportion of labels to mask.
    feature_idx : int, default 0
        Feature index used by the MAR1 scheme.
    random_state : int or None, default 42
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        'X'          - np.ndarray, features as float array (unscaled)
        'y'          - np.ndarray, true binary labels {0, 1}
        'y_obs'      - np.ndarray, observed labels (-1 = missing)
        'preparator' - fitted DatasetPreparator instance
        'summary'    - pd.DataFrame with missing-rate statistics
        'dataset'    - dataset_name (str)
        'scheme'     - scheme (str)

    Examples
    --------
    >>> result = get_dataset_with_missing('breast_cancer', 'MNAR', missing_rate=0.3)
    >>> X, y, y_obs = result['X'], result['y'], result['y_obs']
    >>> print(result['summary'])
    """
    _validate_dataset_name(dataset_name)
    _validate_scheme(scheme)

    print(f"\n{'='*60}")
    print(f"  dataset={dataset_name}  |  scheme={scheme}  |  missing_rate={missing_rate}")
    print(f"{'='*60}")

    raw  = load_dataset(dataset_name)
    prep = DatasetPreparator()
    X, y = prep.fit_transform(raw["X"], raw["y"], dataset_name=dataset_name)

    print(
        f"[prepare] features={X.shape[1]}  samples={X.shape[0]}  "
        f"class_balance(y=1)={y.mean():.3f}"
    )

    y_obs = generate_missing_labels(
        X, y,
        scheme=scheme,
        missing_rate=missing_rate,
        feature_idx=feature_idx,
        random_state=random_state,
    )

    return {
        "X":          X,
        "y":          y,
        "y_obs":      y_obs,
        "preparator": prep,
        "summary":    summarise_missing(y, {scheme: y_obs}),
        "dataset":    dataset_name,
        "scheme":     scheme,
    }


def get_all_schemes_for_dataset(
    dataset_name: str,
    missing_rate: float = 0.30,
    feature_idx: int = 0,
    random_state: int | None = 42,
) -> dict:
    """Load, prepare, and generate all four missing-label schemes at once.

    Parameters
    ----------
    dataset_name : str
        One of: 'spambase', 'sonar', 'breast_cancer', 'phishing'.
    missing_rate : float, default 0.30
    feature_idx : int, default 0
    random_state : int or None, default 42

    Returns
    -------
    dict with keys:
        'X'          - np.ndarray, features as float array (unscaled)
        'y'          - np.ndarray, true labels {0, 1}
        'y_obs'      - dict mapping scheme name -> y_obs np.ndarray
        'preparator' - fitted DatasetPreparator
        'summary'    - pd.DataFrame comparing all four schemes

    Examples
    --------
    >>> r = get_all_schemes_for_dataset('spambase', missing_rate=0.25)
    >>> print(r['summary'])
    >>> y_obs_mcar = r['y_obs']['MCAR']
    """
    _validate_dataset_name(dataset_name)

    raw  = load_dataset(dataset_name)
    prep = DatasetPreparator()
    X, y = prep.fit_transform(raw["X"], raw["y"], dataset_name=dataset_name)

    y_obs_dict: dict[str, np.ndarray] = {
        scheme: generate_missing_labels(
            X, y,
            scheme=scheme,
            missing_rate=missing_rate,
            feature_idx=feature_idx,
            random_state=random_state,
        )
        for scheme in AVAILABLE_SCHEMES
    }

    return {
        "X":          X,
        "y":          y,
        "y_obs":      y_obs_dict,
        "preparator": prep,
        "summary":    summarise_missing(y, y_obs_dict),
    }


# ===========================================================================
# 7. Input Validation
# ===========================================================================

def _validate_dataset_name(name: str) -> None:
    if name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'.\n"
            f"Available: {AVAILABLE_DATASETS}"
        )


def _validate_scheme(scheme: str) -> None:
    if scheme not in AVAILABLE_SCHEMES:
        raise ValueError(
            f"Unknown scheme '{scheme}'.\n"
            f"Available: {AVAILABLE_SCHEMES}"
        )

# ===========================================================================
# 8. Convertion to Partial Labeling
# ===========================================================================

def convert_to_pl(y: np.ndarray) -> np.ndarray:
    """
    Convert 1D label array with values {-1, 0, 1} into 2D partial label matrix.

    Mapping:
        0  -> [1, 0]
        1  -> [0, 1]
       -1  -> [1, 1]

    Parameters
    -------
    y : np.ndarray
        Labels with missing values {0, 1, -1}.

    Returns
    -------
    np.ndarray of shape (n_samples, 2)
    """
    new_y = np.empty((y.shape[0],2),dtype=int)
    new_y[:, 0] = (y != 1)
    new_y[:, 1] = (y != 0)
    return new_y