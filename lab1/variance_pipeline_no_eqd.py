#!/usr/bin/env python3
"""
variance_pipeline_no_eqd.py

Full pipeline for imputation, scaling, automated variance-threshold selection (multiple heuristics),
and a PyTorch TimeSeriesDataset — EQD/EVT method intentionally removed per request.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import io
import math
import warnings

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import genpareto  # used by nothing here; kept for completeness
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import VarianceThreshold

# kneed is optional for elbow detection
try:
    from kneed import KneeLocator
    _HAS_KNEED = True
except Exception:
    _HAS_KNEED = False


# -----------------------------
# Scalers
# -----------------------------
class BaseScaler:
    def fit(self, X: np.ndarray):
        raise NotImplementedError
    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

class SimpleMinMaxScaler(BaseScaler):
    def __init__(self, feature_range=(0.0, 1.0)):
        self._min = None
        self._max = None
        self._range = feature_range
        self._denom = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self._min = np.nanmin(X, axis=0)
        self._max = np.nanmax(X, axis=0)
        self._denom = np.where(self._max - self._min == 0, 1.0, (self._max - self._min))

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        scale = (self._range[1] - self._range[0]) / self._denom
        return self._range[0] + (X - self._min) * scale

class SimpleZScoreScaler(BaseScaler):
    def __init__(self):
        self._mean = None
        self._std = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self._mean = np.nanmean(X, axis=0)
        self._std  = np.nanstd(X, axis=0)
        self._std = np.where(self._std == 0, 1.0, self._std)

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self._mean) / self._std

class SimpleRobustScaler(BaseScaler):
    def __init__(self):
        self._median = None
        self._iqr = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        # compute per-column 75th and 25th percentiles
        q75 = np.nanpercentile(X, 75, axis=0)
        q25 = np.nanpercentile(X, 25, axis=0)
        self._median = np.nanmedian(X, axis=0)
        self._iqr = q75 - q25

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        iqr_safe = np.where(self._iqr == 0, 1.0, self._iqr)
        return (X - self._median) / iqr_safe


# -----------------------------
# Imputer
# -----------------------------
class Imputer:
    """
    Imputer supports: strategy in {'mean','median','most_frequent'}.
    Supply columns to restrict imputation, otherwise infers numeric vs categorical.
    """
    def __init__(self, strategy="mean", columns: Optional[List[str]] = None):
        assert strategy in ("mean", "median", "most_frequent")
        self.strategy = strategy
        self.columns = columns
        self.statistics_: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame):
        if self.columns is None:
            if self.strategy in ("mean", "median"):
                cols = df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        else:
            cols = list(self.columns)

        for col in cols:
            if self.strategy in ("mean", "median"):
                s = pd.to_numeric(df[col], errors='coerce')
                stat = s.mean(skipna=True) if self.strategy == "mean" else s.median(skipna=True)
                if pd.isna(stat):
                    mode = df[col].mode(dropna=True)
                    stat = mode.iloc[0] if not mode.empty else 0.0
                self.statistics_[col] = stat
            else:
                mode = df[col].mode(dropna=True)
                self.statistics_[col] = mode.iloc[0] if not mode.empty else np.nan
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for col, val in self.statistics_.items():
            if col in df_copy.columns:
                df_copy[col].fillna(val, inplace=True)
        return df_copy

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# -----------------------------
# AutomatedVarianceThreshold (without EQD/EVT)
# -----------------------------
class AutomatedVarianceThreshold:
    """
    Automated variance threshold selection with multiple heuristic/statistical methods.
    Methods available:
      - 'percentile'        : keep features above a percentile of variance
      - 'elbow'             : elbow/knee detection on sorted variances (kneed fallback)
      - 'cross_validation'  : supervised CV to choose threshold (requires y)
      - 'information_theory' : GMM + AIC to find natural separation
      - 'emc_inspired'      : EMC-like separability (unsupervised: CV; supervised: F-ratio)
    """
    def __init__(self, method: str = 'percentile', **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.threshold_ : Optional[float] = None
        self.selected_features_ : Optional[np.ndarray] = None
        self.variance_scores_ : Optional[np.ndarray] = None
        self.selector_info_ : Dict[str, Any] = {}

    def _percentile_method(self, variances: np.ndarray, percentile: float = 95.0) -> float:
        return float(np.percentile(variances, percentile))

    def _elbow_method(self, variances: np.ndarray) -> float:
        if not _HAS_KNEED:
            warnings.warn("kneed not available; falling back to percentile(95)")
            return self._percentile_method(variances, percentile=95.0)

        sorted_vars = np.sort(variances)[::-1]
        x_range = list(range(len(sorted_vars)))
        try:
            kneedle = KneeLocator(x_range, sorted_vars, curve="convex", direction="decreasing", interp_method='polynomial')
            if kneedle.knee is not None:
                idx = int(kneedle.knee)
                return float(sorted_vars[idx])
            else:
                return self._percentile_method(variances, percentile=95.0)
        except Exception:
            return self._percentile_method(variances, percentile=95.0)

    def _cross_validation_method(self, X: np.ndarray, y: Optional[np.ndarray] = None, cv_folds: int = 5) -> float:
        if y is None:
            # unsupervised fallback
            return self._percentile_method(np.var(X, axis=0))
        variances = np.var(X, axis=0)
        thresholds = np.percentile(variances, np.arange(50, 99, 5))
        best_threshold = thresholds[0]
        best_score = -np.inf
        clf = LogisticRegression(random_state=42, max_iter=1000)
        for thr in thresholds:
            sel = variances >= thr
            if sel.sum() < 2:
                continue
            Xs = X[:, sel]
            try:
                scores = cross_val_score(clf, Xs, y, cv=cv_folds, scoring='accuracy')
                mean_score = float(np.mean(scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_threshold = thr
            except Exception:
                continue
        return float(best_threshold)

    def _information_theory_method(self, variances: np.ndarray) -> float:
        v = np.asarray(variances).reshape(-1, 1)
        # Try a few components and choose by AIC
        best_aic = np.inf
        best_thr = self._percentile_method(variances, percentile=95.0)
        max_components = min(6, max(2, len(variances)//10))
        for n in range(2, max_components+1):
            try:
                gmm = GaussianMixture(n_components=n, random_state=self.kwargs.get('random_state', 42))
                gmm.fit(v)
                aic = gmm.aic(v)
                if aic < best_aic:
                    best_aic = aic
                    means = np.sort(gmm.means_.flatten())
                    if len(means) >= 2:
                        best_thr = float((means[-1] + means[-2]) / 2.0)
            except Exception:
                continue
        return float(best_thr)

    def _emc_inspired_method(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        # Unsupervised: coefficient of variation ranking; supervised: between/within variance ratio
        X = np.asarray(X)
        if y is None:
            means = np.mean(X, axis=0)
            stds = np.std(X, axis=0)
            cv = np.where(np.abs(means) > 0, stds / np.abs(means), stds)
            thr = np.percentile(cv, self.kwargs.get('percentile', 95))
            mask = cv >= thr
            self.selector_info_['cv_scores'] = cv
            self.selector_info_['cv_threshold'] = thr
            return mask
        else:
            y = np.asarray(y)
            labels = np.unique(y)
            separability = []
            for j in range(X.shape[1]):
                col = X[:, j]
                overall_mean = np.mean(col)
                between = 0.0
                within = 0.0
                for lab in labels:
                    mask = (y == lab)
                    class_vals = col[mask]
                    c_mean = np.mean(class_vals) if len(class_vals) > 0 else 0.0
                    between += len(class_vals) * (c_mean - overall_mean)**2
                    within += np.sum((class_vals - c_mean)**2)
                score = (between / within) if within > 0 else between
                separability.append(score)
            sep = np.array(separability)
            thr = np.percentile(sep, self.kwargs.get('percentile', 95))
            mask = sep >= thr
            self.selector_info_['separability_scores'] = sep
            self.selector_info_['separability_threshold'] = thr
            return mask

    # ---------- Public API ----------
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit the selector on feature matrix X (n_samples, n_features).
        If method requires labels (cross_validation or supervised emc_inspired), supply y.
        """
        X = np.asarray(X)
        self.variance_scores_ = np.var(X, axis=0)
        method = self.method

        if method == 'percentile':
            percentile = float(self.kwargs.get('percentile', 95.0))
            self.threshold_ = self._percentile_method(self.variance_scores_, percentile)
            self.selected_features_ = (self.variance_scores_ >= self.threshold_)

        elif method == 'elbow':
            self.threshold_ = self._elbow_method(self.variance_scores_)
            self.selected_features_ = (self.variance_scores_ >= self.threshold_)

        elif method == 'cross_validation':
            cv_folds = int(self.kwargs.get('cv_folds', 5))
            self.threshold_ = self._cross_validation_method(X, y, cv_folds=cv_folds)
            self.selected_features_ = (self.variance_scores_ >= self.threshold_)

        elif method == 'information_theory':
            self.threshold_ = self._information_theory_method(self.variance_scores_)
            self.selected_features_ = (self.variance_scores_ >= self.threshold_)

        elif method == 'emc_inspired':
            mask = self._emc_inspired_method(X, y)
            self.selected_features_ = np.asarray(mask, dtype=bool)
            if self.selected_features_.any():
                self.threshold_ = float(np.min(self.variance_scores_[self.selected_features_]))
            else:
                self.threshold_ = float(np.min(self.variance_scores_))

        else:
            raise ValueError(f"Unknown method '{method}'")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted yet.")
        return np.asarray(X)[:, self.selected_features_]

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_selected_features_mask(self) -> np.ndarray:
        return self.selected_features_

    def get_threshold(self) -> float:
        return float(self.threshold_) if self.threshold_ is not None else None

    def get_feature_scores(self) -> np.ndarray:
        return self.variance_scores_


# -----------------------------
# TimeSeriesDataset
# -----------------------------
class TimeSeriesDataset(Dataset):
    """
    PyTorch dataset producing sliding windows and outputs for time series.
    Returns x: shape (window_size, n_features), y: shape (num_outputs,) or (num_outputs, n_targets)
    """
    def __init__(self, dataframe: pd.DataFrame, window_size: int, num_outputs: int, stride: int = 1, target_column: Optional[str] = None):
        self.df = dataframe.reset_index(drop=True)
        self.window_size = int(window_size)
        self.num_outputs = int(num_outputs)
        self.stride = int(stride)
        if target_column is not None:
            if target_column not in self.df.columns:
                raise ValueError("target_column not found in DataFrame")
            self.target_idx = list(self.df.columns).index(target_column)
        else:
            self.target_idx = 0
        self.data = self.df.values.astype(np.float32)
        max_idx = len(self.data) - self.window_size - self.num_outputs + 1
        self.indices = list(range(0, max(0, max_idx), self.stride)) if max_idx > 0 else []

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        start = self.indices[idx]
        x = self.data[start:start + self.window_size]
        y_start = start + self.window_size
        y = self.data[y_start:y_start + self.num_outputs, self.target_idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# -----------------------------
# Analysis utilities
# -----------------------------
def analyze_variance_threshold_methods(X: np.ndarray, y: Optional[np.ndarray] = None, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    methods = ['percentile', 'elbow', 'information_theory', 'emc_inspired']
    if y is not None:
        methods.append('cross_validation')
    results = {}
    for method in methods:
        try:
            selector = AutomatedVarianceThreshold(method=method)
            # pass parameters if desired
            if method == 'percentile':
                selector.kwargs['percentile'] = 95
            if method == 'emc_inspired':
                selector.kwargs['percentile'] = 95
            X_selected = selector.fit_transform(X, y)
            mask = selector.get_selected_features_mask()
            results[method] = {
                'threshold': selector.get_threshold(),
                'n_features_selected': X_selected.shape[1],
                'selected_features_mask': mask,
                'feature_scores': selector.get_feature_scores()
            }
            if feature_names is not None:
                results[method]['selected_feature_names'] = [n for n, m in zip(feature_names, mask) if m]
        except Exception as e:
            results[method] = {'error': str(e)}
    return results


# -----------------------------
# Example demo and pipeline runner
# -----------------------------
def demonstrate_automated_thresholding():
    print("=== Automated Variance Threshold Selection Demo (no EQD/EVT) ===\n")
    np.random.seed(42)
    n_samples, n_features = 1000, 50
    X = np.random.randn(n_samples, n_features)
    X[:, :10] *= 0.01
    X[:, 10:20] *= 0.1
    X[:, 20:35] *= 1.0
    X[:, 35:] *= 2.0
    y = (X[:, 35] + X[:, 40] + 0.5 * X[:, 45] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    results = analyze_variance_threshold_methods(X, y, feature_names)
    print("Results Summary:")
    print("-" * 60)
    for method, res in results.items():
        if 'error' in res:
            print(f"{method:20}: ERROR - {res['error']}")
        else:
            thr = res['threshold']
            nsel = res['n_features_selected']
            print(f"{method:20}: {nsel:2d} features selected (threshold: {thr:.6g})")
    return results


def run_pipeline_on_telco_file(telco_path: str, output_dir: Optional[str] = None):
    """
    Load telco_customer_churn.csv, run imputation, one-hot encoding,
    scale using three scalers, and apply AutomatedVarianceThreshold methods.
    Saves pruned CSVs to output_dir if provided.
    """
    df = pd.read_csv(telco_path)
    # ensure TotalCharges numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # drop identifier columns commonly present
    for idcol in ['customerID', 'CustomerID', 'id', 'Id']:
        if idcol in df.columns:
            df = df.drop(columns=[idcol])

    # identify categorical columns (exclude target 'Churn' if present)
    exclude = ['Churn']
    cat_cols = [c for c in df.select_dtypes(include=['object','category','bool']).columns if c not in exclude]

    # impute numeric columns (median)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_imputer = Imputer(strategy="median", columns=numeric_cols)
    df = num_imputer.fit_transform(df)

    # impute categorical (most frequent)
    if cat_cols:
        cat_imputer = Imputer(strategy="most_frequent", columns=cat_cols)
        df = cat_imputer.fit_transform(df)

    # one-hot encode categorical columns
    df_ohe = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)

    # safety fill remaining NaNs with numeric medians
    if df_ohe.isna().values.any():
        df_ohe = df_ohe.fillna(df_ohe.median(numeric_only=True))

    # choose numeric columns (exclude Churn if present)
    numeric_cols = [c for c in df_ohe.columns if df_ohe[c].dtype.kind in "fi" and c != 'Churn']

    scalers = {
        'minmax': SimpleMinMaxScaler(feature_range=(0.0, 1.0)),
        'zscore': SimpleZScoreScaler(),
        'robust': SimpleRobustScaler()
    }

    # run each scaler and several selector methods
    overall_reports = {}
    for scaler_name, scaler in scalers.items():
        print(f"\n--- Scaler: {scaler_name} ---")
        Xnum = df_ohe[numeric_cols].to_numpy(dtype=float)
        scaler.fit(Xnum)
        Xs = scaler.transform(Xnum)
        scaled_df = pd.DataFrame(Xs, columns=numeric_cols, index=df_ohe.index)
        other_cols = [c for c in df_ohe.columns if c not in numeric_cols]
        combined = pd.concat([scaled_df, df_ohe[other_cols].reset_index(drop=True)], axis=1)

        # compute numeric variances (only numeric columns)
        numeric_combined = combined.select_dtypes(include=[np.number])
        var_series = numeric_combined.var(axis=0, ddof=0)

        # choose selection method for this run (you can change)
        methods_to_try = ['percentile', 'elbow', 'information_theory', 'emc_inspired']
        if 'Churn' in combined.columns:
            methods_to_try.append('cross_validation')

        scaler_reports = {}
        arb_thresh = 0.01   # 👈 you can pick 0.01, 0.02, etc. for the lab
        selector = VarianceThreshold(threshold=arb_thresh)
        try:
            X_sel = selector.fit_transform(numeric_combined.to_numpy())
            mask = selector.get_support()
            names_num = numeric_combined.columns.tolist()
            kept = [n for n, m in zip(names_num, mask) if m]
            removed = [n for n, m in zip(names_num, mask) if not m]
            kept_count = len(kept)
            total = len(names_num)
            fraction_var_kept = var_series.loc[kept].sum() / var_series.sum() if var_series.sum() > 0 else np.nan
            scaler_reports['arbitrary'] = {
                'threshold': arb_thresh,
                'kept': kept,
                'removed': removed,
                'kept_count': kept_count,
                'total_numeric': total,
                'fraction_variance_kept': fraction_var_kept
            }
            print(f"Method=arbitrary        -> threshold={arb_thresh:.6g}, kept {kept_count}/{total}")
            # optionally save
            pruned_df = combined.loc[:, kept + ([c for c in other_cols if c == 'Churn'])]
            if output_dir:
                outname = f"{output_dir}/telco_pruned_{scaler_name}_arbitrary.csv"
                pruned_df.to_csv(outname, index=False)
        except Exception as e:
            scaler_reports['arbitrary'] = {'error': str(e)}
            print(f"Method=arbitrary        -> ERROR: {e}")

        methods = ['percentile', 'elbow', 'information_theory', 'emc_inspired', 'cross_validation']

        for method in methods:
            try:
                selector = AutomatedVarianceThreshold(method=method)
                # for cross_validation supply labels if present
                y = combined['Churn'].to_numpy() if 'Churn' in combined.columns and method == 'cross_validation' else None
                selector.fit(numeric_combined.to_numpy(), y)
                mask = selector.get_selected_features_mask()
                names_num = numeric_combined.columns.tolist()
                kept = [n for n, m in zip(names_num, mask) if m]
                removed = [n for n, m in zip(names_num, mask) if not m]
                kept_count = len(kept)
                total = len(names_num)
                fraction_var_kept = var_series.loc[kept].sum() / var_series.sum() if var_series.sum() > 0 else np.nan
                scaler_reports[method] = {
                    'threshold': selector.get_threshold(),
                    'kept': kept,
                    'removed': removed,
                    'kept_count': kept_count,
                    'total_numeric': total,
                    'fraction_variance_kept': fraction_var_kept
                }
                print(f"Method={method:17} -> threshold={selector.get_threshold():.6g}, kept {kept_count}/{total}")
                # optionally save pruned DataFrame
                pruned_df = combined.loc[:, kept + ([c for c in other_cols if c == 'Churn'])]
                if output_dir:
                    outname = f"{output_dir}/telco_pruned_{scaler_name}_{method}.csv"
                    pruned_df.to_csv(outname, index=False)
            except Exception as e:
                scaler_reports[method] = {'error': str(e)}
                print(f"Method={method:17} -> ERROR: {e}")
        overall_reports[scaler_name] = scaler_reports

    return overall_reports


# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Demonstration on synthetic data
    demo_results = demonstrate_automated_thresholding()

    # 2) Try run on Telco file if present in working directory
    #telco_path = "telco_customer_churn.csv"
    #print("\nRunning pipeline on telco_customer_churn.csv ...")
    #reports = run_pipeline_on_telco_file(telco_path, output_dir=".")
    #print("\nSaved pruned CSVs and produced reports.")
    #else:
        #print("\nNo telco_customer_churn.csv found in the working directory - skipping Telco run.")

    # 3) Try demo for TimeSeriesDataset using a small sample or electric_production.csv
    df_ts = pd.read_csv("electric_production.csv") # keep numeric columns
    numeric_df = df_ts.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    ds = TimeSeriesDataset(numeric_df, window_size=12, num_outputs=3, stride=1)
    print(f"TimeSeriesDataset length: {len(ds)}")
    #else:
       # print("\nNo electric_production.csv found - skipping time series demo.")

if __name__ == "__main__":
    main()