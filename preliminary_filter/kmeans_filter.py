"""
preliminary_filter/kmeans_filter.py
Winter-crop detection: constrained k-means on (RD, TA). Pure functions, no I/O.
Cluster 0 = fixed summer anchor (high RD, high TA), cluster 2 = fixed winter anchor
(low RD, low TA), clusters 1 and 3 = free.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import kmeans_plusplus
from sklearn.preprocessing import StandardScaler


def to_wide(ndvi_long: pd.DataFrame, id_col="id_parcel", date_col="date") -> pd.DataFrame:
    """Long (id_parcel, date, ndvi) -> one row per parcel, one column per date."""
    wide = ndvi_long.pivot(index=id_col, columns=date_col, values="ndvi")
    wide.columns = pd.to_datetime(wide.columns)
    return wide.sort_index(axis=1)


def compute_rd_ta(wide: pd.DataFrame, period_start: pd.Timestamp,
                  win_start: pd.Timestamp, win_end: pd.Timestamp,
                  min_ndvi: float) -> pd.DataFrame:
    """
    RD: (ndvi - mean of all parcels on that date) / that mean, averaged over the window.
    TA: (ndvi - parcel's own mean over the period) / that mean, averaged over the window.
    Parcels with no valid value in the window are dropped.
    """
    w = wide.loc[:, wide.columns >= period_start]
    w = w.mask(w < min_ndvi)

    date_mean = w.mean()
    rd = w.sub(date_mean, axis=1).div(date_mean, axis=1)

    parcel_mean = w.mean(axis=1)
    ta = w.sub(parcel_mean, axis=0).div(parcel_mean, axis=0)

    in_win = (w.columns >= win_start) & (w.columns < win_end)
    out = pd.DataFrame({"RD": rd.loc[:, in_win].mean(axis=1),
                        "TA": ta.loc[:, in_win].mean(axis=1)})
    return out.dropna()


def constrained_kmeans(X: np.ndarray, q: float = 0.95, seed: int = 42,
                       max_iter: int = 300, tol: float = 1e-6):
    """4 clusters: 0 and 2 stay on the quantile corners, 1 and 3 move freely."""
    upper_right = np.quantile(X, q, axis=0)
    bottom_left = np.quantile(X, 1 - q, axis=0)
    seeds, _ = kmeans_plusplus(X, n_clusters=2, random_state=seed)

    centers = np.array([upper_right, seeds[0], bottom_left, seeds[1]], dtype=float)

    for it in range(max_iter):
        old = centers.copy()
        labels = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2).argmin(axis=1)
        centers[0], centers[2] = upper_right, bottom_left
        for k in (1, 3):
            if (labels == k).any():
                centers[k] = X[labels == k].mean(axis=0)
        if np.linalg.norm(centers - old) < tol:
            break
    else:
        print(f"    [kmeans] max iterations ({max_iter}) reached")

    labels = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2).argmin(axis=1)
    return labels, centers


def classify(rd_ta: pd.DataFrame, q: float, seed: int) -> pd.DataFrame:
    """Standardize RD and TA, cluster, return RD, TA, RD_std, TA_std, cluster per parcel."""
    X = StandardScaler().fit_transform(rd_ta[["RD", "TA"]].to_numpy())
    labels, _ = constrained_kmeans(X, q=q, seed=seed)
    out = rd_ta.copy()
    out["RD_std"], out["TA_std"] = X[:, 0], X[:, 1]
    out["cluster"] = labels
    return out