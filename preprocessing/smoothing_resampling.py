"""
smoothing_resampling.py
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve


def _to_daily_grid(dates: pd.DatetimeIndex, values: np.ndarray):
    """Interpolate irregular obs onto a daily grid. Returns (daily_index, interpolated_values, weights).
    weights = 1 at observed dates, 0 at gap-filled ones (for weighted smoothers)."""
    daily_index = pd.date_range(dates.min(), dates.max(), freq="D")
    s = pd.Series(values, index=dates).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    daily = s.reindex(daily_index)
    weights = daily.notna().astype(float).values
    daily_interp = daily.interpolate(method="linear", limit_direction="both").values
    return daily_index, daily_interp, weights


def gaussian_smooth(dates: pd.DatetimeIndex, values: np.ndarray, sigma_days: float = 8.0) -> np.ndarray:
    """Time-weighted Gaussian kernel smoothing, directly on irregular dates — no gridding needed."""
    t = np.array([(d - dates[0]).days for d in dates], dtype=float)
    out = np.empty(len(t))
    for i, ti in enumerate(t):
        w = np.exp(-0.5 * ((t - ti) / sigma_days) ** 2)
        out[i] = np.sum(w * values) / np.sum(w)
    return out


def harmonic_smooth(dates: pd.DatetimeIndex, values: np.ndarray, n_harmonics: int = 3) -> np.ndarray:
    """Fit a Fourier/harmonic regression with period tied to the data's own span,
    and evaluate back at the original dates. Returns NaN-filled array if too few points."""
    t = np.array([(d - dates[0]).days for d in dates], dtype=float)

    if len(t) < 2 * n_harmonics + 2:
        return np.full_like(values, np.nan, dtype=float)

    omega = 1.0 / (t[-1] - t[0])

    def _design(tt):
        tt = np.asarray(tt)
        cols = [np.ones_like(tt)]
        for k in range(1, n_harmonics + 1):
            cols += [np.cos(2 * np.pi * k * omega * tt), np.sin(2 * np.pi * k * omega * tt)]
        return np.column_stack(cols)

    try:
        theta, *_ = np.linalg.lstsq(_design(t), values, rcond=None)
        return _design(t) @ theta
    except Exception:
        return np.full_like(values, np.nan, dtype=float)


# probably best with window 45ish and poly_order 3
def savgol_smooth(dates: pd.DatetimeIndex, values: np.ndarray, window_days: int = 10,
                   polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay needs uniform spacing, so: interpolate to daily grid, filter, sample back."""
    daily_index, daily_vals, _ = _to_daily_grid(dates, values)
    window = window_days if window_days % 2 == 1 else window_days + 1
    window = min(window, len(daily_vals) - (1 - len(daily_vals) % 2))
    if window < polyorder + 2:
        return values  # series too short to smooth meaningfully — return as-is
    smoothed_daily = savgol_filter(daily_vals, window_length=window, polyorder=polyorder)
    return pd.Series(smoothed_daily, index=daily_index).reindex(dates).values


def whittaker_smooth(dates: pd.DatetimeIndex, values: np.ndarray, lam: float = 10.0) -> np.ndarray:
    """Weighted Whittaker smoother on a daily grid (weight=1 observed, 0 gap-filled), sample back."""
    daily_index, daily_vals, weights = _to_daily_grid(dates, values)
    n = len(daily_vals)
    W = diags(weights)
    D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n), dtype=float)
    A = csc_matrix(W + lam * D.T @ D)
    b = weights * daily_vals
    smoothed_daily = spsolve(A, b)
    return pd.Series(smoothed_daily, index=daily_index).reindex(dates).values


SMOOTHERS = {
    "whittaker": whittaker_smooth,
    "savgol": savgol_smooth,
    "gaussian": gaussian_smooth,
    "harmonic": harmonic_smooth,
}

def smooth_parcel_bands(df: pd.DataFrame, method: str, cols: list[str],
                        date_col: str = "date", **kwargs) -> pd.DataFrame:
    """Smooth each band in `cols` independently for ONE parcel.
    df must already be filtered to a single id_parcel. Bands are overwritten in place."""
    if method not in SMOOTHERS:
        raise ValueError(f"Unknown method '{method}' — choose from {list(SMOOTHERS)}")

    out = df.reset_index(drop=True).copy()
    dates_all = pd.to_datetime(out[date_col])

    for c in cols:
        valid = out[c].notna()
        if valid.sum() < 2:                  # nothing usable to smooth
            continue                         # band left as-is
        sub_dates = pd.DatetimeIndex(dates_all[valid])
        sub_values = out.loc[valid, c].to_numpy(dtype=float)
        out.loc[valid, c] = SMOOTHERS[method](sub_dates, sub_values, **kwargs)

    return out


# -- Resampling ---------------------------------------------------------------

def resample_Xday(df: pd.DataFrame, cols: list[str], start: str, end: str,
                  date_col: str = "date", days: int = 10) -> pd.DataFrame:
    """Resample one parcel's smoothed bands onto a regular X-day grid shared by all parcels."""
    s = df[cols].astype(float).set_index(pd.to_datetime(df[date_col])).sort_index()
    s = s[~s.index.duplicated(keep="first")]

    grid = pd.date_range(start, end, freq=f"{days}D")
    combined = s.reindex(s.index.union(grid)).interpolate(method="time", limit_area="inside")

    return combined.reindex(grid).rename_axis(date_col).reset_index()


def pivot_to_wide(df: pd.DataFrame, cols: list[str], id_col: str = "id_parcel",
                  date_col: str = "date") -> pd.DataFrame:
    """Long (id_parcel, date, bands...) -> wide: one row per parcel, one column per band x timestep."""
    wide = df.pivot(index=id_col, columns=date_col, values=cols)
    wide.columns = [f"{band}_{pd.Timestamp(d):%Y-%m-%d}" for band, d in wide.columns]
    return wide.reset_index()