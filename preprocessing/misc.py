"""
misc.py 
"""
import numpy as np
import pandas as pd

# -- Deduplicate parcels overlapping Sentinel tiles  ---------------------------------------------------------------
def deduplicate_observations(df, id_col="id_parcel", date_col="date",
                             orbit_col: str | None = None):
    """Collapse duplicate (parcel, date) rows caused by overlapping Sentinel tiles.
    Numeric columns are aggregated by median; non-numeric (categorical) columns keep the first value."""
    keys = [date_col, id_col] + ([orbit_col] if orbit_col else [])
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    other_cols = [c for c in df.columns if c not in numeric_cols + keys]
    agg = {c: "median" for c in numeric_cols if c not in keys}
    agg.update({c: "first" for c in other_cols})
    return df.groupby(keys, as_index=False).agg(agg)

# -- Flag spikes and dips in S1 & S2 ---------------------------------------------------------------

def _flag_spikes_dips(df: pd.DataFrame, col: str, threshold: float,
                       id_col: str = "id_parcel", mode: str = "both") -> pd.Series:
    """Boolean mask, True where a point is an isolated spike/dip relative to
    BOTH its immediate temporal neighbors (same direction, magnitude > threshold).
    mode: 'dip' flags only sudden drop-then-recovery; 'spike' flags only sudden
    rise-then-fall; 'both' flags either (default)."""
    grouped = df.groupby(id_col)[col]
    diff_prev = grouped.diff(1)
    diff_next = grouped.diff(-1)

    is_dip = (diff_prev < -threshold) & (diff_next < -threshold)
    is_spike = (diff_prev > threshold) & (diff_next > threshold)

    if mode == "dip":
        return is_dip
    elif mode == "spike":
        return is_spike
    elif mode == "both":
        return is_dip | is_spike
    else:
        raise ValueError(f"mode must be 'dip', 'spike', or 'both' - got '{mode}'")

# -- S1 ---------------------------------------------------------------1

def to_db(df: pd.DataFrame, cols: list[str] = ["VV", "VH"],
          suffix: str = "_db") -> pd.DataFrame:
    """Convert linear-scale radar backscatter columns to dB (10*log10).
    Adds new columns (col + suffix) rather than overwriting the linear originals,
    since downstream filtering (spikes/dips) may want either scale."""
    df = df.copy()
    for c in cols:
        df[c + suffix] = 10 * np.log10(df[c])
    return df


def add_vh_vv_ratio(df: pd.DataFrame, vv_col: str = "VV", vh_col: str = "VH",
                     ratio_col: str = "VH_VV_ratio") -> pd.DataFrame:
    """Add the VH/VV cross-polarization ratio, computed from linear-scale values.
    Kept independent from `to_db` (both derive from the same linear source)."""
    df = df.copy()
    df[ratio_col] = df[vh_col] / df[vv_col]
    return df

def filter_radar_(df: pd.DataFrame, threshold: float = 5.0,
                  id_col: str = "id_parcel", date_col: str = "date",
                  radar_col: str = "VV_db", orbit_col: str = "orbit_number") -> pd.DataFrame:
    """Drop rows where VV (in dB) is an isolated spike or dip within its (parcel, relative orbit) series.
    Run after to_db(). Whole rows are dropped, so VH and the ratio stay consistent."""
    df = df.sort_values([id_col, orbit_col, date_col]).copy()
    grp = df[id_col].astype(str) + "_" + df[orbit_col].astype(str)
    mask = _flag_spikes_dips(df.assign(_grp=grp), col=radar_col,
                             threshold=threshold, id_col="_grp", mode="both")
    return df.loc[~mask].reset_index(drop=True)




# -- S2 ---------------------------------------------------------------

def compute_ndvi(df: pd.DataFrame) -> pd.DataFrame:
    """id_parcel, date, ndvi from B8 and B4 (NaN where B8 + B4 == 0)."""
    s = df["B8"] + df["B4"]
    out = df[["id_parcel", "date"]].copy()
    out["ndvi"] = (df["B8"] - df["B4"]) / s.where(s != 0)
    return out.reset_index(drop=True)


def filter_msk_cldprb(df: pd.DataFrame, threshold: int = 1):
    """ Eliminate dates with cloud probability higher than a certain threshold"""
    return df[df['MSK_CLDPRB']< threshold ]


def filter_ndvi_(df: pd.DataFrame, threshold: float = 0.4,
                 id_col: str = "id_parcel", date_col: str = "date") -> pd.DataFrame:
    """Drop rows at isolated NDVI dips (residual cloud/shadow). NDVI is computed from B8 and B4
    only to flag the rows and is not kept. Spikes are not flagged (normal phenology)."""
    df = df.sort_values([id_col, date_col]).copy()
    ndvi = (df["B8"] - df["B4"]) / (df["B8"] + df["B4"])
    mask = _flag_spikes_dips(df.assign(NDVI=ndvi), col="NDVI",
                             threshold=threshold, id_col=id_col, mode="dip")
    return df.loc[~mask].reset_index(drop=True)


