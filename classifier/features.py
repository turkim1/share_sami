# carto_irrig/classifier/features.py
"""Pure functions: S1/S2 parquet products -> one wide monthly feature row per parcel."""
import re
import numpy as np
import pandas as pd

from classifier import classifier_config as C

SENSOR_COL_RE = re.compile(r"^(s[12])_.+_(m\d{2})$")
BASE_RE = re.compile(r"^(s[12]_.+)_(m\d{2})$")
ID_COLS = ["id_parcel", "agri_year", "code_cultu", "code_group"]


# -- S1 angle normalisation ------------------------------------------------
def ref_angle(s1_paths) -> float:
    """Mean incidence angle over the TRAINING years' s1_clean files."""
    tot, n = 0.0, 0
    for p in s1_paths:
        a = pd.read_parquet(p, columns=["angle"])["angle"].dropna().to_numpy(float)
        tot += a.sum()
        n += a.size
    if n == 0:
        raise ValueError("No incidence angles found in the training S1 files.")
    ref = tot / n
    print(f"  [ref_angle] {ref:.2f} deg (from {len(s1_paths)} training year(s))")
    return ref


def normalise_s1(df: pd.DataFrame, ref: float) -> pd.DataFrame:
    """sigma_norm = sigma * cos^2(ref) / cos^2(angle), on linear VV and VH; then VV/VH ratio."""
    df = df.copy()
    f = np.cos(np.deg2rad(ref)) ** 2 / np.cos(np.deg2rad(df["angle"].to_numpy(float))) ** 2
    df["VV"] = df["VV"].to_numpy(float) * f
    df["VH"] = df["VH"].to_numpy(float) * f
    df["VV_VH_ratio"] = df["VV"] / df["VH"].where(df["VH"] > 0)
    return df


# -- monthly wide ----------------------------------------------------------
def _monthly_wide(df, cols, prefix, orbit=None) -> pd.DataFrame:
    """One median per (parcel, month). Columns like s1_VV_asc_median_m06 / s2_B4_median_m06."""
    month = df["date"].dt.month.rename("month")
    g = df.groupby(["id_parcel", month])[cols].median().unstack("month")
    mid = f"_{orbit}" if orbit else ""
    g.columns = [f"{prefix}_{c}{mid}_median_m{m:02d}" for c, m in g.columns]
    return g


def _meta(df) -> pd.DataFrame:
    df = df.copy()
    for c in ("code_cultu", "code_group"):
        if c not in df.columns:
            df[c] = np.nan
    return df.groupby("id_parcel")[["code_cultu", "code_group"]].first()


def year_features(s1_path, s2_path, year, mode, ref) -> pd.DataFrame:
    """Wide feature frame for one year: id_parcel, agri_year, code_cultu, code_group, features."""
    use_s1, use_s2, tags = C.MODES[mode]
    parts, metas = [], []

    if use_s2:
        s2 = pd.read_parquet(s2_path)
        s2 = s2[s2["date"].dt.month.isin(C.MONTH_NUMS)]
        parts.append(_monthly_wide(s2, C.S2_BANDS, "s2"))
        metas.append(_meta(s2))

    if use_s1:
        s1 = pd.read_parquet(s1_path)
        s1 = s1[s1["date"].dt.month.isin(C.MONTH_NUMS)]
        metas.append(_meta(s1))
        s1 = normalise_s1(s1, ref)
        for tag in tags:
            sub = s1 if tag is None else s1[s1["orbit_direction"].str.upper() == C.ORBIT_DIR[tag]]
            if sub.empty:
                print(f"  [warn] {year}: no S1 rows for orbit {tag}")
                continue
            parts.append(_monthly_wide(sub, C.S1_FEATURES, "s1", tag))

    if not parts:
        raise ValueError(f"{year}: no usable observations for mode {mode}")
    wide = pd.concat(parts, axis=1, join="outer")
    meta = pd.concat(metas).groupby(level=0).first()
    out = wide.join(meta, how="left").reset_index()
    out["id_parcel"] = out["id_parcel"].astype(str)
    out["agri_year"] = year
    print(f"  [features] {year}: {len(out):,} parcels x {len(wide.columns)} features")
    return out


def stack_years(frames) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)
    feats = sorted(c for c in df.columns if SENSOR_COL_RE.match(c))
    return df[ID_COLS + feats]


# -- window / interpolation (as in the original ml_preproc) ------------------
def feature_columns(df):
    return [c for c in df.columns if SENSOR_COL_RE.match(c)]


def drop_unavailable_months(df, predict_year):
    """Drop every sensor column that is entirely NaN in the predict year (all years)."""
    pred = df[df["agri_year"] == predict_year]
    if pred.empty:
        raise ValueError(f"No feature rows for predict year {predict_year}")
    empty = [c for c in feature_columns(df) if pred[c].isna().all()]
    for p in ("s1", "s2"):
        allc = [c for c in feature_columns(df) if c.startswith(p + "_")]
        if allc:
            kept = sorted({c[-3:] for c in allc if c not in empty})
            dropped = sorted({c[-3:] for c in allc if c in empty})
            print(f"  [window] {p.upper()} {predict_year}: using {kept or 'NO months'}"
                  + (f" | dropped: {dropped}" if dropped else ""))
    return df.drop(columns=empty)


def count_valid_months(df, cols, prefix) -> pd.Series:
    """Per parcel: months with at least one real value for sensor `prefix`. Call BEFORE interpolation."""
    by_month = {}
    for c in cols:
        m = SENSOR_COL_RE.match(c)
        if m and m.group(1) == prefix:
            by_month.setdefault(m.group(2), []).append(c)
    n = pd.Series(0, index=df.index)
    for mc in by_month.values():
        n += df[mc].notna().any(axis=1).astype(int)
    return n


def _interp_rows(a: np.ndarray) -> np.ndarray:
    n, m = a.shape
    idx = np.arange(m)
    valid = ~np.isnan(a)
    last = np.maximum.accumulate(np.where(valid, idx, -1), axis=1)
    nxt = np.minimum.accumulate(np.where(valid, idx, m)[:, ::-1], axis=1)[:, ::-1]
    rows = np.arange(n)[:, None]
    lo, hi = np.clip(last, 0, m - 1), np.clip(nxt, 0, m - 1)
    vlo, vhi = a[rows, lo], a[rows, hi]
    denom = hi - lo
    frac = np.where(denom > 0, (idx - lo) / np.where(denom > 0, denom, 1), 0.0)
    out = vlo + (vhi - vlo) * frac
    out = np.where(last == -1, vhi, out)
    out = np.where(nxt == m, vlo, out)
    return out


def interpolate_monthly(df) -> pd.DataFrame:
    """Row-wise linear interpolation across months, per (sensor, variable, orbit)."""
    df = df.copy()
    groups = {}
    for c in df.columns:
        m = BASE_RE.match(c)
        if m:
            groups.setdefault(m.group(1), []).append((C.MONTH_ORDER.index(m.group(2)), c))
    for base, items in groups.items():
        cols = [c for _, c in sorted(items)]
        df[cols] = _interp_rows(df[cols].to_numpy(dtype=float))
    return df


def window_label(cols, prefix) -> str:
    months = {c[-3:] for c in cols if c.startswith(prefix + "_")}
    ordered = [m for m in C.MONTH_ORDER if m in months]
    return f"{ordered[0]}-{ordered[-1]}" if ordered else ""


# -- classes -----------------------------------------------------------------
def build_classes(train: pd.DataFrame):
    """Class defs kept after MIN_PARCELS pruning. Returns (class_crops {id: codes}, names {id: name})."""
    years = sorted(train["agri_year"].unique())
    per_year = (train.groupby(["agri_year", "code_cultu"])["id_parcel"].nunique()
                .unstack(fill_value=0).reindex(years, fill_value=0))
    kept = {}
    for name, codes in C.CLASS_DEFS_ALL.items():
        counts = per_year.reindex(columns=list(codes), fill_value=0).sum(axis=1)
        ok = (not C.MIN_PARCELS) or counts.min() >= C.MIN_PARCELS
        print(f"  [classes] {name}: {counts.to_dict()} --> "
              + ("kept" if ok else f"merged into Other (< {C.MIN_PARCELS} in a year)"))
        if ok:
            kept[name] = codes
    if not kept:
        raise ValueError("MIN_PARCELS removed every class")
    crops = {i: codes for i, codes in enumerate(kept.values())}
    names = {**dict(enumerate(kept)), len(kept): C.OTHER_NAME}
    return crops, names


def assign_labels(codes: pd.Series, crops: dict) -> np.ndarray:
    lut = {code: i for i, cs in crops.items() for code in cs}
    other = len(crops)
    lab = codes.astype(object).map(lut)
    return pd.to_numeric(lab, errors="coerce").fillna(other).to_numpy(int)