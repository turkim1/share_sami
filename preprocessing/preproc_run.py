# preprocessing/preproc_run.py
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from config import PART_FILE_RE, SMOOTHING_METHOD, SMOOTHING_PARAMS, YearPaths, season_bounds, season_months
from common.checkpoint import step, invalidate
from preprocessing.misc import deduplicate_observations, to_db, add_vh_vv_ratio, filter_radar_, filter_msk_cldprb, filter_ndvi_, compute_ndvi
from preprocessing.smoothing_resampling import (
    smooth_parcel_bands, resample_Xday,
)

S2_BANDS = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]   # MSK_CLDPRB is dropped after filtering


def _check_parts(files, year):
    by_month, totals = defaultdict(set), set()
    for f in files:
        m = PART_FILE_RE.match(f.name)
        if not m:
            print(f"[warn] unrecognised file name: {f.name}")
            continue
        by_month[m["month"]].add(int(m["idx"]))
        if m["n"]:
            totals.add(int(m["n"]))

    if len(totals) > 1:
        print(f"[warn] files from different tilings mixed (of-N = {sorted(totals)}); "
              "duplicates are collapsed by the dedup step, but check the Drive folder")

    expected = {f"{cy}-{mo:02d}" for cy, mo, _, _ in season_months(year)}
    missing = sorted(expected - by_month.keys())
    if missing:
        print(f"[warn] {year}: no files for months {missing}")

    if by_month:
        usual = Counter(frozenset(v) for v in by_month.values()).most_common(1)[0][0]
        for month, parts in sorted(by_month.items()):
            if frozenset(parts) != usual:
                print(f"[warn] {month}: tiles {sorted(parts)} differ from the usual {sorted(usual)}")


def _load_parts(raw_dir: Path, year: int) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {raw_dir} — download them from Drive first.")
    _check_parts(files, year)
    df = pd.concat(
        (pd.read_csv(f, dtype={"id_parcel": str, "orbit_number": str}) for f in files),
        ignore_index=True,
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def _per_parcel(df: pd.DataFrame, fn, **kw) -> pd.DataFrame:
    return pd.concat(
        (fn(g, **kw).assign(id_parcel=pid) for pid, g in df.groupby("id_parcel")),
        ignore_index=True,
    )


def run_s1(site: str, year: int, redo_from: str | None = None, threshold: float = 5.0):
    print(f"===== S1 {site} {year} =====")
    yp = YearPaths(site, year)
    p = {"merged": yp.s1_merged, "clean": yp.s1_clean}
    invalidate(p, redo_from)
    df = step(p["merged"], lambda: deduplicate_observations(
        _load_parts(yp.raw_s1, year), orbit_col="orbit_direction"))
    return step(p["clean"], lambda: filter_radar_(
        add_vh_vv_ratio(to_db(df)), threshold=threshold))


def run_s2(site: str, year: int, redo_from: str | None = None,
           cloud_thr: float = 1, ndvi_thr: float = 0.4,
           days: int = 10, smooth: bool = True):
    """smooth=False -> stop at s2_ndvi. smooth=True -> also smooth, then resample."""
    print(f"===== S2 {site} {year} =====")
    yp = YearPaths(site, year, days=days)
    start, end = season_bounds(year)
    p = {"merged": yp.s2_merged, "filtered": yp.s2_filtered, "ndvi": yp.s2_ndvi,
         "smoothed": yp.s2_ndvi_smoothed, "resampled": yp.s2_ndvi_resampled}
    invalidate(p, redo_from)

    df = step(p["merged"], lambda: deduplicate_observations(_load_parts(yp.raw_s2, year)))
    df = step(p["filtered"], lambda: filter_ndvi_(filter_msk_cldprb(df, cloud_thr), ndvi_thr))
    nd = step(p["ndvi"], lambda: compute_ndvi(df))
    if not smooth:
        return nd

    nd = step(p["smoothed"], lambda: _per_parcel(
        nd, smooth_parcel_bands, method=SMOOTHING_METHOD,
        cols=["ndvi"], **SMOOTHING_PARAMS[SMOOTHING_METHOD]))
    return step(p["resampled"], lambda: _per_parcel(
        nd, resample_Xday, cols=["ndvi"], start=start, end=end, days=days))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--years", required=True, type=int, nargs="+")
    ap.add_argument("--no-smooth", action="store_true")
    a = ap.parse_args()
    for y in a.years:
        run_s1(a.site, y)
        run_s2(a.site, y, smooth=not a.no_smooth)