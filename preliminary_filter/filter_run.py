"""
preliminary_filter/filter_run.py

Reads   products/s2_ndvi_resampled_{SMOOTH_TAG}_{days}d.parquet   (parcel list + NDVI)
        interim/s2_merged.parquet                                  (labels, optional, scoring only)
Writes  products/winter_filter_{SMOOTH_TAG}_{days}d.parquet
        diagnostics/winter_filter_{SMOOTH_TAG}_{days}d/*

Label-free: every parcel with NDVI is clustered. Labels, if present, are attached
for diagnostics only and never influence the clustering.

columns: id_parcel, [code_cultu, code_group], RD, TA, RD_std, TA_std, cluster, status, is_winter, year
  status: "clustered" | "no_ndvi"
"""
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import YearPaths, WINTER_FILTER as P
from common.checkpoint import step, invalidate
from preliminary_filter.kmeans_filter import to_wide, compute_rd_ta, classify
from preliminary_filter.diagnostics import write_diagnostics

LABEL_COLS = ["code_cultu", "code_group"]


def _load_parcels(yp, ndvi) -> pd.DataFrame:
    """Parcel universe = parcels in the NDVI file. Labels are attached if available (scoring only)."""
    parcels = pd.DataFrame({"id_parcel": ndvi["id_parcel"].unique()})
    have = []
    if yp.s2_merged.exists():
        have = [c for c in LABEL_COLS if c in pq.read_schema(yp.s2_merged).names]
    if have:
        lab = (pd.read_parquet(yp.s2_merged, columns=["id_parcel"] + have)
                 .drop_duplicates("id_parcel"))
        lab["id_parcel"] = lab["id_parcel"].astype(str)
        parcels = parcels.merge(lab, on="id_parcel", how="left")
    return parcels


def run_winter_filter(site: str, year: int, days: int = 15, redo: bool = False) -> pd.DataFrame:
    print(f"===== winter filter {site} {year} =====")
    yp = YearPaths(site, year, days=days)
    if not yp.s2_ndvi_resampled.exists():
        raise FileNotFoundError(f"{yp.s2_ndvi_resampled} — run S2 preprocessing with smooth=True first.")

    invalidate({"winter_filter": yp.winter_filter}, "winter_filter" if redo else None)

    def compute():
        ndvi = pd.read_parquet(yp.s2_ndvi_resampled)
        ndvi["id_parcel"] = ndvi["id_parcel"].astype(str)
        parcels = _load_parcels(yp, ndvi)

        def ts(md):
            return pd.Timestamp(year, *md)
        rd_ta = compute_rd_ta(to_wide(ndvi), ts(P["period_start"]),
                              ts(P["window"][0]), ts(P["window"][1]), P["min_ndvi"])
        res = classify(rd_ta, q=P["anchor_quantile"], seed=P["seed"])

        out = parcels.merge(res.reset_index(), on="id_parcel", how="left")
        out["status"] = np.where(out["cluster"].isna(), "no_ndvi", "clustered")
        out["cluster"] = out["cluster"].astype("Int64")
        out["is_winter"] = out["cluster"].isin(P["winter_clusters"]).fillna(False).astype(bool)
        out["year"] = year

        print(f"    {out['status'].value_counts().to_dict()}")
        print(f"    cluster sizes: {out['cluster'].value_counts().sort_index().to_dict()}")
        print(f"    flagged: {out['is_winter'].sum()}")
        return out

    flags = step(yp.winter_filter, compute)
    write_diagnostics(flags, yp.diag, f"Winter filter {site} {year}")
    return flags


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--years", required=True, type=int, nargs="+")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    for y in a.years:
        run_winter_filter(a.site, y, a.days, a.redo)