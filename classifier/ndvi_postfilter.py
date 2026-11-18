# carto_irrig/classifier/ndvi_postfilter.py
"""Split 'Other vegetation' into cultivated / non_cultivated from the predict-year NDVI."""
import numpy as np
import pandas as pd
from classifier import classifier_config as C


def build_ndvi_table(s2_path) -> pd.DataFrame:
    s2 = pd.read_parquet(s2_path, columns=["id_parcel", "date", "B4", "B8"])
    s2 = s2[s2["date"].dt.month.between(C.NDVI_FILTER_START_MONTH, C.NDVI_FILTER_END_MONTH)]
    ndvi = (s2["B8"] - s2["B4"]) / (s2["B8"] + s2["B4"]).where((s2["B8"] + s2["B4"]) != 0)
    t = (ndvi.groupby(s2["id_parcel"]).quantile(C.NDVI_FILTER_QUANTILE)
         .rename("ndvi_stat").reset_index())
    t["id_parcel"] = t["id_parcel"].astype(str)
    return t


def add_veg_subclass(out: pd.DataFrame, ndvi: pd.DataFrame) -> pd.DataFrame:
    out = out.merge(ndvi, on="id_parcel", how="left")
    other = (out["classified"] & (out["pred_name"] == C.OTHER_NAME)).to_numpy()
    nd = out["ndvi_stat"]
    out["veg_subclass"] = np.select(
        [other & nd.ge(C.NDVI_FILTER_THRESHOLD).to_numpy(),
         other & nd.lt(C.NDVI_FILTER_THRESHOLD).to_numpy(),
         other],
        ["cultivated", "non_cultivated", "unknown"], default="")
    return out