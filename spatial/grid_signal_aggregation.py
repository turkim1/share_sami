"""
Per-year bare-soil S1 signal per SAFRAN cell.

Reads  (from preprocessing): products/s1_clean.parquet, products/s2_ndvi.parquet
Writes (in data/{site}/{year}/):
    interim/ndvi_match.parquet       slow step, thresholds not applied
    products/grid_signal.parquet     cell's own signal only
    products/grid_signal_filled.parquet   own + neighbour-filled (optional)

A (parcel, S1 date) pair counts as bare soil if:
  A) always bare: seasonal NDVI quantile < ALWAYS_BARE_NDVI (all its dates kept), or
  B) bare today:  NDVI of nearest clear S2 image (within tolerance) < BARE_ON_DATE_NDVI.
"""
import geopandas as gpd
import pandas as pd

from config import SITES_ROOT
from common.checkpoint import step
from spatial.correspondance_grid_safran import build_correspondence

# --- parameters ---
NDVI_TOLERANCE_DAYS = 10
BARE_SEASON_MONTHS = (4, 10)      # not the extraction season in config
SEASON_QUANTILE = 0.99
MIN_S2_OBS = 5
ALWAYS_BARE_NDVI = 0.35
BARE_ON_DATE_NDVI = 0.25


def _root(site, year):
    return SITES_ROOT / site / str(year)


# -- Step 1: NDVI match (slow) ------------------------------------------------
def build_ndvi_match(site, year, tolerance_days=NDVI_TOLERANCE_DAYS):
    root = _root(site, year)

    def compute():
        s1 = pd.read_parquet(root / "products" / "s1_clean.parquet")
        s2 = pd.read_parquet(root / "products" / "s2_ndvi.parquet")

        in_season = s2["date"].dt.month.between(*BARE_SEASON_MONTHS)
        g = s2[in_season].groupby("id_parcel")["ndvi"]
        stats = pd.DataFrame({"ndvi_q": g.quantile(SEASON_QUANTILE),
                              "n_s2_obs": g.size()}).reset_index()

        pairs = (s1[["id_parcel", "date"]].drop_duplicates()
                 .rename(columns={"date": "s1_date"}).sort_values("s1_date"))
        s2 = s2.rename(columns={"date": "s2_date"}).sort_values("s2_date")
        matched = pd.merge_asof(pairs, s2, left_on="s1_date", right_on="s2_date",
                                by="id_parcel", direction="nearest",
                                tolerance=pd.Timedelta(days=tolerance_days))
        print(f"    [{year}] {matched['ndvi'].isna().mean():.1%} of S1 pairs "
              f"have no clear S2 within {tolerance_days} d")

        corr = build_correspondence(site, year)
        return (matched.merge(stats, on="id_parcel", how="left")
                       .merge(corr, on="id_parcel", how="left"))

    return step(root / "interim" / "ndvi_match.parquet", compute)


# -- Step 2: eligibility (instant, thresholds applied here) ---------------------
def select_eligible(m, always=ALWAYS_BARE_NDVI, on_date=BARE_ON_DATE_NDVI, min_obs=MIN_S2_OBS):
    always_bare = (m["ndvi_q"] < always) & (m["n_s2_obs"] >= min_obs)
    keep = always_bare | (m["ndvi"] < on_date)
    out = m.loc[keep, ["id_parcel", "s1_date", "cell"]].copy()
    out["is_always_bare"] = always_bare[keep].to_numpy()
    print(f"    Eligible pairs: {keep.sum()}/{len(m)} ({keep.mean():.1%})")
    return out


# -- Step 3: aggregate to cell --------------------------------------------------
def build_grid_signal(site, year):
    root = _root(site, year)

    def compute():
        elig = select_eligible(pd.read_parquet(root / "interim" / "ndvi_match.parquet"))
        s1 = pd.read_parquet(root / "products" / "s1_clean.parquet")
        merged = s1.merge(elig, left_on=["id_parcel", "date"],
                          right_on=["id_parcel", "s1_date"], how="inner")
        agg = (merged.groupby(["cell", "date", "orbit_number", "orbit_direction"], as_index=False)
               .agg(VV_mean=("VV", "mean"), 
                    VH_mean=("VH", "mean"),
                    angle_mean=("angle", "mean"),
                    n_parcels=("id_parcel", "nunique"),
                    n_parcels_always_bare=("is_always_bare", "sum")))
        agg["n_parcels_bare_on_date"] = agg["n_parcels"] - agg["n_parcels_always_bare"]
        agg["year"] = year
        return agg

    return step(root / "products" / "grid_signal.parquet", compute)


# -- Step 4: optional neighbour fill -------------------------------------------
def neighbour_edges(safran) -> pd.DataFrame:
    adj = gpd.sjoin(safran, safran, predicate="touches")
    return (adj[["cell_left", "cell_right"]]
            .rename(columns={"cell_left": "cell", "cell_right": "neigh"})
            .reset_index(drop=True))


def fill_empty_cells(site, year, edges):
    root = _root(site, year)

    def compute():
        own = pd.read_parquet(root / "products" / "grid_signal.parquet")
        corr = pd.read_parquet(root / "interim" / "correspondence.parquet")
        value_cols = ["VV_mean", "VH_mean", "angle_mean"]
        key = ["cell", "date", "orbit_number"]

        cells = pd.DataFrame({"cell": corr["cell"].dropna().unique()})
        acq = own[["date", "orbit_number", "orbit_direction"]].drop_duplicates()
        targets = cells.merge(acq, how="cross")

        missing = targets.merge(own[key].assign(has_own=True), on=key, how="left")
        missing = missing[missing["has_own"].isna()].drop(columns="has_own")

        nb = (missing.merge(edges, on="cell")
                     .merge(own[key + value_cols].rename(columns={"cell": "neigh"}),
                            on=["neigh", "date", "orbit_number"]))
        filled = (nb.groupby(key + ["orbit_direction"], as_index=False)
                    .agg(**{c: (c, "mean") for c in value_cols},
                         n_neighbours=("neigh", "nunique")))
        filled["source"] = "neighbours"
        filled["year"] = year

        out = pd.concat([own.assign(source="own", n_neighbours=0), filled],
                        ignore_index=True).sort_values(key)
        print(f"    [{year}] own rows: {len(own)} | filled rows: {len(filled)}")
        return out

    return step(root / "products" / "grid_signal_filled.parquet", compute)


# -- Convenience: load several years --------------------------------------------
def load_grid_signal(site, years, filled=False):
    name = "grid_signal_filled.parquet" if filled else "grid_signal.parquet"
    return pd.concat((pd.read_parquet(_root(site, y) / "products" / name) for y in years),
                     ignore_index=True)