"""correspondance_grid_safran.py
Parcel <-> SAFRAN cell correspondence, one table per year."""
import geopandas as gpd
import pandas as pd

from config import SITES_ROOT, PARCELS_FILTERED, SAFRAN_GRID_PATH
from common.checkpoint import step

SAFRAN_ID_COL = "cell"
ID_COL = "id_parcel"
TARGET_CRS = 2154


def load_safran_grid(path=SAFRAN_GRID_PATH, id_col=SAFRAN_ID_COL):
    safran = gpd.read_file(path)
    if safran.crs is None:
        raise ValueError(f"Missing CRS for SAFRAN grid: {path}")
    if safran.crs.to_epsg() != TARGET_CRS:
        safran = safran.to_crs(TARGET_CRS)
    return safran[[id_col, "geometry"]]


def build_correspondence(site: str, year: int, safran=None) -> pd.DataFrame:
    path = SITES_ROOT / site / str(year) / "interim" / "correspondence.parquet"

    def compute():
        shp = PARCELS_FILTERED / site / f"rpg_{year}" / f"rpg_{year}_{site}.shp"
        if not shp.exists():
            raise FileNotFoundError(f"{shp} — run the parcels step for {year} first.")
        grid = safran if safran is not None else load_safran_grid()
        parcels = gpd.read_file(shp).to_crs(TARGET_CRS)
        parcels = parcels.dissolve(by=ID_COL).reset_index()      # one geometry per parcel (merges multipart/repeated rows)
        pts = gpd.GeoDataFrame(parcels[[ID_COL]],
                            geometry=parcels.geometry.representative_point(), crs=parcels.crs)
        joined = gpd.sjoin(pts, grid, how="left", predicate="within")
        out = pd.DataFrame({"id_parcel": joined[ID_COL].astype(str),
                            "cell": joined[SAFRAN_ID_COL].astype("Int64")})
        n_miss = joined[SAFRAN_ID_COL].isna().sum()
        if n_miss:
            print(f"    [{year}] {n_miss}/{len(joined)} parcels outside any SAFRAN cell")
        return out

    return step(path, compute)