# common/io.py
import requests
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import geopandas as gpd
from shapely.validation import make_valid


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def out_file(out_dir: Path, year: int) -> Path:
    site = out_dir.name
    return out_dir / f"rpg_{year}" / f"rpg_{year}_{site}.shp"


def year_already_fetched(out_dir: Path, year: int) -> bool:
    return out_file(out_dir, year).exists()


def save_year(gdf: gpd.GeoDataFrame, year: int, out_dir: Path):
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(
        lambda geom: make_valid(geom).buffer(0) if geom is not None else geom
    )
    gdf.columns = [
        c if c == "geometry" else c.replace(" ", "_").replace(".", "_")[:10]
        for c in gdf.columns
    ]
    fp = out_file(out_dir, year)
    fp.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(fp, driver="ESRI Shapefile")
    size_mb = fp.stat().st_size / 1_048_576
    print(f"[save] rpg_{year} — {len(gdf):,} parcels --> {fp.parent}  ({size_mb:.1f} MB)")