# common/geo_utils.py
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import mapping
from shapely.ops import unary_union
from shapely.validation import make_valid

WGS84 = "EPSG:4326"


def load_aoi(filepath: str) -> tuple[str, str]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"AOI file not found: {filepath}")

    gdf = gpd.read_file(filepath)
    print(f"[AOI] {len(gdf)} feature(s), CRS = {gdf.crs}")

    if gdf.crs is None:
        raise ValueError("AOI has no CRS defined.")
    if gdf.crs.to_epsg() != 4326:
        print(f"[AOI] Reprojecting {gdf.crs} --> {WGS84}")
        gdf = gdf.to_crs(WGS84)

    invalid = gdf[~gdf.geometry.is_valid]
    if not invalid.empty:
        print(f"[AOI] Repairing {len(invalid)} invalid geometry(ies) ...")
        gdf["geometry"] = gdf["geometry"].apply(make_valid)

    union = unary_union(gdf.geometry)
    return json.dumps(mapping(union), separators=(",", ":")), path.stem