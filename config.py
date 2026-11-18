# carto_irrig/config.py
from pathlib import Path
import calendar
from datetime import date, timedelta
import re


# GEE tiling: parcels are split into GRID_COLS x GRID_ROWS tiles per month 
GRID_COLS = 2    # west -> east
GRID_ROWS = 2    # north -> south

def part_description(sensor, site, year, month_str, idx, n_parts) -> str:
    """Single definition of the export name; PART_FILE_RE below parses it back."""
    return f"{sensor}_{site}_{year}_{month_str}_part{idx}of{n_parts}"


PART_FILE_RE = re.compile(
    r"^(?P<sensor>S[12])_(?P<site>.+)_(?P<year>\d{4})_(?P<month>\d{4}-\d{2})"
    r"_part(?P<idx>\d+)(?:of(?P<n>\d+))?\.csv$"
)


# Data directories
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent                # ...\Carto_irrigation
DATA_ROOT    = PROJECT_ROOT / "data"

# Parcels: <PARCELS_ROOT>/region_<site>/rpg_[<site>_]<yr>/rpg_<yr>_<site>.shp
PARCELS_ROOT     = DATA_ROOT / "parcels" / "raw"
PARCELS_FILTERED = DATA_ROOT / "parcels" / "filtered"
AOI_DIR          = DATA_ROOT / "aoi"

# Satellite: <SATELLITE_ROOT>/S1_Exports_<site>, S2_BANDS_Exports_<site>
SATELLITE_ROOT   = DATA_ROOT / "satellite"

# Per-site outputs: <SITES_ROOT>/<site>/<year>/{correspondance,ndvi,s1_grid,safran_data}
SITES_ROOT       = DATA_ROOT / "sites_output"

# SAFRAN
SAFRAN_DIR         = PACKAGE_ROOT / "meteo_france_safran" / "mailles_safran"
SAFRAN_GRID_PATH   = SAFRAN_DIR / "safran_polygones.gpkg"
SAFRAN_POINTS_PATH = SAFRAN_DIR / "SHP_SIM_FRANCE.shp"

# Run params
SITES      = ["LaRochelle", "Strasbourg"]      # sites with satellite data on disk (FOR NOW TESTING ONLY)
CROP_YEARS = [2020, 2021, 2022, 2023, 2024]
EE_PROJECT = "watsim"

# seasons
SEASON_MONTHS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # must stay in calendar order

# Smoothng parameters
SMOOTHING_METHOD = "savgol"  # "savgol", "whittaker", "gaussian" or "harmonic"
SMOOTHING_PARAMS = {
    "savgol":    {"window_days": 11, "polyorder": 2},   # was window_length, which savgol_smooth doesn't accept
    "whittaker": {"lam": 1e5},
    "gaussian": {"sigma_days": 8},
    "harmonic": {"n_harmonics": 3},
}
SMOOTH_TAG = "_".join([SMOOTHING_METHOD] + [f"{k}{v}" for k, v in sorted(SMOOTHING_PARAMS[SMOOTHING_METHOD].items())])

# Path management
class YearPaths:
    def __init__(self, site: str, year: int, days: int = 10):
        self.site, self.year = site, year
        self.root     = SITES_ROOT / site / str(year)
        interim, prod = self.root / "interim", self.root / "products"

        # inputs
        self.raw_s1 = SATELLITE_ROOT / site / str(year) / "S1"
        self.raw_s2 = SATELLITE_ROOT / site / str(year) / "S2"
        self.filtered_shp = PARCELS_FILTERED / site / f"rpg_{year}" / f"rpg_{year}_{site}.shp"

        # interim 
        self.s1_merged      = interim / "s1_merged.parquet"
        self.s2_merged      = interim / "s2_merged.parquet"
        self.correspondence = interim / "correspondence.parquet"
        self.ndvi_match     = interim / "ndvi_match.parquet"

        # products
        self.s1_clean           = prod / "s1_clean.parquet"
        self.s2_filtered        = prod / "s2_filtered.parquet"
        self.s2_ndvi            = prod / "s2_ndvi.parquet"
        self.s2_ndvi_smoothed   = prod / f"s2_ndvi_smoothed_{SMOOTH_TAG}.parquet"
        self.s2_ndvi_resampled  = prod / f"s2_ndvi_resampled_{SMOOTH_TAG}_{days}d.parquet"
        self.grid_signal        = prod / "grid_signal.parquet"
        self.grid_signal_filled = prod / "grid_signal_filled.parquet"
        self.winter_filter = prod / f"winter_filter_{SMOOTH_TAG}_{days}d.parquet"
        self.diag          = self.root / "diagnostics" / f"winter_filter_{SMOOTH_TAG}_{days}d"


# Winter-crop filter (k-means on RD / TA) 
WINTER_FILTER = {
    "period_start":    (4, 1),             # NDVI used from this (month, day)
    "window":          ((6, 1), (10, 30)), # RD/TA averaged here, end excluded
    "min_ndvi":        0.05,               # values below are masked
    "anchor_quantile": 0.95,
    "seed":            5,
    "winter_clusters": [2],                # clusters flagged is_winter
}

def _month_bounds(year: int, month: int) -> tuple[str, str]:
    first = date(year, month, 1)
    last  = date(year, month, calendar.monthrange(year, month)[1])
    return first.isoformat(), (last + timedelta(days=1)).isoformat()


def season_months(crop_year: int) -> list[tuple[int, int, str, str]]:
    result   = []
    wraps    = SEASON_MONTHS[0] > SEASON_MONTHS[-1]
    cal_year = crop_year - 1 if wraps else crop_year
    prev_m   = None
    for m in SEASON_MONTHS:
        if prev_m is not None and m < prev_m:
            cal_year += 1
        start, end = _month_bounds(cal_year, m)
        result.append((cal_year, m, start, end))
        prev_m = m
    return result


def season_bounds(crop_year: int) -> tuple[str, str]:
    """First and last day (inclusive) of the season, e.g. 2022 -> ('2022-03-01', '2022-12-31')."""
    months = season_months(crop_year)
    start = months[0][2]
    end = (date.fromisoformat(months[-1][3]) - timedelta(days=1)).isoformat()
    return start, end
