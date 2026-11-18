import argparse

from common.checkpoint import invalidate
from spatial.correspondance_grid_safran import build_correspondence, load_safran_grid
from spatial.grid_signal_aggregation import (
    _root, build_ndvi_match, build_grid_signal, fill_empty_cells, neighbour_edges)


def run_gridding(site, years, fill=True, redo_from=None):
    # precondition: preprocessing must exist for every year
    missing = [str(p) for y in years for p in (
        _root(site, y) / "products" / "s1_clean.parquet",
        _root(site, y) / "products" / "s2_ndvi.parquet") if not p.exists()]
    if missing:
        raise FileNotFoundError("Run preprocessing first, missing:\n  " + "\n  ".join(missing))

    safran = load_safran_grid()
    edges = neighbour_edges(safran) if fill else None

    for y in years:
        print(f"===== {site} {y} =====")
        r = _root(site, y)
        invalidate({"correspondence": r/"interim"/"correspondence.parquet",
                    "ndvi_match":     r/"interim"/"ndvi_match.parquet",
                    "grid_signal":    r/"products"/"grid_signal.parquet",
                    "grid_filled":    r/"products"/"grid_signal_filled.parquet"}, redo_from)
        build_correspondence(site, y, safran)
        build_ndvi_match(site, y)
        build_grid_signal(site, y)
        if fill:
            fill_empty_cells(site, y, edges)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--years", required=True, type=int, nargs="+")
    ap.add_argument("--no-fill", action="store_true")
    ap.add_argument("--redo-from", default=None,
                    choices=["correspondence", "ndvi_match", "grid_signal", "grid_filled"])
    a = ap.parse_args()
    run_gridding(a.site, a.years, fill=not a.no_fill, redo_from=a.redo_from)