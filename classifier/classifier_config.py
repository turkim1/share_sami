# carto_irrig/classifier/classifier_config.py
"""All classifier constants. Edit here only."""

# -- classes ---------------------------------------------------------------
CLASS_DEFS_ALL = {                    # order = class id
    "Maize":     {"MIS", "MIE", "MID"},
    "Soybean":   {"SOJ"},
    "Sunflower": {"TRN"},
}
OTHER_NAME = "Other vegetation"
MIN_PARCELS = 100                     # class with fewer parcels in ANY training year -> merged into Other (0 = keep all)
EXCLUDED_CODE_GROUPS = {20, 21, 28, 11}   # vines, orchards, misc, fallow (training years only)

# -- features --------------------------------------------------------------
MONTH_ORDER = ["m04", "m05", "m06", "m07", "m08", "m09", "m10", "m11"]
MONTH_NUMS = [int(m[1:]) for m in MONTH_ORDER]
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S1_FEATURES = ["VV", "VH", "VV_VH_ratio"]   # VV_VH_ratio computed here from normalised VV / VH
ORBIT_DIR = {"asc": "ASCENDING", "desc": "DESCENDING"}

# mode -> (uses_s1, uses_s2, orbit tags; None = all orbits pooled)
MODES = {
    "S1":               (True,  False, [None]),
    "S2":               (False, True,  [None]),
    "both":             (True,  True,  [None]),
    "S1_ASC":           (True,  False, ["asc"]),
    "S1_DESC":          (True,  False, ["desc"]),
    "S1_SPLIT_ORBIT":   (True,  False, ["asc", "desc"]),
    "both_split_orbit": (True,  True,  ["asc", "desc"]),
}

# -- model -----------------------------------------------------------------
RANDOM_SEED = 5
POWER_ALPHA = 0.07
XGB_PARAMS = {
    "n_estimators": 400, "max_depth": 5, "learning_rate": 0.1894,
    "subsample": 1.0, "colsample_bytree": 0.8521, "min_child_weight": 1.793,
    "gamma": 0.091, "reg_alpha": 3.6117, "reg_lambda": 4.7272,
    "objective": "multi:softprob", "tree_method": "hist",
    "random_state": RANDOM_SEED,
}

# -- NDVI post-filter (splits "Other vegetation") --------------------------
NDVI_POSTFILTER = True
NDVI_FILTER_START_MONTH = 3
NDVI_FILTER_END_MONTH = 8
NDVI_FILTER_QUANTILE = 0.90     # robust "peak NDVI" of the parcel in the window
NDVI_FILTER_THRESHOLD = 0.45