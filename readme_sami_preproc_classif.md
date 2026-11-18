# Crop classification pipeline: steps 3 to 6

Covers preprocessing, gridding, the kmeans winter filter and the summer-crop classification.

You can ignore all the `--redo` tags cuz they are remnants of my testing and will eventually be cleaned. For a first run you dont need them.
## Overview

```
raw CSVs ──► 3. preprocessing ──┬──► 4. gridding (SAFRAN cells)
(GEE export)   s1_clean         │      grid_signal[_filled].parquet
               s2_filtered      │
               s2_ndvi          └──► 5. kmeans winter filter ──► 6. classifier
               s2_ndvi_resampled       winter_filter_*.parquet    classification_*.parquet/csv
```

Run order is 3 → 4 → 5 → 6. Steps 4 and 5 both read step 3 outputs and don't depend on each other. The classifier (step 6) uses `s1_clean` and `s2_filtered` from step 3.

Everything is per `site` and `year`. Outputs go to `data/sites_output/{site}/{year}/{interim,products,diagnostics}/`. Paths and global parameters live in `config.py` (`YearPaths`). Smoothing and resampling use `days=10` everywhere, and the value appears in filenames.

**Train/predict logic.** The classifier trains on all labelled years before the predict year (`CROP_YEARS`, currently 2020 to 2024) and predicts the newest year. The predict year is always unlabelled, to simulate an operational context where the latest labels aren't available. Labels (`code_cultu`, `code_group`) come from the RPG parcel shapefile and are attached to the rows in the GEE export.

## Checkpoints

Every step uses `common.checkpoint.step`: if the output file exists it is reused, otherwise it is computed and saved.

- **Changing a parameter does not invalidate anything.** Delete the file or use `--redo-from` / `--redo`.
- `--redo-from X` deletes checkpoint X and every later one in that step.
- Classifier outputs are tagged with mode and training years, so different runs don't overwrite each other.

---

## Input folder structure

`data/` sits next to the package folder (`PROJECT_ROOT = package parent`). Paths are defined in `config.py`.

```
Carto_irrigation/
├── carto_irrig/                         # the package (config.py, classifier/, preprocessing/, ...)
│   └── meteo_france_safran/
│       └── mailles_safran/
│           └── safran_polygones.gpkg    # SAFRAN cells, id column "cell" (step 4)
└── data/
    ├── parcels/
    │   └── filtered/
    │       └── {site}/
    │           └── rpg_{year}/
    │               └── rpg_{year}_{site}.shp   # filtered RPG, id_parcel column (step 4)
    ├── satellite/
    │   └── {site}/
    │       └── {year}/
    │           ├── S1/   # S1_{site}_{year}_{YYYY-MM}_part{i}of{n}.csv
    │           └── S2/   # S2_{site}_{year}_{YYYY-MM}_part{i}of{n}.csv
    └── sites_output/ 
```

Example: `data/satellite/Strasbourg/2023/S1/S1_Strasbourg_2023_2023-05_part1of4.csv`

**Rules**
- One folder per `{site}` and `{year}`, with all monthly CSVs for the season (Mar to Dec, see `config.SEASON_MONTHS`) in `S1/` and `S2/`.
- File names must match `PART_FILE_RE` in `config.py` (`part{i}of{n}`), otherwise the file is flagged as unrecognised.
- Every year in `CROP_YEARS` used for training needs both S1 and S2 CSVs. The predict year too.
- The shapefile is only required for step 4 (gridding), one per year.

**Expected CSV columns**

| Sensor | Columns |
|---|---|
| S1 | `id_parcel`, `date`, `orbit_number`, `orbit_direction`, `VV`, `VH` (linear), `angle` |
| S2 | `id_parcel`, `date`, `B2`, `B3`, `B4`, `B5`, `B6`, `B7`, `B8`, `B8A`, `B11`, `B12`, `MSK_CLDPRB` |
| Both | `code_cultu`, `code_group` (from the RPG, left empty or absent for the predict year) |

**Generated automatically** under `data/sites_output/{site}/{year}/`: `interim/`, `products/` (including `products/classifier/`) and `diagnostics/`.
---

## Step 3: Preprocessing (`preprocessing/preproc_run.py`)

```
python -m preprocessing.preproc_run --site Strasbourg --years 2020 2021 2022 2023 2024
```

**Input:** `data/satellite/{site}/{year}/S1/*.csv` and `S2/*.csv`, named `{S1|S2}_{site}_{year}_{YYYY-MM}_part{i}of{n}.csv`. The loader warns about missing months, mixed tilings and unusual tile sets.

| Sensor | Checkpoint chain | What happens |
|---|---|---|
| S1 | `s1_merged` → `s1_clean` | Dedup of overlapping tiles (median per parcel/date/orbit direction), then dB conversion (`VV_db`, `VH_db`) and `VH_VV_ratio`. Spike/dip removal on `VV_db` (threshold 5 dB, per parcel and relative orbit), which drops whole rows. |
| S2 | `s2_merged` → `s2_filtered` | Dedup, then cloud filter `MSK_CLDPRB < 1`, then isolated NDVI dip removal (drop > 0.4 vs both neighbours). |
| S2 NDVI | `s2_ndvi` → `s2_ndvi_smoothed_*` → `s2_ndvi_resampled_*_10d` | NDVI from B8/B4, Savitzky-Golay smoothing (window 11 d, order 2), then resampling to a 10-day grid over the season (Mar to Dec). Smoothing and resampling are skipped with `--no-smooth`. |

**Notes**
- Season months are set in `config.SEASON_MONTHS`. Smoothing method and parameters are in `SMOOTHING_PARAMS`.
- `s1_clean` keeps linear `VV`/`VH`. Steps 4 and 6 use those, not the dB columns.
- `s2_ndvi` is not smoothed (it is computed from the filtered S2). Only the `smoothed`/`resampled` files are smoothed, and only step 5 uses them.
- The CLI has no `--redo-from`. Call `run_s1` / `run_s2` from Python, or delete the files.

---

## Step 4: Gridding (`gridding_run.py`, imports from `spatial/`)

Builds a bare-soil S1 signal per SAFRAN cell, for each year.

```
python -m spatial.gridding_run --site Strasbourg --years 2020 2021 2022 2023 2024 \
    [--no-fill] [--redo-from correspondence|ndvi_match|grid_signal|grid_filled]
```

(Adjust the module path to wherever `gridding_run.py` actually sits.)

**Input** (the run fails if any is missing for any requested year):
- `products/s1_clean.parquet` and `products/s2_ndvi.parquet` (step 3)
- the filtered RPG shapefile `data/parcels/filtered/{site}/rpg_{year}/rpg_{year}_{site}.shp`
- the SAFRAN polygons `meteo_france_safran/mailles_safran/safran_polygones.gpkg` (cell id column `cell`, reprojected to EPSG:2154)

**Checkpoint chain:** `correspondence` → `ndvi_match` → `grid_signal` → `grid_filled`.

1. **correspondence** (`interim/correspondence.parquet`): each parcel's representative point is assigned to the SAFRAN cell that contains it. Parcels outside any cell (which shouldn't actually happen) get a null `cell` (a warning prints the count).
2. **ndvi_match** (`interim/ndvi_match.parquet`, slow): for each (parcel, S1 date), the nearest clear S2 NDVI within 10 days (`NDVI_TOLERANCE_DAYS`), plus per-parcel season stats (Apr to Oct): the 99th-percentile NDVI (`ndvi_q`) and the number of S2 observations (`n_s2_obs`). **No thresholds are applied here.**
3. **grid_signal** (`products/grid_signal.parquet`): eligibility thresholds are applied here, then S1 is averaged per `cell, date, orbit_number, orbit_direction`. A (parcel, S1 date) pair is kept as bare soil if either:
   - it is **always bare**: `ndvi_q < 0.35` and `n_s2_obs >= 5` (all its dates kept), or
   - it is **bare on the date**: the matched NDVI is `< 0.25` (mostly this rule catches more parcels)

   Output columns are `VV_mean`, `VH_mean`, `angle_mean` (linear scale), `n_parcels`, `n_parcels_always_bare`, `n_parcels_bare_on_date` and `year`.
4. **grid_filled** (`products/grid_signal_filled.parquet`, skipped with `--no-fill`): for each cell and acquisition (`date`, `orbit_number`) missing from `grid_signal`, fill with the mean of the values of the directly touching neighbour cells that have it. It is one level only, not iterative (because technically we'll always have a neighboring cell with enough parcels, theis is just for neighboring cells). Added columns are `source` (`own` / `neighbours`) and `n_neighbours`. Filled values are not weighted by `n_parcels`.

**Notes**
- Thresholds (`ALWAYS_BARE_NDVI`, `BARE_ON_DATE_NDVI`, `MIN_S2_OBS`, tolerance, season) are constants at the top of `grid_signal_aggregation.py`. Changing them requires `--redo-from grid_signal` (and `--redo-from ndvi_match` for tolerance, season months or quantile).
- `--redo-from` applies to every year in the run.
- `load_grid_signal(site, years, filled=False)` concatenates several years.

---

## Step 5: Kmeans winter filter (`preliminary_filter/filter_run.py`)

Run for the predict year only (training years don't need it).

```
python -m preliminary_filter.filter_run --site Strasbourg --years 2024 [--days 10] [--redo]
```

**Input:** `products/s2_ndvi_resampled_*_10d.parquet`. `interim/s2_merged.parquet` is optional and is used for diagnostics only. Labels never influence the clustering.

**Method:** for each parcel, two indicators are computed from the smoothed NDVI, with values below 0.05 masked:
- **RD** is the deviation from the all-parcel mean on each date.
- **TA** is the deviation from the parcel's own mean over the period.

Both are averaged over 1 Jun to 30 Oct (period starts 1 Apr), standardised, then clustered with a constrained 4-cluster kmeans. Cluster 0 is a fixed summer anchor (high RD/TA), cluster 2 is a fixed winter anchor (low RD/TA), and clusters 1 and 3 are free. Parameters are in `config.WINTER_FILTER`.

**Output:** `products/winter_filter_{SMOOTH_TAG}_10d.parquet` with `id_parcel`, `RD`, `TA`, `RD_std`, `TA_std`, `cluster`, `status` (`clustered` / `no_ndvi`), `is_winter` (cluster in `winter_clusters` = [2]) and `year`. Diagnostics go to `diagnostics/winter_filter_*/`: `clusters.png` always, and with labels also `crop_group_counts.csv`, `crop_group_percentages.csv`, `composition.png` and `irrigated_leakage.csv`.

`irrigated_leakage.csv` is the key check. It shows how many maize, soybean and sunflower parcels were flagged winter, and those are lost for good. It only works on labelled years.

---

## Step 6: Summer-crop classifier (`classifier/`)

```
python -m classifier.classifier_run --site Strasbourg --year 2024 \
    [--mode both_split_orbit] [--train-years 2021 2022 2023] [--redo-from features|predictions|final]
```

**Input:** `s1_clean` and/or `s2_filtered` for the predict year and every training year, depending on mode. Years with missing files are skipped with a message. All constants are in `classifier_config.py`.

**Modes** (`C.MODES`): `S1`, `S2`, `both`, `S1_ASC`, `S1_DESC`, `S1_SPLIT_ORBIT`, `both_split_orbit` (default). Split-orbit modes build separate S1 features per ascending/descending orbit.

**Classes:** Maize (MIS, MIE, MID), Soybean (SOJ), Sunflower (TRN), and Other vegetation (everything else). A class with fewer than `MIN_PARCELS` (100) parcels in any training year is merged into Other. Code groups 20, 21, 28, 11 (vines, orchards, misc, fallow) are removed from training only.

**Pipeline** (checkpoints in order: `features` → `predictions` → `final`):

1. **features:** per parcel and month (Apr to Nov), the median of the S2 bands (10) and S1 `VV`, `VH`, `VV_VH_ratio`. S1 is normalised to a reference incidence angle (the mean over the training years): `σ · cos²(ref) / cos²(angle)`.
2. **predictions:**
   - Months entirely empty in the predict year are dropped for all years. Gaps are linearly interpolated across months, and the remaining NaN are filled with the training median.
   - XGBoost (`XGB_PARAMS`) is trained with power-law class weights (`POWER_ALPHA` 0.07).
   - A parcel is not classified if a used sensor has no valid month, or has no observation in the season window. The `reason` column says why.
3. **final:** NDVI post-filter. Parcels predicted "Other vegetation" are split by the 90th-percentile NDVI of Mar to Aug (from `s2_filtered`): `>= 0.45` is `cultivated`, `< 0.45` is `non_cultivated`, and a missing value is `unknown`.

**Outputs** in `products/classifier/`, tagged `{mode}_train_{years}`:
- `features_*.parquet`
- `predictions_{year}_*.parquet` with `classified`, `reason`, `pred_class`, `pred_name`, `confidence`, `proba_*`, `n_valid_months_*` and `window_*`
- `classification_{year}_*.parquet` and `.csv`, which are the predictions plus `ndvi_stat` and `veg_subclass`

---

## Known open point: kmeans filter is not yet consumed by the classifier

`apply_prefilter` exists in `classifier_run.py` but is **not called** by `run_classifier`. For now the winter flags from step 5 have no effect on the classification output. Wiring is still to be decided. The function overrides flagged parcels to "Other vegetation" and keeps `pred_name_raw` and `prefiltered` for auditing. Its error message mentions `filter_run --no-labels`, which doesn't exist (the command is the one in step 5).

## Things to verify

- **Step 3:** check the warnings from `_check_parts` (missing months, tile mismatches), the number of rows dropped by each filter, and that the thresholds (5 dB radar, cloud < 1, NDVI dip 0.4) behave sensibly.
- **Step 4:**
  - the share of parcels outside any SAFRAN cell
  - the share of S1 pairs with no clear S2 within 10 days
  - the eligible-pair percentage per year
  - cells with very low `n_parcels`
  - the proportion of neighbour-filled rows
- **Step 5:** check `clusters.png`, and `irrigated_leakage.csv` on a labelled year (it should be close to 0% for MIS/MIE/MID/SOJ/TRN).
- **Step 6:**
  - the printed class counts per year and which classes were merged into Other
  - the `[window]` lines (months used or dropped in the predict year)
  - the share of parcels classified and the `reason` values
  - that `ref_angle` is reasonable
- **Consistency:** `days=10` must be used in steps 3 and 5 (the filename includes it).