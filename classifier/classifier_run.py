# carto_irrig/classifier/classifier_run.py
"""
Train on the previous labelled years of a site, predict the newest (unlabelled) year.
Usage: python -m classifier.classifier_run --site Strasbourg --year 2024
"""
import argparse
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config import YearPaths, CROP_YEARS
from common.checkpoint import step, invalidate
from classifier import classifier_config as C
from classifier import features as F
from classifier.ndvi_postfilter import build_ndvi_table, add_veg_subclass


# -- inputs / paths -----------------------------------------------------------
def _needed(yp, mode):
    use_s1, use_s2, _ = C.MODES[mode]
    return ([yp.s1_clean] if use_s1 else []) + ([yp.s2_filtered] if use_s2 else [])


def find_train_years(site, predict_year, mode, years=None):
    cand = years or [y for y in CROP_YEARS if y < predict_year]
    ok = []
    for y in cand:
        missing = [p for p in _needed(YearPaths(site, y), mode) if not p.exists()]
        if missing:
            print(f"  [train] skipping {y}: missing {[p.name for p in missing]}")
        else:
            ok.append(y)
    if not ok:
        raise FileNotFoundError(f"[{site}] no preprocessed training year before {predict_year} "
                                f"(need {[p.name for p in _needed(YearPaths(site, predict_year), mode)]} for each).")
    return ok


def classifier_paths(site, predict_year, mode, train_years):
    root = YearPaths(site, predict_year).root / "products" / "classifier"
    tag = f"{mode}_train_{'_'.join(map(str, train_years))}"
    return {"features":    root / f"features_{tag}.parquet",       # order = pipeline order (for invalidate)
            "predictions": root / f"predictions_{predict_year}_{tag}.parquet",
            "final":       root / f"classification_{predict_year}_{tag}.parquet"}


def apply_prefilter(preds: pd.DataFrame, yp) -> pd.DataFrame:
    """Parcels flagged by the label-free k-means filter are forced to 'Other vegetation'."""
    flags_path = yp.winter_filter
    if not flags_path.exists():
        raise FileNotFoundError(f"{flags_path} — run filter_run --no-labels for {yp.year} first.")
    flags = pd.read_parquet(flags_path, columns=["id_parcel", "is_winter"])
    flags["id_parcel"] = flags["id_parcel"].astype(str)

    out = preds.merge(flags, on="id_parcel", how="left")
    out["prefiltered"] = out["is_winter"].fillna(False).astype(bool) & out["classified"].astype(bool)
    out["pred_name_raw"] = out["pred_name"]                 # what the model said before the filter
    out.loc[out["prefiltered"], "pred_name"] = C.OTHER_NAME
    n = int(out["prefiltered"].sum())
    changed = int((out["pred_name"] != out["pred_name_raw"]).sum())
    print(f"  [prefilter] {n:,} parcels flagged, {changed:,} predictions changed to Other")
    return out.drop(columns="is_winter")

# -- model --------------------------------------------------------------------
def _power_weights(y, alpha):
    classes, counts = np.unique(y, return_counts=True)
    w = (counts.max() / counts) ** alpha
    return w[np.searchsorted(classes, y)]


def classify(feats: pd.DataFrame, predict_year: int, mode: str, all_ids) -> pd.DataFrame:
    use_s1, use_s2, _ = C.MODES[mode]
    used = [p for p, u in (("s1", use_s1), ("s2", use_s2)) if u]

    feats = F.drop_unavailable_months(feats, predict_year)
    cols = F.feature_columns(feats)
    counts = {p: F.count_valid_months(feats, cols, p) for p in used}    # before filling
    feats = F.interpolate_monthly(feats)
    for p, n in counts.items():
        feats[f"n_valid_months_{p}"] = n

    pred = feats[feats["agri_year"] == predict_year].reset_index(drop=True)
    train_all = feats[feats["agri_year"] != predict_year]
    if train_all.empty:
        raise ValueError("No training parcels (need labelled years before the predict year).")
    nan_share = train_all.groupby("agri_year")["code_cultu"].apply(lambda s: s.isna().mean())
    bad = nan_share[nan_share > 0.5]
    if not bad.empty:
        raise ValueError(f"code_cultu is mostly missing in training year(s) {list(bad.index)}: "
                         "these years have no labels.")

    crops, names = F.build_classes(train_all)
    train = train_all[~train_all["code_group"].isin(C.EXCLUDED_CODE_GROUPS)]
    y_raw = F.assign_labels(train["code_cultu"], crops)
    classes, y_idx = np.unique(y_raw, return_inverse=True)
    if len(classes) < 2:
        raise ValueError(f"Training data has a single class: {classes}")
    print(f"  [train] {len(train):,} parcels, years {sorted(int(y) for y in train['agri_year'].unique())} "
          f"-> predict {predict_year} ({len(pred):,} parcels)")
    print("  [train] class counts:",
          {names[int(c)]: int((y_raw == c).sum()) for c in classes})

    X_tr = train[cols].to_numpy(float)
    X_pr = pred[cols].to_numpy(float)
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    for X in (X_tr, X_pr):
        r, c = np.where(np.isnan(X))
        X[r, c] = med[c]

    clf = XGBClassifier(**C.XGB_PARAMS)
    clf.fit(X_tr, y_idx, sample_weight=_power_weights(y_idx, C.POWER_ALPHA))

    # -- usability: a used sensor with no valid month -> not classified
    parts = [np.where(pred[f"n_valid_months_{p}"].to_numpy() == 0, f"no valid {p.upper()} month; ", "")
             for p in used]
    reason = pd.Series(["".join(t) for t in zip(*parts)])
    ok = (reason == "").to_numpy()

    K = len(names)
    P = np.full((len(pred), K), np.nan)
    if ok.any():
        Pok = np.zeros((int(ok.sum()), K))
        Pok[:, classes] = clf.predict_proba(X_pr[ok])
        P[ok] = Pok
    Pz = np.nan_to_num(P)

    out = pred[["id_parcel", "agri_year"]].copy()
    out["classified"] = ok
    out["reason"] = reason.str.strip("; ").to_numpy()
    out["pred_class"] = np.where(ok, Pz.argmax(1), np.nan)
    out["pred_name"] = out["pred_class"].map(names)
    out["confidence"] = np.where(ok, Pz.max(1), np.nan)
    for i in range(K):
        out[f"proba_{names[i].lower().replace(' ', '_')}"] = P[:, i]
    for p in used:
        out[f"n_valid_months_{p}"] = pred[f"n_valid_months_{p}"].to_numpy()
        out[f"window_{p}"] = F.window_label(cols, p)

    # parcels present in the inputs but with no observation inside the season window
    lost = sorted(set(all_ids) - set(out["id_parcel"]))
    if lost:
        out = pd.concat([out, pd.DataFrame({
            "id_parcel": lost, "agri_year": predict_year, "classified": False,
            "reason": "no valid observation in season window"})], ignore_index=True)

    n_ok = int(out["classified"].sum())
    print(f"  [predict] {n_ok:,} classified / {len(out):,} parcels ({100 * n_ok / max(len(out), 1):.1f}%)")
    print(out.loc[out["classified"], "pred_name"].value_counts().to_string())
    return out


# -- runner -------------------------------------------------------------------
def run_classifier(site, predict_year=None, mode="both_split_orbit",
                   train_years=None, redo_from=None) -> pd.DataFrame:
    if mode not in C.MODES:
        raise ValueError(f"mode must be one of {list(C.MODES)}, got {mode!r}")
    predict_year = predict_year or CROP_YEARS[-1]
    use_s1, use_s2, _ = C.MODES[mode]

    miss = [p for p in _needed(YearPaths(site, predict_year), mode) if not p.exists()]
    if miss:
        raise FileNotFoundError(f"Run preprocessing for {site} {predict_year} first, missing: {miss}")
    train_years = find_train_years(site, predict_year, mode, train_years)
    print(f"[classifier] {site} | predict {predict_year} | train {train_years} | mode {mode}")

    p = classifier_paths(site, predict_year, mode, train_years)
    invalidate(p, redo_from)
    ypred = YearPaths(site, predict_year)

    def compute_features():
        ref = F.ref_angle([YearPaths(site, y).s1_clean for y in train_years]) if use_s1 else None
        frames = []
        for y in train_years + [predict_year]:
            yp = YearPaths(site, y)
            frames.append(F.year_features(yp.s1_clean, yp.s2_filtered, y, mode, ref))
        return F.stack_years(frames)

    feats = step(p["features"], compute_features)

    def all_ids():
        ids = []
        for flag, path in ((use_s1, ypred.s1_clean), (use_s2, ypred.s2_filtered)):
            if flag:
                ids.append(pd.read_parquet(path, columns=["id_parcel"])["id_parcel"].astype(str))
        return pd.concat(ids).unique()

    preds = step(p["predictions"], lambda: classify(feats, predict_year, mode, all_ids()))

    def compute_final():
        if C.NDVI_POSTFILTER and ypred.s2_filtered.exists():
            return add_veg_subclass(preds, build_ndvi_table(ypred.s2_filtered))
        if C.NDVI_POSTFILTER:
            print("  [warn] NDVI post-filter skipped: s2_filtered missing")
        return preds

    final = step(p["final"], compute_final)
    final.to_csv(p["final"].with_suffix(".csv"), index=False)
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--year", type=int, default=None, help="year to predict (default: last CROP_YEARS)")
    ap.add_argument("--mode", default="both_split_orbit", choices=list(C.MODES))
    ap.add_argument("--train-years", type=int, nargs="+", default=None)
    ap.add_argument("--redo-from", default=None, choices=["features", "predictions", "final"])
    a = ap.parse_args()
    run_classifier(a.site, a.year, a.mode, a.train_years, a.redo_from)