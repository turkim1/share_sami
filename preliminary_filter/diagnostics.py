"""
preliminary_filter/diagnostics.py
Checks that the winter cluster really contains winter crops.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from classifier import classifier_config as C

# irrigated target crops (maize, soybean, sunflower): the codes we cannot afford to lose
IRRIGATED = set().union(*C.CLASS_DEFS_ALL.values())

WINTER_CEREAL  = {"BTH", "BDH", "ORH", "AVH", "CHA", "CHH", "CHS", "CHT", "SGH", "TTH", "CAH"}
WINTER_CZH     = {"CZH"}
SPRING_CEREAL  = {"BTP", "ORP", "AVP", "BDP", "BDT", "CPA", "CPH", "CPS", "CPT", "CPZ", "SGP", "TTP"}
SPRING_CZP     = {"CZP"}
SPRING_SPECIAL = {"MID", "MIE", "MIS", "TRN", "SOJ"}

CLUSTER_COLORS = {0: "orange", 1: "purple", 2: "green", 3: "red"}
CLUSTER_NAMES  = {0: "0 summer anchor", 1: "1 free", 2: "2 winter anchor", 3: "3 free"}


def classify_crop(code) -> str:
    if code in WINTER_CEREAL:
        return "Winter cereal"
    if code in WINTER_CZH:
         return "Winter CZH"
    if code in SPRING_CEREAL:
        return "Spring cereal"
    if code in SPRING_CZP:
         return "Spring CZP"
    if code in SPRING_SPECIAL:
        return "Spring MID/MIE/MIS/TRN/SOJ"
    return "Other"

def has_labels(flags: pd.DataFrame) -> bool:
    return "code_cultu" in flags and flags["code_cultu"].notna().any()


def composition(flags):
    """Counts and row-percentages of crop groups per cluster (clustered parcels only)."""
    d = flags[flags["status"] == "clustered"].copy()
    d["crop_group"] = d["code_cultu"].map(classify_crop)
    counts = pd.crosstab(d["cluster"], d["crop_group"])
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    return counts, pct


def irrigated_leakage(flags: pd.DataFrame) -> pd.DataFrame:
    """Per irrigated crop code: how many parcels were flagged non-irrigated. These are lost for good."""
    d = flags[flags["code_cultu"].isin(IRRIGATED)]
    out = d.groupby("code_cultu")["is_winter"].agg(n="size", flagged="sum")
    out.loc["TOTAL"] = out.sum()
    out["pct_flagged"] = (100 * out["flagged"] / out["n"]).round(2)
    return out


def write_diagnostics(flags: pd.DataFrame, out_dir: Path, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    c = flags[flags["status"] == "clustered"]

    fig, ax = plt.subplots(figsize=(8, 7))
    for k, g in c.groupby("cluster"):
        ax.scatter(g["RD_std"], g["TA_std"], s=25, alpha=0.6,
                   color=CLUSTER_COLORS[int(k)], label=CLUSTER_NAMES[int(k)])
    ax.axhline(0, ls="--", lw=1, alpha=0.6)
    ax.axvline(0, ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Standardized RD")
    ax.set_ylabel("Standardized TA")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, ls="--", lw=0.5, alpha=0.7)
    fig.savefig(out_dir / "clusters.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if not has_labels(flags):
        print("    [diag] no labels: scatter only")
        return

    counts, pct = composition(flags)
    counts.to_csv(out_dir / "crop_group_counts.csv")
    pct.to_csv(out_dir / "crop_group_percentages.csv")
    leak = irrigated_leakage(flags)
    leak.to_csv(out_dir / "irrigated_leakage.csv")
    print("    [diag] irrigated crops flagged as non-irrigated:\n", leak.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    counts.plot(kind="bar", stacked=True, ax=axes[0])
    axes[0].set_ylabel("Number of parcels")
    axes[0].set_title("Parcels per cluster")
    pct.plot(kind="bar", stacked=True, ax=axes[1], legend=False)
    axes[1].set_ylabel("Crop composition (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Crop-group composition")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / "composition.png", dpi=200, bbox_inches="tight")
    plt.close(fig)