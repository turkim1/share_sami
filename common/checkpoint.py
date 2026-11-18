# common/checkpoint.py
import time
from pathlib import Path
import pandas as pd

def step(path: Path, compute) -> pd.DataFrame:
    """Load the checkpoint if it exists, otherwise compute and save it."""
    where = "/".join(path.parts[-5:])
    if path.exists():
        print(f"[checkpoint] reuse {where}")
        return pd.read_parquet(path)
    t = time.time()
    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[checkpoint] wrote {where}  ({len(df):,} rows, {time.time() - t:.0f}s)")
    return df

def invalidate(paths: dict[str, Path], redo_from: str | None):
    """Delete the checkpoint `redo_from` and every later one, so nothing stale survives."""
    if redo_from is None:
        return
    names = list(paths)
    for name in names[names.index(redo_from):]:
        paths[name].unlink(missing_ok=True)