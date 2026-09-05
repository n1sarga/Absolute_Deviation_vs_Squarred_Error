from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

DATASETS = {
    "boston_housing": {
        "file": "boston_housing.csv",
        "target": "MEDV",
        "drop": ["B"],
        "skiprows": 1,
    },
    "concrete_strength": {
        "file": "concrete_strength.csv",
        "target": "Strength",
        "drop": [],
        "skiprows": 0,
    },
    "hbk": {
        "file": "hbk.csv",
        "target": "Y",
        "drop": ["Observation", "CaseGroup"],
        "skiprows": 0,
    },
}


def load_dataset(name: str):
    cfg = DATASETS[name]
    df = pd.read_csv(DATA_DIR / cfg["file"], skiprows=cfg["skiprows"])
    df = df.dropna().copy()
    y = df[cfg["target"]].astype(float).to_numpy()
    Xdf = df.drop(columns=[cfg["target"], *cfg["drop"]], errors="ignore")
    Xdf = Xdf.select_dtypes(include="number")
    return Xdf.to_numpy(dtype=float), y, list(Xdf.columns)
