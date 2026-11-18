from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_loans() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "loans.csv", low_memory=True)
