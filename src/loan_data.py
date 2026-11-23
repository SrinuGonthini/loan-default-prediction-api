import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def load_loans() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "loans.csv", low_memory=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["HasCoSigner", "HasDependents", "HasMortgage"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map({"Yes": 1, "No": 0})
    return df


def get_engine():
    url = (
        f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    )
    return create_engine(url)


def save_to_db(df: pd.DataFrame, table: str = "loans") -> None:
    df.to_sql(table, get_engine(), if_exists="replace", index=False, chunksize=10_000)


def read_from_db(query: str = "SELECT * FROM loans") -> pd.DataFrame:
    return pd.read_sql(query, get_engine())
