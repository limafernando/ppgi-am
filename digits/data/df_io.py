import pandas as pd
from pathlib import Path
from typing import Tuple

def read(path: Path, sep=";") -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)

def get_feat_lables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    label = df["label"]
    features = df.drop("label", axis=1)
    return features, pd.DataFrame(label)
