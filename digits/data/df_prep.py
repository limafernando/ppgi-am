from typing import Tuple
import numpy as np
import pandas as pd


def get_specific_split(X: pd.DataFrame, Y: pd.DataFrame, d1: int, d2: int) -> Tuple[np.array, np.array]:
    df = pd.concat([X, Y], axis=1)

    df1 = df[(df["label"] == d1)]
    df2 =  df[(df["label"] == d2)]

    df = pd.concat([df1, df2]).reset_index()

    return df[["intensity", "symmetry"]].to_numpy(), df["label"].replace(5, -1).to_numpy()

def get_intensity(df: pd.DataFrame) -> pd.Series:
    aux = df.apply(np.sum, axis=1)
    return aux.apply(lambda x: round(x/255, 2)) # Series does not have axis


def reshape_rows(df: pd.DataFrame) -> np.array:
    reshaped_arrays = []
    for _, row in df.iterrows():
        reshaped_array = row.values.reshape((28, 28))
        reshaped_arrays.append(reshaped_array)
    return np.array(reshaped_arrays)


def get_vertical_sym(df: pd.DataFrame) -> pd.Series:
    reshaped_df = reshape_rows(df)
    left, right = reshaped_df[:, :, :14], reshaped_df[:, :, 14:]
    diff = np.abs(left - right)
    res = np.zeros(reshaped_df.shape[0])
    idx = 0
    aux = None
    for x in diff:
        aux = round(np.sum(x)/255, 2)
        res[idx] = aux
        idx += 1
    return pd.Series(res)


def get_horizontal_sym(df: pd.DataFrame) -> pd.Series:
    pass


def get_symmetry(df: pd.DataFrame, sym_type: str = "vertical") -> pd.Series:
    if sym_type == "vertical":
        return get_vertical_sym(df)
    if sym_type == "horizontal":
        return get_horizontal_sym(df)
    

def get_pixel_count(df: pd.DataFrame) -> pd.Series:
    aux = df.apply(np.count_nonzero, axis=1)
    return aux.apply(lambda x: round(x/255, 2)) # Series does not have axis
     

def pre_processing(df: pd.DataFrame, sym_type: str = "vertical") -> pd.DataFrame:
    intensity = get_intensity(df)
    symmetry = get_symmetry(df, sym_type)
    return pd.DataFrame({"intensity": intensity, "symmetry": symmetry})


def pre_processing2(df: pd.DataFrame, sym_type: str = "vertical") -> pd.DataFrame:
    px_count = get_pixel_count(df)
    symmetry = get_symmetry(df, sym_type)
    return pd.DataFrame({"pixel count": px_count, "symmetry": symmetry})
