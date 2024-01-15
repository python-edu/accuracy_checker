import numpy as np
import pandas as pd


def ar2list(ar: np.array) -> list[list]:
    res = ar.tolist()
    return res
# --


def df2list(df: pd.DataFrame) -> list[list]:
    df = df.copy()
    if df.index.name is None and df.columns.name is None:
        name = '-'

    elif df.index.name is not None and df.columns.name is not None:
        name = f'{df.index.name}/{df.columns.name}'
    elif df.index.name is None:
        name = df.columns.name
    else:
        name = df.index.name

    df.index.name = name

    df = df.reset_index()
    cols = df.columns.to_list()
    res = df.to_numpy()
    res = ar2list(res)
    res.insert(0, cols)
    return res
