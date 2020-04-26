import pandas as pd
import glob
import os


def get_df(res_path, cat_col):
    file_path = os.path.dirname(os.path.abspath(__file__))
    all_files = sorted(glob.glob(f"{file_path}/{res_path}/*.csv"),
                       key=lambda x: int(x[x.index('stats') + 5:-4]))
    li = [pd.read_csv(filename, index_col=None, header=0)
          for filename in all_files]
    df = pd.concat(li, axis=0, ignore_index=True)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    # change data type
    for col in df.columns:
        if col not in cat_col:
            df[col] = pd.to_numeric(df[col])
    return df.reset_index(drop=True)
