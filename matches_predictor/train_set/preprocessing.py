import pandas as pd
import numpy as np
import os


def _impute_nan(df, thresh='half'):
    # handling odds cols
    if 'odd_under' in df.columns:
        df.loc[df['odd_under'] == 0, 'odd_under'] = 2
    if 'odd_over' in df.columns:
        df.loc[df['odd_over'] == 0, 'odd_over'] = 2
    if 'odd_1' in df.columns:
        df.loc[df['odd_1'] == 0, 'odd_1'] = 3
    if 'odd_X' in df.columns:
        df.loc[df['odd_X'] == 0, 'odd_X'] = 3
    if 'odd_2' in df.columns:
        df.loc[df['odd_2'] == 0, 'odd_2'] = 3

    # imputing the other nans
    nan_cols = [i for i in df.columns if df[i].isnull().any() if i not in [
        'home_final_score', 'away_final_score']]
    for col in nan_cols:
        col_df = df[(~df['home_' + col[5:]].isnull()) &
                    (~df['away_' + col[5:]].isnull())]
        if 'away' in col:
            continue
        col = col[5:]
        nan_mask = df['home_' + col].isnull() | df['away_' + col].isnull()
        if "possesso_palla" in col:
            df.loc[nan_mask, 'home_possesso_palla'] = 50
            df.loc[nan_mask, 'away_possesso_palla'] = 50
            continue
        for m in np.arange(5, 90, 5):
            mask_min_test = df['minute'] >= m
            mask_max_test = df['minute'] <= m + 5
            mask_min_train = col_df['minute'] >= m
            mask_max_train = col_df['minute'] <= m + 5
            df.loc[(mask_min_test) & (mask_max_test) & (nan_mask), 'home_' +
                   col] = col_df.loc[mask_min_train & mask_max_train, ['home_' + col, 'away_' + col]].mean().mean()
            df.loc[(mask_min_test) & (mask_max_test) & (nan_mask), 'away_' +
                   col] = col_df.loc[mask_min_train & mask_max_train, ['home_' + col, 'away_' + col]].mean().mean()
    df.dropna(inplace=True)


def _drop_nan(df, thresh='half'):
    # eliminate duplicate rows
    subset = [col for col in df.columns if col != 'minute']
    df.drop_duplicates(subset=subset, inplace=True)

    # eliminate rows with a lot of nans
    if thresh == 'half':
        thresh = len(df.columns) // 2
    df.dropna(axis=0, thresh=thresh, inplace=True)

    # eliminate rows with nans on target or on important columns
    important_cols = ['home_final_score', 'away_final_score', 'id_partita']
    df.dropna(axis=0, subset=important_cols, how='any', inplace=True)

    # drop matches already in over
    over_mask = (df['home_score'] + df['away_score']) >= 3
    df.drop(df[over_mask].index, inplace=True)


def _to_numeric(df, cat_col):
    # change data type
    for col in df.columns:
        if col not in cat_col:
            df[col] = pd.to_numeric(df[col])


def _add_outcome_col(df):
    df['final_uo'] = np.where(
        df['home_final_score'] + df['away_final_score'] > 2, 0, 1)


def _add_input_cols(df):
    df['actual_total_goals'] = df['home_score'] + df['away_score']
    df['over_strongness'] = (
        df['home_score'] + df['away_score']) * (90 - df['minute'])


def _drop_prematch_odds_col(df):
    df.drop(columns=['odd_over', 'odd_under',
                     'odd_1', 'odd_X', 'odd_2'], inplace=True)


def _drop_live_odds_col(df):
    df.drop(columns=['live_odd_over', 'live_odd_under',
                     'live_odd_1', 'live_odd_X', 'live_odd_2'], inplace=True)


def _drop_odds_cols(df):
    _drop_prematch_odds_col(df)
    _drop_live_odds_col(df)


def _save(df):
    file_path = os.path.dirname(os.path.abspath(__file__))
    df.reset_index(drop=True).to_csv(
        file_path + "/../../res/dataframes/training_goals.csv")


########################### main function #############
def execute(train_df, cat_col, prod=True):
    _to_numeric(train_df, cat_col)
    _drop_odds_cols(train_df)
    _drop_nan(train_df)
    _impute_nan(train_df)
    _add_outcome_col(train_df)
    _add_input_cols(train_df)
    if prod:
        _save(train_df)
