import pandas as pd
import numpy as np
import glob
import os


'''date', 'id_partita', 'minute', 'home', 'away', 'campionato',
       'home_score', 'away_score', 'home_possesso_palla',
       'away_possesso_palla', 'home_tiri', 'away_tiri', 'home_tiri_in_porta',
       'away_tiri_in_porta', 'home_tiri_fuori', 'away_tiri_fuori',
       'home_tiri_fermati', 'away_tiri_fermati', 'home_punizioni',
       'away_punizioni', 'home_calci_d_angolo', 'away_calci_d_angolo',
       'home_fuorigioco', 'away_fuorigioco', 'home_rimesse_laterali',
       'away_rimesse_laterali', 'home_parate', 'away_parate', 'home_falli',
       'away_falli', 'home_cartellini_rossi', 'away_cartellini_rossi',
       'home_cartellini_gialli', 'away_cartellini_gialli',
       'home_passaggi_totali', 'away_passaggi_totali',
       'home_passaggi_completati', 'away_passaggi_completati',
       'home_contrasti', 'away_contrasti', 'home_attacchi', 'away_attacchi',
       'home_attacchi_pericolosi', 'away_attacchi_pericolosi', 'odd_1',
       'odd_X', 'odd_2', 'odd_over', 'odd_under', 'live_odd_1', 'live_odd_X',
       'live_odd_2', 'live_odd_over', 'live_odd_under', 'home_final_score',
       'away_final_score'''


def get_df(path):
    file_path = os.path.dirname(os.path.abspath(__file__))
    all_files = sorted(glob.glob(f"{file_path}/{res_path}/*.csv"),
                       key=lambda x: int(x[x.index('/csv/') + 10:-4]))
    input_df = pd.read_csv(all_files[-1], index_col=None, header=0)
    if 'Unnamed: 0' in input_df.columns:
        input_df.drop(columns=['Unnamed: 0'], inplace=True)
    return input_df.sort_values(by=['id_partita', 'minute'], ascending=[True, False])


def drop_nan(df, thresh='half'):
    # eliminate duplicate rows
    subset = [col for col in df.columns if col != 'minute']
    df.drop_duplicates(subset=subset, inplace=True)

    # eliminate rows with a lot of nans
    if thresh == 'half':
        thresh = len(df.columns) // 2
    df.dropna(axis=0, thresh=thresh, inplace=True)

    # eliminate rows with nans on target or on important columns
    important_cols = ['id_partita']
    df.dropna(axis=0, subset=important_cols, how='any', inplace=True)

    # drop matches already in over
    over_mask = (df['home_score'] + df['away_score']) >= 3
    ids = df.loc[over_mask, 'id_partita'].unique()
    df.drop(df[df['id_partita'].isin(ids)].index, inplace=True)


def normalize_prematch_odds(input_df):
    tmp = (1 - ((1 / input_df['odd_over']) + (1 / input_df['odd_under']))) / 2
    input_df['odd_over'] = (1 / input_df['odd_over']) + tmp
    input_df['odd_under'] = (1 / input_df['odd_under']) + tmp


def pop_prematch_odds_data(input_df):
    prematch_odds_input = input_df.loc[:, [
        'id_partita', 'minute', 'odd_under', 'odd_over']].copy()
    input_df.drop(columns=['odd_1', 'odd_2', 'odd_X',
                           'odd_over', 'odd_under'], inplace=True)
    return prematch_odds_input


def pop_live_odds_data(input_df):
    live_odds_input = input_df.loc[:, [
        'id_partita', 'minute', 'live_odd_under', 'live_odd_over']].copy()
    input_df.drop(columns=['live_odd_1', 'live_odd_2', 'live_odd_X',
                           'live_odd_over', 'live_odd_under'], inplace=True)
    return live_odds_input


def to_numeric(df, cat_col):
    # change data type
    for col in df.columns:
        if col not in cat_col:
            df[col] = pd.to_numeric(df[col])


def drop_outcome_cols(df):
    df.drop(columns=['home_final_score', 'away_final_score'], inplace=True)


def add_input_cols(df):
    df['actual_total_goals'] = df['home_score'] + df['away_score']
    df['over_strongness'] = (
        df['home_score'] + df['away_score']) * (90 - df['minute'])


def impute_nan(train_df, test_df, thresh='half'):
    # handling odds cols
    if 'odd_under' in test_df.columns:
        test_df.loc[test_df['odd_under'] == 0, 'odd_under'] = 2
    if 'odd_over' in test_df.columns:
        test_df.loc[test_df['odd_over'] == 0, 'odd_over'] = 2
    if 'odd_1' in test_df.columns:
        test_df.loc[test_df['odd_1'] == 0, 'odd_1'] = 3
    if 'odd_X' in test_df.columns:
        test_df.loc[test_df['odd_X'] == 0, 'odd_X'] = 3
    if 'odd_2' in test_df.columns:
        test_df.loc[test_df['odd_2'] == 0, 'odd_2'] = 3

    # imputing the other nans
    nan_cols = [i for i in test_df.columns if test_df[i].isnull(
    ).any() if i not in ['home_final_score', 'away_final_score']]
    for col in nan_cols:
        col_df = train_df[(~train_df['home_' + col[5:]].isnull())
                          & (~train_df['away_' + col[5:]].isnull())]
        if 'away' in col:
            continue
        col = col[5:]
        nan_mask = test_df['home_' +
                           col].isnull() | test_df['away_' + col].isnull()
        if "possesso_palla" in col:
            test_df.loc[nan_mask, 'home_possesso_palla'] = 50
            test_df.loc[nan_mask, 'away_possesso_palla'] = 50
            continue
        for m in np.arange(5, 90, 5):
            mask_min_test = test_df['minute'] >= m
            mask_max_test = test_df['minute'] <= m + 5
            mask_min_train = col_df['minute'] >= m
            mask_max_train = col_df['minute'] <= m + 5
            test_df.loc[(mask_min_test) & (mask_max_test) & (nan_mask), 'home_' +
                        col] = col_df.loc[mask_min_train & mask_max_train, ['home_' + col, 'away_' + col]].mean().mean()
            test_df.loc[(mask_min_test) & (mask_max_test) & (nan_mask), 'away_' +
                        col] = col_df.loc[mask_min_train & mask_max_train, ['home_' + col, 'away_' + col]].mean().mean()
    test_df.dropna(inplace=True)
