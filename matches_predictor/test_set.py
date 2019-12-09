from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import glob


def get_df():
    all_files = sorted(glob.glob("./csv/*.csv"), key=lambda x: int(x[x.index('/csv/') + 10:-4]))
    li = [pd.read_csv(filename, index_col=None, header=0) for filename in all_files]
    df = pd.concat(li, axis=0, ignore_index=True)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    return df.reset_index(drop=True)


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


def to_numeric(df, cat_col):
    # change data type
    for col in df.columns:
        if col not in cat_col:
            df[col] = pd.to_numeric(df[col])


def normalize_prematch_odds(input_df):
    tmp = (1 - ((1 / input_df['odd_over']) + (1 / input_df['odd_under']))) / 2
    input_df['odd_over'] = (1 / input_df['odd_over']) + tmp
    input_df['odd_under'] = (1 / input_df['odd_under']) + tmp


def pop_prematch_odds_data(input_df):
    prematch_odds_input = input_df.loc[:, ['id_partita', 'minute', 'odd_under', 'odd_over']].copy()
    input_df.drop(columns=['odd_1', 'odd_2', 'odd_X', 'odd_over', 'odd_under'], inplace=True)
    return prematch_odds_input


def pop_live_odds_data(input_df):
    live_odds_input = input_df.loc[:, ['id_partita', 'minute', 'live_odd_under', 'live_odd_over']].copy()
    input_df.drop(columns=['live_odd_1', 'live_odd_2', 'live_odd_X', 'live_odd_over', 'live_odd_under'], inplace=True)
    return live_odds_input


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
    nan_cols = [i for i in test_df.columns if test_df[i].isnull().any() if i not in ['home_final_score', 'away_final_score']]
    for col in nan_cols:
        col_df = train_df[(~train_df['home_' + col[5:]].isnull()) & (~train_df['away_' + col[5:]].isnull())]
        if 'away' in col:
            continue
        col = col[5:]
        nan_mask = test_df['home_' + col].isnull() | test_df['away_' + col].isnull()
        if "possesso_palla" in col:
            test_df.loc[nan_mask, 'home_possesso_palla'] = 50
            test_df.loc[nan_mask, 'away_possesso_palla'] = 50
            continue
        for m in np.arange(5, 90, 5):
            mask_min_test = test_df['minute'] >= m
            mask_max_test = test_df['minute'] <= m + 5
            mask_min_train = col_df['minute'] >= m
            mask_max_train = col_df['minute'] <= m + 5
            test_df.loc[(mask_min_test) & (mask_max_test) & (nan_mask), 'home_' + col] = col_df.loc[mask_min_train & mask_max_train, ['home_' + col, 'away_' + col]].mean().mean()
            test_df.loc[(mask_min_test) & (mask_max_test) & (nan_mask), 'away_' + col] = col_df.loc[mask_min_train & mask_max_train, ['home_' + col, 'away_' + col]].mean().mean()   
    test_df.dropna(inplace=True)


def get_ids_for_test(df, prematch_odds=True, live_odds=True):
    if prematch_odds and live_odds:
        total_odds_mask = (df['live_odd_1'] != 0) & (df['odd_1'] != 0)
        id_partita_total_odds = df.loc[total_odds_mask, 'id_partita'].unique()
        if len(id_partita_total_odds) > len(df['id_partita'].unique()) // 4:
            id_partita_test = np.random.choice(id_partita_total_odds, size=len(df['id_partita'].unique()) // 4, replace=False)
        else:
            id_partita_test = id_partita_total_odds
    elif prematch_odds:
        prematch_odds_mask = df['odd_1'] != 0
        id_partita_prematch_odds = df.loc[prematch_odds_mask, 'id_partita'].unique()
        if len(id_partita_prematch_odds) > len(df['id_partita'].unique()) // 4:
            id_partita_test = np.random.choice(id_partita_prematch_odds, size=len(df['id_partita'].unique()) // 4, replace=False)
        else:
            id_partita_test = id_partita_prematch_odds
    elif live_odds:
        live_odds_mask = df['live_odd_1'] != 0
        id_partita_live_odds = df.loc[live_odds_mask, 'id_partita'].unique()
        if len(id_partita_live_odds) > len(df['id_partita'].unique()) // 4:
            id_partita_test = np.random.choice(id_partita_live_odds, size=len(df['id_partita'].unique()) // 4, replace=False)
        else:
            id_partita_test = id_partita_live_odds
    # splittare test and train in modo che nel train ci siano alcune partite, nel test altre
    else:
        id_partita_test = np.random.choice(df['id_partita'].unique(), size=len(df['id_partita'].unique()) // 4, replace=False)
    return id_partita_test


def add_outcome_col(df):
    df['final_uo'] = np.where(df['home_final_score'] + df['away_final_score'] > 2, 0, 1)


def drop_outcome_cols(df):
    df.drop(columns=['home_final_score', 'away_final_score', 'final_uo'], inplace=True)


def add_input_cols(df):
    df['actual_total_goals'] = df['home_score'] + df['away_score']
    df['over_strongness'] = (df['home_score'] + df['away_score']) * (90 - df['minute'])


def split_test_train(df, prematch_odds=True, live_odds=True, minute=80):
    id_partita_test = get_ids_for_test(df, prematch_odds=prematch_odds, live_odds=live_odds)
    test_mask = df['id_partita'].isin(id_partita_test)
    test = df.loc[test_mask, :].copy(deep=True)
    train = df.loc[~(test_mask), :].copy(deep=True)
    test = drop_easy_predictions(test, minute)
    return train, test


def drop_easy_predictions(test, minute=80):
    # drop too easy predictions
    under_minute_mask = test['minute'] <= minute
    return test.loc[under_minute_mask, :].copy()


def get_revenues(df, thresh=0.75):
    res = 0
    for _, row in df.iterrows():
        if row['probability_final_under'] >= thresh or row['probability_final_under'] <= 1-thresh:
            if row['final_uo'] == row['prediction_final_over']:
                if row['prediction_final'] == 'under':
                    if row['live_odd_under'] > 1:
                        res += (row['live_odd_under'] - 1)
                else:
                    if row['live_odd_over'] > 1:
                        res += (row['live_odd_over'] - 1)
            else:
                res -= 1
    return res


def get_insights(df):
    predictions_final = df['prediction_final_over']
    true_y = df['final_uo']
    acc = accuracy_score(true_y, predictions_final)
    print(f'Accuracy: {acc:.2f} \n')
    rev = get_revenues(df)
    print(f'Revenues: {rev:.2f} \n')
