import pandas as pd
import numpy as np


def _drop_nan(df, thresh='half'):
    # eliminate duplicate rows
    subset = [col for col in df.columns if col != 'minute']
    df.drop_duplicates(subset=subset, inplace=True)

    # eliminate rows with a lot of nans
    if thresh == 'half':
        thresh = len(df.columns) // 2
    df.dropna(axis=0, thresh=thresh, inplace=True)

    # eliminate rows with nans on target or on important columns
    important_cols = ['id_partita', 'minute']
    df.dropna(axis=0, subset=important_cols, how='any', inplace=True)

    # drop matches already in over
    over_mask = (df['home_score'] + df['away_score']) >= 3
    ids = df.loc[over_mask, 'id_partita'].unique()
    df.drop(df[df['id_partita'].isin(ids)].index, inplace=True)


def _normalize_prematch_odds(input_df):
    tmp = (1 - ((1 / input_df['odd_over']) + (1 / input_df['odd_under']))) / 2
    input_df['odd_over'] = (1 / input_df['odd_over']) + tmp
    input_df['odd_under'] = (1 / input_df['odd_under']) + tmp


def _pop_prematch_odds_data(input_df):
    prematch_odds_input = input_df.loc[:, [
        'id_partita', 'minute', 'odd_under', 'odd_over']].copy()
    input_df.drop(columns=['odd_1', 'odd_2', 'odd_X',
                           'odd_over', 'odd_under'], inplace=True)
    return prematch_odds_input


def _pop_live_odds_data(input_df):
    live_odds_input = input_df.loc[:, [
        'id_partita', 'minute', 'live_odd_under', 'live_odd_over']].copy()
    input_df.drop(columns=['live_odd_1', 'live_odd_2', 'live_odd_X',
                           'live_odd_over', 'live_odd_under'], inplace=True)
    return live_odds_input


def _to_numeric(df, cat_col):
    # change data type
    for col in df.columns:
        if col not in cat_col:
            df[col] = pd.to_numeric(df[col])


def _drop_outcome_cols(df):
    df.drop(columns=['home_final_score', 'away_final_score'], inplace=True)


def _add_input_cols(df):
    df['actual_total_goals'] = df['home_score'] + df['away_score']
    df['over_strongness'] = (
        df['home_score'] + df['away_score']) * (90 - df['minute'])


def _impute_nan(train_df, test_df, thresh='half'):
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


def _add_prematch_vars(training_df, test_df):

    test_df['avg_camp_goals'] = 0
    campionati = test_df['campionato'].unique()

    for camp in campionati:
        if camp not in training_df['campionato'].unique():
            test_df.loc[test_df['campionato'] == camp,
                        'avg_camp_goals'] = training_df['avg_camp_goals'].mean()
        else:
            test_df.loc[test_df['campionato'] == camp, 'avg_camp_goals'] = training_df.loc[training_df['campionato'] == camp, :].reset_index()[
                'avg_camp_goals'][0]

    test_df['home_avg_goal_fatti'] = 0
    test_df['away_avg_goal_fatti'] = 0
    test_df['home_avg_goal_subiti'] = 0
    test_df['away_avg_goal_subiti'] = 0

    squadre = set((test_df['home'].unique().tolist() +
                   test_df['away'].unique().tolist()))
    for team in squadre:
        if team not in training_df['home'].unique() or team not in training_df['away'].unique():
            test_df.loc[test_df['home'] == team,
                        'home_avg_goal_fatti'] = training_df['home_avg_goal_fatti'].mean()
            test_df.loc[test_df['away'] == team,
                        'away_avg_goal_fatti'] = training_df['away_avg_goal_fatti'].mean()
            test_df.loc[test_df['home'] == team,
                        'home_avg_goal_subiti'] = training_df['home_avg_goal_subiti'].mean()
            test_df.loc[test_df['away'] == team,
                        'away_avg_goal_subiti'] = training_df['away_avg_goal_subiti'].mean()
        else:
            test_df.loc[test_df['home'] == team, 'home_avg_goal_fatti'] = training_df.loc[training_df['home'] == team, :].reset_index()[
                'home_avg_goal_fatti'][0]
            test_df.loc[test_df['away'] == team, 'away_avg_goal_fatti'] = training_df.loc[training_df['away'] == team, :].reset_index()[
                'away_avg_goal_fatti'][0]
            test_df.loc[test_df['home'] == team, 'home_avg_goal_subiti'] = training_df.loc[training_df['home'] == team, :].reset_index()[
                'home_avg_goal_subiti'][0]
            test_df.loc[test_df['away'] == team, 'away_avg_goal_subiti'] = training_df.loc[training_df['away'] == team, :].reset_index()[
                'away_avg_goal_subiti'][0]


def execute(input_df, train_df, cat_col):
    _to_numeric(input_df, cat_col)
    _drop_nan(input_df)
    _impute_nan(train_df, input_df)
    _normalize_prematch_odds(input_df)
    input_prematch_odds = _pop_prematch_odds_data(input_df)
    input_live_odds = _pop_live_odds_data(input_df)
    _drop_outcome_cols(input_df)
    _add_prematch_vars(train_df, input_df)
    _add_input_cols(input_df)
    return input_prematch_odds
