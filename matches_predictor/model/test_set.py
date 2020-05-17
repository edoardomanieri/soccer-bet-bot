import numpy as np
from matches_predictor.model import base
import pandas as pd
import os


class Retrieving(base.Retrieving):

    @staticmethod
    def starting_df(cat_cols, api_missing_cols):
        file_path = os.path.dirname(os.path.abspath(__file__))
        # import dataset
        df_API = pd.read_csv(f"{file_path}/../../res/df_api.csv", index_col=0, header=0)
        # put on the API df all nans (will be dropped later on)
        for col in api_missing_cols:
            df_API[col] = np.nan
        df_scraping = pd.read_csv(f"{file_path}/../../res/df_scraping.csv", index_col=0, header=0)
        df = pd.concat([df_API, df_scraping], axis=0, ignore_index=True)
        # change data type
        for col in df.columns:
            if col not in cat_cols:
                df[col] = pd.to_numeric(df[col])
        return df.reset_index(drop=True)


class Preprocessing(base.Preprocessing):
    def __init__(self):
        pass

    @staticmethod
    def drop_nan(df, thresh='half'):
        # eliminate duplicate rows
        subset = [col for col in df.columns if col != 'minute']
        df.drop_duplicates(subset=subset, inplace=True)

        # eliminate rows with a lot of nans
        if thresh == 'half':
            thresh = len(df.columns) // 2
        df.dropna(axis=0, thresh=thresh, inplace=True)

        # eliminate rows with nans on target or on important columns
        important_cols = ['id_partita', 'minute',
                          'home_final_score', 'away_final_score']
        df.dropna(axis=0, subset=important_cols, how='any', inplace=True)

        # drop matches already in over
        over_mask = (df['home_score'] + df['away_score']) >= 3
        ids = df.loc[over_mask, 'id_partita'].unique()
        df.drop(df[df['id_partita'].isin(ids)].index, inplace=True)

    @staticmethod
    def add_prematch_vars(training_df, test_df):

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

    @staticmethod
    def pop_live_odds_data(df):
        live_odds_input = df.loc[:, [
            'id_partita', 'minute', 'live_odd_under', 'live_odd_over']].copy()
        df.drop(columns=['live_odd_1', 'live_odd_2', 'live_odd_X',
                         'live_odd_over', 'live_odd_under'], inplace=True)
        return live_odds_input

    @staticmethod
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
            col_df = train_df[(~train_df['home_' + col[5:]].isnull())
                              & (~train_df['away_' + col[5:]].isnull())]
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
                test_df.loc[(mask_min_test) & (mask_max_test) & (nan_mask), 'home_' +
                            col] = col_df.loc[mask_min_train & mask_max_train, ['home_' + col, 'away_' + col]].mean().mean()
                test_df.loc[(mask_min_test) & (mask_max_test) & (nan_mask), 'away_' +
                            col] = col_df.loc[mask_min_train & mask_max_train, ['home_' + col, 'away_' + col]].mean().mean()
        test_df.dropna(inplace=True)

    @staticmethod
    def drop_outcome_cols(df):
        df.drop(columns=['home_final_score',
                         'away_final_score', 'final_uo'], inplace=True)

    @staticmethod
    def execute(test_df, train_df, missing_cols):
        Preprocessing.drop_nan(test_df)
        Preprocessing.add_outcome_col(test_df)
        Preprocessing.drop_API_missing_cols(test_df, missing_cols)
        Preprocessing.impute_nan(train_df, test_df)
        Preprocessing.prematch_odds_to_prob(test_df)
        test_prematch_odds = Preprocessing.pop_prematch_odds_data(test_df)
        test_live_odds = Preprocessing.pop_live_odds_data(test_df)
        Preprocessing.add_input_cols(test_df)
        test_y = test_df.loc[:, ['id_partita', 'minute', 'final_uo']]
        Preprocessing.drop_outcome_cols(test_df)
        test_y = test_y.merge(test_df, on=['id_partita', 'minute']).loc[:, [
            'id_partita', 'minute', 'final_uo']].copy()
        return test_y, test_prematch_odds, test_live_odds
