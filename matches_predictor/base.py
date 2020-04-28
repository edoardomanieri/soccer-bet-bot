import numpy as np
import pandas as pd
import os


class Retrieving():
    pass


class Preprocessing():
    def __init__(self):
        pass

    @staticmethod
    def to_numeric(df, cat_col):
        # change data type
        for col in df.columns:
            if col not in cat_col:
                df[col] = pd.to_numeric(df[col])

    @staticmethod
    def one_hot_encoding(df, features_list):
        df_new = df.copy()
        for feature in features_list:
            df[feature] = df[feature].astype('category')
        for feature in features_list:
            df_new = pd.concat([df_new, pd.get_dummies(
                df[feature], drop_first=True, prefix=feature)], axis=1).copy()
        return df_new

    @staticmethod
    def normalize_prematch_odds(df):
        tmp = (1 - ((1 / df['odd_over']) +
                    (1 / df['odd_under']))) / 2
        df['odd_over'] = (1 / df['odd_over']) + tmp
        df['odd_under'] = (1 / df['odd_under']) + tmp

    @staticmethod
    def pop_live_odds_data(df):
        live_odds_input = df.loc[:, [
            'id_partita', 'minute', 'live_odd_under', 'live_odd_over']].copy()
        df.drop(columns=['live_odd_1', 'live_odd_2', 'live_odd_X',
                         'live_odd_over', 'live_odd_under'], inplace=True)
        return live_odds_input

    @staticmethod
    def add_input_cols(df):
        df['actual_total_goals'] = df['home_score'] + df['away_score']
        df['over_strongness'] = (
            df['home_score'] + df['away_score']) * (90 - df['minute'])

    @staticmethod
    def add_outcome_col(df):
        df['final_uo'] = np.where(
            df['home_final_score'] + df['away_final_score'] > 2, 1, 0)

    @staticmethod
    def pop_prematch_odds_data(df):
        prematch_odds_input = df.loc[:, [
            'id_partita', 'minute', 'odd_under', 'odd_over']].copy()
        df.drop(columns=['odd_1', 'odd_2', 'odd_X',
                         'odd_over', 'odd_under'], inplace=True)
        return prematch_odds_input

    @staticmethod
    def drop_odds_cols(df):
        df.drop(columns=['live_odd_over', 'live_odd_under',
                         'live_odd_1', 'live_odd_X', 'live_odd_2'], inplace=True)
        df.drop(columns=['odd_over', 'odd_under',
                         'odd_1', 'odd_X', 'odd_2'], inplace=True)

    @staticmethod
    def save(df):
        file_path = os.path.dirname(os.path.abspath(__file__))
        df.reset_index(drop=True).to_csv(
            f"{file_path}/../res/dataframes/training_goals.csv")


class Modeling():
    pass
