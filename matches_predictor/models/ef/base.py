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
    def drop_API_missing_cols(df, missing_cols):
        if missing_cols[0] in df.columns:
            df.drop(columns=missing_cols, inplace=True)

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
    def _calc_smooth_mean(df, by, m):
        # Compute the global mean
        mean = df['final_ef'].mean()
        # Compute the number of values and the mean of each group
        agg = df.groupby(by)['final_ef'].agg(['count', 'mean']).reset_index()
        agg['smooth'] = (agg['count'] * agg['mean'] + m * mean) / (agg['count'] + m)
        d = pd.Series(agg['smooth'].values, index=agg[by]).to_dict()
        # Replace each value by the according smoothed mean
        return d, mean

    @staticmethod
    def smooth_handling(df_train, df_test, cat_vars, m=10):
        for var in cat_vars:
            replace_dict, mean = Preprocessing._calc_smooth_mean(df_train, by=var, m=10)
            df_train = df_train.replace(replace_dict)
            df_test.loc[:, var] = np.where(df_test.loc[:, var].isin(
                replace_dict.keys()), df_test.loc[:, var].map(replace_dict), mean).astype(float)

    @staticmethod
    def prematch_odds_to_prob(df):
        tmp = (1 - ((1 / df['odd_1']) +
                    (1 / df['odd_2']) +
                    (1 / df['odd_X']))) / 3
        df['odd_1'] = (1 / df['odd_1']) + tmp
        df['odd_X'] = (1 / df['odd_X']) + tmp
        df['odd_2'] = (1 / df['odd_2']) + tmp

    @staticmethod
    def pop_live_odds_uo(df):
        if 'live_odd_1' in df.columns:
            df.drop(columns=['live_odd_1', 'live_odd_2', 'live_odd_X',
                             'live_odd_over', 'live_odd_under'], inplace=True)

    @staticmethod
    def add_input_cols(df):
        df['actual_total_goals'] = df['home_score'] + df['away_score']
        df['result_strongness'] = (df['home_score'] - df['away_score']) * df['minute']

    @staticmethod
    def add_outcome_col(df):
        df['final_ef'] = np.where(df['home_final_score'] > df['away_final_score'], 1, np.where(
            df['home_final_score'] == df['away_final_score'], 2, 3))

    @staticmethod
    def pop_prematch_odds_data(df):
        prematch_odds_input = df.loc[:, [
            'id_partita', 'minute', 'odd_1', 'odd_X', 'odd_2']]
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
            f"{file_path}/../../../res/dataframes/training_ef.csv")


class Modeling():
    pass
