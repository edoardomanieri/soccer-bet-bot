import numpy as np
from matches_predictor.models.ef import base
import pandas as pd
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


class Retrieving(base.Retrieving):

    @staticmethod
    def starting_df(cat_cols, api_missing_cols):
        file_path = os.path.dirname(os.path.abspath(__file__))
        # import dataset
        df_temp = pd.read_csv(f"{file_path}/../../res/temp.csv", index_col=0, header=0)
        # change data type
        for col in df_temp.columns:
            if col not in cat_cols:
                df_temp[col] = pd.to_numeric(df_temp[col])
        return df_temp.reset_index(drop=True)


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
        important_cols = ['id_partita', 'minute']
        df.dropna(axis=0, subset=important_cols, how='any', inplace=True)


    @staticmethod
    def impute_nan(train_df, test_df):
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
        nan_cols = [i for i in test_df.columns if test_df[i].isnull().any()]
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

    @staticmethod
    def add_prematch_vars(training_df, test_df):
        test_df['avg_camp_goals'] = 0
        campionati = test_df['campionato'].unique()

        for camp in campionati:
            if camp not in training_df['campionato'].unique():
                test_df.loc[test_df['campionato'] == camp, 'avg_camp_goals'] = training_df['avg_camp_goals'].mean()
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
    def drop_outcome_cols(df):
        df.drop(columns=['home_final_score',
                         'away_final_score'], inplace=True)

    @staticmethod
    def execute(input_df, train_df, cat_cols):
        Preprocessing.to_numeric(input_df, cat_cols)
        Preprocessing.impute_nan(train_df, input_df)
        Preprocessing.prematch_odds_to_prob(input_df)
        input_prematch_odds = Preprocessing.pop_prematch_odds_data(input_df)
        Preprocessing.smooth_handling(train_df, input_df, ['campionato'])
        Preprocessing.add_input_cols(input_df)
        return input_prematch_odds
