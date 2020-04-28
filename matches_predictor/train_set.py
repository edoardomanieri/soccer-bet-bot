from matches_predictor import base
import numpy as np
import pandas as pd
import glob
import os
import joblib


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
    def starting_df(res_path):
        file_path = os.path.dirname(os.path.abspath(__file__))
        # import dataset
        all_files = sorted(glob.glob(f"{file_path}/{res_path}/*.csv"),
                           key=lambda x: int(x[x.index('stats') + 5:-4]))
        li = [pd.read_csv(filename, index_col=None, header=0)
              for filename in all_files[:-1]]
        df = pd.concat(li, axis=0, ignore_index=True)
        return df.reset_index(drop=True)


class Preprocessing(base.Preprocessing):
    def __init__(self):
        pass

    @staticmethod
    def impute_nan(df, thresh='half'):
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
        important_cols = ['home_final_score',
                          'minute', 'away_final_score', 'id_partita']
        df.dropna(axis=0, subset=important_cols, how='any', inplace=True)

        # drop matches already in over
        over_mask = (df['home_score'] + df['away_score']) >= 3
        df.drop(df[over_mask].index, inplace=True)

    @staticmethod
    def add_prematch_vars(df):
        df['avg_camp_goals'] = 0
        campionati = df['campionato'].unique()

        for camp in campionati:
            if camp not in df['campionato'].unique():
                df.loc[df['campionato'] == camp,
                       'avg_camp_goals'] = df['avg_camp_goals'].mean()
            else:
                df.loc[df['campionato'] == camp, 'avg_camp_goals'] = df.loc[df['campionato']
                                                                            == camp, :].reset_index()['avg_camp_goals'][0]

        df['home_avg_goal_fatti'] = 0
        df['away_avg_goal_fatti'] = 0
        df['home_avg_goal_subiti'] = 0
        df['away_avg_goal_subiti'] = 0

        squadre = set((df['home'].unique().tolist() +
                       df['away'].unique().tolist()))
        for team in squadre:
            if team not in df['home'].unique() or team not in df['away'].unique():
                df.loc[df['home'] == team,
                       'home_avg_goal_fatti'] = df['home_avg_goal_fatti'].mean()
                df.loc[df['away'] == team,
                       'away_avg_goal_fatti'] = df['away_avg_goal_fatti'].mean()
                df.loc[df['home'] == team,
                       'home_avg_goal_subiti'] = df['home_avg_goal_subiti'].mean()
                df.loc[df['away'] == team,
                       'away_avg_goal_subiti'] = df['away_avg_goal_subiti'].mean()
            else:
                df.loc[df['home'] == team, 'home_avg_goal_fatti'] = df.loc[df['home']
                                                                           == team, :].reset_index()['home_avg_goal_fatti'][0]
                df.loc[df['away'] == team, 'away_avg_goal_fatti'] = df.loc[df['away']
                                                                           == team, :].reset_index()['away_avg_goal_fatti'][0]
                df.loc[df['home'] == team, 'home_avg_goal_subiti'] = df.loc[df['home']
                                                                            == team, :].reset_index()['home_avg_goal_subiti'][0]
                df.loc[df['away'] == team, 'away_avg_goal_subiti'] = df.loc[df['away']
                                                                            == team, :].reset_index()['away_avg_goal_subiti'][0]

    ########################### main function #############
    @staticmethod
    def execute(train_df, cat_col, prod=True):
        Preprocessing.to_numeric(train_df, cat_col)
        Preprocessing.drop_odds_cols(train_df)
        Preprocessing.drop_nan(train_df)
        Preprocessing.impute_nan(train_df)
        Preprocessing.add_outcome_col(train_df)
        Preprocessing.add_prematch_vars(train_df)
        Preprocessing.add_input_cols(train_df)
        if prod:
            Preprocessing.save(train_df)


class Modeling(base.Modeling):

    @staticmethod
    def train_model(train_df, clf, cat_col, outcome_cols, prod=True):
        """
        Create model and save it with joblib
        """
        train_y = train_df['final_uo'].values
        to_drop = cat_col + outcome_cols
        train_X = train_df.drop(columns=to_drop)
        clf.fit(train_X, train_y)
        file_path = os.path.dirname(os.path.abspath(__file__))
        prod_path = "production" if prod else "development"
        path = f"{file_path}/../res/models/{prod_path}/goals.joblib"
        joblib.dump(clf, path)

    @staticmethod
    def save_model(clf):
        file_path = os.path.dirname(os.path.abspath(__file__))
        path = f"{file_path}/../res/models/development/goals.joblib"
        joblib.dump(clf, path)

    @staticmethod
    def get_prod_model():
        file_path = os.path.dirname(os.path.abspath(__file__))
        path = f"{file_path}/../res/models/production/goals.joblib"
        return joblib.load(path)

    @staticmethod
    def get_dev_model():
        file_path = os.path.dirname(os.path.abspath(__file__))
        path = f"{file_path}/../res/models/development/goals.joblib"
        return joblib.load(path)
