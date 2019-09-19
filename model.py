import pandas as pd
import numpy as np
import glob
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_training_df():
    #import dataset
    all_files = glob.glob("./csv/*.csv")
    li = [pd.read_csv(filename, index_col=None, header=0) for filename in all_files]
    df = pd.concat(li, axis=0, ignore_index=True)
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']

    #change data type
    for col in df.columns:
        if col not in cat_col:
            df[col] = pd.to_numeric(df[col])

    #dropna
    nan_col = ['home_rimesse_laterali', 'away_rimesse_laterali', 'home_tiri_fermati', 'away_tiri_fermati',\
        'home_punizioni', 'away_punizioni', 'home_passaggi_totali', 'away_passaggi_totali', 'home_passaggi_completati', 'away_passaggi_completati', 'home_contrasti', 'away_contrasti']
    df.drop(columns = nan_col, inplace = True)
    df.dropna(inplace = True)


    #adding outcome columns
    df['result'] = np.where(df['home_final_score'] > df['away_final_score'], 1, np.where(df['home_final_score'] == df['away_final_score'], 2, 3))
    df['final_total_goals'] = df['home_final_score'] + df['away_final_score']

    train['actual_result'] = np.where(train['home_score'] > train['away_score'], 1, np.where(train['home_score'] == train['away_score'], 2, 3))

    campionati = df['campionato'].unique()
    df['avg_camp_goals'] = 0

    df_matches = df[['home', 'away', 'campionato', 'home_final_score', 'away_final_score', 'id_partita', 'final_total_goals']].groupby('id_partita').first().reset_index()
    for camp in campionati:
        df.loc[df['campionato'] == camp,'avg_camp_goals'] = df_matches.loc[df_matches['campionato'] == camp,'final_total_goals'].mean()

    df['home_avg_goal_fatti'] = 0
    df['away_avg_goal_fatti'] = 0

    df['home_avg_goal_subiti'] = 0
    df['away_avg_goal_subiti'] = 0

    squadre = set((df['home'].unique().tolist() + df['away'].unique().tolist()))

    for team in squadre:
        n_match_home = len(df_matches[df_matches['home'] == team])
        n_match_away = len(df_matches[df_matches['away'] == team])

        sum_home_fatti = df_matches.loc[(df_matches['home'] == team),'home_final_score'].sum()
        sum_away_fatti = df_matches.loc[(df_matches['away'] == team),'away_final_score'].sum()

        #divide by 0
        if (n_match_home + n_match_away) == 0:
            n_match_away += 1

        df.loc[df['home'] == team,'home_avg_goal_fatti'] = (sum_home_fatti + sum_away_fatti) / (n_match_home + n_match_away)
        df.loc[df['away'] == team,'away_avg_goal_fatti'] = (sum_home_fatti + sum_away_fatti) / (n_match_home + n_match_away)

        sum_home_subiti = df_matches.loc[(df_matches['home'] == team),'away_final_score'].sum()
        sum_away_subiti = df_matches.loc[(df_matches['away'] == team),'home_final_score'].sum()

        df.loc[df['home'] == team,'home_avg_goal_subiti'] = (sum_home_subiti + sum_away_subiti) / (n_match_home + n_match_away)
        df.loc[df['away'] == team,'away_avg_goal_subiti'] = (sum_home_subiti + sum_away_subiti) / (n_match_home + n_match_away)

    df.drop(columns = cat_col, inplace = True)

    #tmp_y_col_to_be_dropped = ['home_avg_goal_fatti', 'away_avg_goal_fatti', 'home_avg_goal_subiti', 'away_avg_goal_subiti', 'avg_camp_goals']
    # train_X = train_X.drop(columns = tmp_y_col_to_be_dropped)
    return df

def get_model():
    drop_y_column = ['home_final_score', 'away_final_score', 'result', 'final_total_goals']
    train_y = train['result'].values
    train_X = train.drop(columns = drop_y_column)

def process_test_data(training_df, test_df):
    #introduce target variables
    test_df['result'] = np.where(test_df['home_final_score'] > test_df['away_final_score'], 1, np.where(test_df['home_final_score'] == test_df['away_final_score'], 2, 3))
    test_df['final_total_goals'] = test_df['home_final_score'] + test_df['away_final_score']


    test_df['actual_result'] = np.where(test_df['home_score'] > test_df['away_score'], 1, np.where(test_df['home_score'] == test_df['away_score'], 2, 3))
    test_df['result_strongness'] = (test_df['home_score'] - test_df['away_score']) * test_df['minute']
    
    test_df['avg_camp_goals'] = 0
    campionati = test_df['campionato'].unique()

    for camp in campionati:
        if camp not in training_df['campionato'].unique():
            test_df.loc[test_df['campionato'] == camp,'avg_camp_goals'] = training_df['avg_camp_goals'].mean()
        else:
            test_df.loc[test_df['campionato'] == camp,'avg_camp_goals'] = training_df.loc[training_df['campionato'] == camp,:].reset_index()['avg_camp_goals'][0]

    test_df['home_avg_goal_fatti'] = 0
    test_df['away_avg_goal_fatti'] = 0

    test_df['home_avg_goal_subiti'] = 0
    test_df['away_avg_goal_subiti'] = 0

    squadre = set((test_df['home'].unique().tolist() + test_df['away'].unique().tolist()))
    for team in squadre:
        if team not in training_df['home'].unique() or team not in training_df['away'].unique():
            test_df.loc[test_df['home'] == team,'home_avg_goal_fatti'] = training_df['home_avg_goal_fatti'].mean()
            test_df.loc[test_df['away'] == team,'away_avg_goal_fatti'] = training_df['away_avg_goal_fatti'].mean()
            test_df.loc[test_df['home'] == team,'home_avg_goal_subiti'] = training_df['home_avg_goal_subiti'].mean()
            test_df.loc[test_df['away'] == team,'away_avg_goal_subiti'] = training_df['away_avg_goal_subiti'].mean()
        else:
            test_df.loc[test_df['home'] == team,'home_avg_goal_fatti'] = training_df.loc[training_df['home'] == team,:].reset_index()['home_avg_goal_fatti'][0]
            test_df.loc[test_df['away'] == team,'away_avg_goal_fatti'] = training_df.loc[training_df['away'] == team,:].reset_index()['away_avg_goal_fatti'][0]
            test_df.loc[test_df['home'] == team,'home_avg_goal_subiti'] = training_df.loc[training_df['home'] == team,:].reset_index()['home_avg_goal_subiti'][0]
            test_df.loc[test_df['away'] == team,'away_avg_goal_subiti'] = training_df.loc[training_df['away'] == team,:].reset_index()['away_avg_goal_subiti'][0]
    
    return test_df



def process_input_data(test_df):
    training_df = get_training_df()
    #introduce target variables
    test_df['avg_camp_goals'] = 0
    campionati = test_df['campionato'].unique()
    
    test_df['actual_result'] = np.where(test_df['home_score'] > test_df['away_score'], 1, np.where(test_df['home_score'] == test_df['away_score'], 2, 3))
    test_df['result_strongness'] = (test_df['home_score'] - test_df['away_score']) * test_df['minute']

    for camp in campionati:
        if camp not in training_df['campionato'].unique():
            test_df.loc[test_df['campionato'] == camp,'avg_camp_goals'] = training_df['avg_camp_goals'].mean()
        else:
            test_df.loc[test_df['campionato'] == camp,'avg_camp_goals'] = training_df.loc[training_df['campionato'] == camp,:].reset_index()['avg_camp_goals'][0]

    test_df['home_avg_goal_fatti'] = 0
    test_df['away_avg_goal_fatti'] = 0

    test_df['home_avg_goal_subiti'] = 0
    test_df['away_avg_goal_subiti'] = 0

    squadre = set((test_df['home'].unique().tolist() + test_df['away'].unique().tolist()))
    for team in squadre:
        if team not in training_df['home'].unique() or team not in training_df['away'].unique():
            test_df.loc[test_df['home'] == team,'home_avg_goal_fatti'] = training_df['home_avg_goal_fatti'].mean()
            test_df.loc[test_df['away'] == team,'away_avg_goal_fatti'] = training_df['away_avg_goal_fatti'].mean()
            test_df.loc[test_df['home'] == team,'home_avg_goal_subiti'] = training_df['home_avg_goal_subiti'].mean()
            test_df.loc[test_df['away'] == team,'away_avg_goal_subiti'] = training_df['away_avg_goal_subiti'].mean()
        else:
            test_df.loc[test_df['home'] == team,'home_avg_goal_fatti'] = training_df.loc[training_df['home'] == team,:].reset_index()['home_avg_goal_fatti'][0]
            test_df.loc[test_df['away'] == team,'away_avg_goal_fatti'] = training_df.loc[training_df['away'] == team,:].reset_index()['away_avg_goal_fatti'][0]
            test_df.loc[test_df['home'] == team,'home_avg_goal_subiti'] = training_df.loc[training_df['home'] == team,:].reset_index()['home_avg_goal_subiti'][0]
            test_df.loc[test_df['away'] == team,'away_avg_goal_subiti'] = training_df.loc[training_df['away'] == team,:].reset_index()['away_avg_goal_subiti'][0]

    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    test_df.drop(columns = cat_col, inplace = True)

    nan_col = ['home_rimesse_laterali', 'away_rimesse_laterali', 'home_tiri_fermati', 'away_tiri_fermati',\
            'home_punizioni', 'away_punizioni', 'home_passaggi_totali', 'away_passaggi_totali', 'home_passaggi_completati', 'away_passaggi_completati', 'home_contrasti', 'away_contrasti']
    test_df.drop(columns = nan_col, inplace = True)
    test_df.dropna(inplace = True)
    
    return test_df


def nan_imputation(train_df, test_df, col):

    if 'away' in col:
        return test_df

    col = col[5:]

    nan_mask = test_df['home_' + col].isnull() | test_df['away_' + col].isnull()

    if "possesso_palla" in col:
        test_df.loc[nan_mask, 'home_possesso_palla'] = 50
        test_df.loc[nan_mask, 'away_possesso_palla'] = 50
        return test_df

    for m in np.arange(5, 90, 5):
        mask_min_test = test_df['minute'] >= m
        mask_max_test = test_df['minute'] <= m + 5
        mask_min_train = train_df['minute'] >= m
        mask_max_train = train_df['minute'] <= m + 5
        test_df.loc[(mask_min_test) & (mask_max_test) & (nan_mask), 'home_' + col] = train_df.loc[mask_min_train & mask_max_train, ['home_' + col, 'away_' + col]].mean().mean()
        test_df.loc[(mask_min_test) & (mask_max_test) & (nan_mask), 'away_' + col] = train_df.loc[mask_min_train & mask_max_train, ['home_' + col, 'away_' + col]].mean().mean()

    return test_df