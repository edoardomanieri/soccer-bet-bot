import pandas as pd
import numpy as np
import glob
import os
import time
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

def nan_imputation(train_df, test_df, thresh = 'half'):

    #eliminate rows with a lot of nans
    if thresh == 'half':
        thresh = len(test_df.columns) // 2
    test_df.dropna(axis = 0, thresh = thresh, inplace = True)

    #imputing the other nans
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
    
    test_df.dropna(inplace = True)
    return test_df


def get_input_data():
    all_files = glob.glob("../csv/*.csv")
    input_df = pd.read_csv(all_files[-1], index_col=None, header=0)
    return input_df.sort_values(by = ['id_partita', 'minute'], ascending = [True, False]).groupby(['id_partita']).first().reset_index() 



def get_training_df():
    #import dataset
    all_files = glob.glob("../csv/*.csv")
    li = [pd.read_csv(filename, index_col=None, header=0) for filename in all_files[:-1]]
    df = pd.concat(li, axis=0, ignore_index=True)
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']

    #change data type
    for col in df.columns:
        if col not in cat_col:
            df[col] = pd.to_numeric(df[col])

    #nan imputation
    df = nan_imputation(df,df)

    #adding outcome columns
    df['result'] = np.where(df['home_final_score'] > df['away_final_score'], 1, np.where(df['home_final_score'] == df['away_final_score'], 2, 3))
    df['final_total_goals'] = df['home_final_score'] + df['away_final_score']


    #input columns
    df['actual_total_goals'] = df['home_score'] + df['away_score']
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

    #tmp_y_col_to_be_dropped = ['home_avg_goal_fatti', 'away_avg_goal_fatti', 'home_avg_goal_subiti', 'away_avg_goal_subiti', 'avg_camp_goals']
    #train_X = train_X.drop(columns = tmp_y_col_to_be_dropped)
    return df.reset_index(drop = True)


def process_input_data(input_df, training_df):

    if 'home_final_score' in input_df.columns and 'away_final_score' in input_df.columns:
        input_df.drop(columns = ['home_final_score', 'away_final_score'], inplace = True)

    #introduce target variables
    campionati = input_df['campionato'].unique()
    
    input_df['actual_total_goals'] = input_df['home_score'] + input_df['away_score']
    input_df['avg_camp_goals'] = 0

    for camp in campionati:
        if camp not in training_df['campionato'].unique():
            input_df.loc[input_df['campionato'] == camp,'avg_camp_goals'] = training_df['avg_camp_goals'].mean()
        else:
            input_df.loc[input_df['campionato'] == camp,'avg_camp_goals'] = training_df.loc[training_df['campionato'] == camp,:].reset_index()['avg_camp_goals'][0]

    input_df['home_avg_goal_fatti'] = 0
    input_df['away_avg_goal_fatti'] = 0
    input_df['home_avg_goal_subiti'] = 0
    input_df['away_avg_goal_subiti'] = 0

    squadre = set((input_df['home'].unique().tolist() + input_df['away'].unique().tolist()))
    for team in squadre:
        if team not in training_df['home'].unique() or team not in training_df['away'].unique():
            input_df.loc[input_df['home'] == team,'home_avg_goal_fatti'] = training_df['home_avg_goal_fatti'].mean()
            input_df.loc[input_df['away'] == team,'away_avg_goal_fatti'] = training_df['away_avg_goal_fatti'].mean()
            input_df.loc[input_df['home'] == team,'home_avg_goal_subiti'] = training_df['home_avg_goal_subiti'].mean()
            input_df.loc[input_df['away'] == team,'away_avg_goal_subiti'] = training_df['away_avg_goal_subiti'].mean()
        else:
            input_df.loc[input_df['home'] == team,'home_avg_goal_fatti'] = training_df.loc[training_df['home'] == team,:].reset_index()['home_avg_goal_fatti'][0]
            input_df.loc[input_df['away'] == team,'away_avg_goal_fatti'] = training_df.loc[training_df['away'] == team,:].reset_index()['away_avg_goal_fatti'][0]
            input_df.loc[input_df['home'] == team,'home_avg_goal_subiti'] = training_df.loc[training_df['home'] == team,:].reset_index()['home_avg_goal_subiti'][0]
            input_df.loc[input_df['away'] == team,'away_avg_goal_subiti'] = training_df.loc[training_df['away'] == team,:].reset_index()['away_avg_goal_subiti'][0]


    input_df = nan_imputation(training_df, input_df)
    
    return input_df


#drop_y_column = ['home_final_score', 'away_final_score', 'result', 'final_total_goals']
def train_and_save_model(train):
    """
    Create model and save it with joblib
    """
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'result']
    train_y = train['final_total_goals'].values
    train_X = train.drop(columns = ['final_total_goals'] + cat_col + outcome_cols)

    xgb = XGBRegressor(n_estimators = 2000)
    xgb.fit(train_X, train_y)
    joblib.dump(xgb, "./models/goals.joblib")


def get_model():
    return joblib.load("./models/goals.joblib")


def get_predict(model, input_df):
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    predictions = model.predict(input_df.drop(columns = cat_col))
    input_df['predictions'] = predictions
    input_df.loc[input_df['predictions'] < input_df['actual_total_goals'],'predictions'] = input_df['actual_total_goals']
    predictions = input_df['predictions'].values
    return predictions



def get_complete_predictions_table(input_df,predictions):
    input_df['predictions'] = predictions
    final_df = input_df.loc[:, ['id_partita', 'home', 'away', 'minute', 'home_score','away_score','predictions']]\
             .sort_values(by = ['minute'], ascending = False, inplace = False)
    return final_df

