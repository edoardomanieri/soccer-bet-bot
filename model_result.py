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
import joblib
import utils


def get_input_data():
    all_files = sorted(glob.glob("../csv/*.csv"), key = lambda x: int(x[12:-4]))
    input_df = pd.read_csv(all_files[-1], index_col=None, header=0)
    return input_df.sort_values(by = ['id_partita', 'minute'], ascending = [True, False]).groupby(['id_partita']).first().reset_index() 

def pop_input_odds_data(input_df):
    odds_input = input_df.loc[:,['id_partita', 'minute', 'odd_1', 'odd_X', 'odd_2']]
    input_df.drop(columns=['id_partita', 'minute', 'odd_1', 'odd_X', 'odd_2'], inplace = True)
    return odds_input

def normalize_odds(input_df):
    tmp = (1 - ((1 / input_df['odd_1']) + (1 / input_df['odd_X']) + (1 / input_df['odd_2']))) / 3
    input_df['odd_1'] =  (1 / input_df['odd_1']) + tmp  
    input_df['odd_X'] = (1 / input_df['odd_X']) + tmp  
    input_df['odd_2'] = (1 / input_df['odd_2']) + tmp  
    return input_df


def drop_odds_col(df):
    return df.drop(columns = ['odd_over', 'odd_under','odd_1', 'odd_X', 'odd_2'])

def get_training_df():
    #import dataset
    all_files = sorted(glob.glob("../csv/*.csv"), key = lambda x: int(x[12:-4]))
    li = [pd.read_csv(filename, index_col=None, header=0) for filename in all_files[:-1]]
    df = pd.concat(li, axis=0, ignore_index=True)
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']

    #drop odds variables
    df = drop_odds_col(df)

    #change data type
    for col in df.columns:
        if col not in cat_col:
            df[col] = pd.to_numeric(df[col])

    #nan imputation
    df = utils.nan_imputation(df,df)

    #adding outcome columns
    df['result'] = np.where(df['home_final_score'] > df['away_final_score'], 1, np.where(df['home_final_score'] == df['away_final_score'], 2, 3))
    df['final_total_goals'] = df['home_final_score'] + df['away_final_score']

    df['actual_result'] = np.where(df['home_score'] > df['away_score'], 1, np.where(df['home_score'] == df['away_score'], 2, 3))
    df['result_strongness'] = (df['home_score'] - df['away_score']) * df['minute']

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

    df.reset_index(drop = True).to_csv("../dfs/training_result.csv")
    return df.reset_index(drop = True)


def process_input_data(input_df, training_df):

    if 'home_final_score' in input_df.columns and 'away_final_score' in input_df.columns:
        input_df.drop(columns = ['home_final_score', 'away_final_score'], inplace = True)

    input_df = utils.nan_imputation(training_df, input_df)

    #introduce target variables
    campionati = input_df['campionato'].unique()
    
    input_df['actual_result'] = np.where(input_df['home_score'] > input_df['away_score'], 1, np.where(input_df['home_score'] == input_df['away_score'], 2, 3))
    input_df['result_strongness'] = (input_df['home_score'] - input_df['away_score']) * input_df['minute']
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
    
    return input_df


#drop_y_column = ['home_final_score', 'away_final_score', 'result', 'final_total_goals']
def train_and_save_model(train):
    """
    Create model and save it with joblib
    """
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_total_goals']
    train_y = train['result'].values
    train_X = train.drop(columns = ['result'] + cat_col + outcome_cols)

    xgb = XGBClassifier(n_estimators = 2000)
    xgb.fit(train_X, train_y)
    joblib.dump(xgb, "./models/result.joblib")


def get_model():
    return joblib.load("./models/result.joblib")


def get_predict_proba(model, input_df):
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    predictions = model.predict(input_df.drop(columns = cat_col))
    probabilities = model.predict_proba(input_df.drop(columns = cat_col))
    return predictions, probabilities


def get_predictions_table_nodraws(input_df,predictions,probabilities,threshold = 0.9):
    #deprecated
    input_df['predictions'] = predictions
    input_df['probability'] = np.max(probabilities, axis = 1) 
    prob_mask = input_df['probability'] >= threshold
    minute_max_mask = input_df['minute'] < 60
    minute_min_mask = input_df['minute'] > 20
    score_mask = input_df['home_score'] == input_df['away_score']
    no_draws_mask = input_df['predictions'] != 2
    final_df = input_df.loc[(prob_mask & minute_max_mask & minute_min_mask & score_mask & no_draws_mask),\
         ['home', 'away', 'minute', 'home_score', 'away_score','probability', 'predictions']]\
             .sort_values(by = 'probability', ascending = False, inplace = False)
    return final_df

def get_complete_predictions_table_old(input_df,predictions,probabilities,threshold = 0.5):
    #deprecated
    input_df['predictions'] = predictions
    input_df['probability_result'] = np.max(probabilities, axis = 1)
    prob_mask = input_df['probability'] >= threshold
    final_df = input_df.loc[prob_mask, ['id_partita','home', 'away', 'minute', 'home_score',\
         'away_score', 'probability','predictions']]\
             .sort_values(by = ['probability', 'minute'], ascending = False, inplace = False)
    return final_df

def get_complete_predictions_table(input_df,predictions,probabilities,threshold = 0.5):
    input_df['predictions'] = predictions
    input_df['probability_1'] = probabilities[0]
    input_df['probability_X'] = probabilities[1]
    input_df['probability_2'] = probabilities[2]
    prob_mask = input_df['probability'] >= threshold
    final_df = input_df.loc[prob_mask, ['id_partita','home', 'away', 'minute', 'home_score',\
         'away_score', 'probability_1','probability_X', 'probability_2' 'predictions']]\
             .sort_values(by = ['minute'], ascending = False, inplace = False)
    return final_df

def get_prior_posterior_predictions(input_pred_df, input_odds_df):
    rate = 0.5 / 90
    res_df = input_pred_df.merge(input_odds_df, on = ['id_partita', 'minute'])
    res_df['probability_final_result_1'] = ((0.5 + (rate*res_df['minute'])) * res_df['probability_1'])\
                                         * ((0.5 - (rate*res_df['minute'])) * res_df['odd_1'])
    res_df['probability_final_result_X'] = ((0.5 + (rate*res_df['minute'])) * res_df['probability_X'])\
                                         * ((0.5 - (rate*res_df['minute'])) * res_df['odd_X'])
    res_df['probability_final_result_2'] = ((0.5 + (rate*res_df['minute'])) * res_df['probability_2'])\
                                         * ((0.5 - (rate*res_df['minute'])) * res_df['odd_2'])
    res_df['prediction_final_result'] = np.argmax(res_df[['probability_final_result_1','probability_final_result_X', 'probability_final_result_2']], axis = 1) + 1
    return res_df                    
