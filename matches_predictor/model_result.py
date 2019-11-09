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
    file_path = os.path.dirname(os.path.abspath(__file__))
    all_files = sorted(glob.glob(file_path + "/../csv/*.csv"), key=lambda x: int(x[15:-4]))
    input_df = pd.read_csv(all_files[-1], index_col=None, header=0)
    if 'Unnamed: 0' in input_df.columns:
        input_df.drop(columns = ['Unnamed: 0'], inplace = True)
    return input_df.sort_values(by = ['id_partita', 'minute'], ascending = [True, False]).groupby(['id_partita']).first().reset_index() 

def pop_input_odds_data(input_df):
    odds_input = input_df.loc[:,['id_partita', 'minute', 'odd_1', 'odd_X', 'odd_2']]
    input_df.drop(columns=['odd_1', 'odd_X', 'odd_2', 'odd_under', 'odd_over'], inplace = True)
    return odds_input

def normalize_odds(input_df):
    tmp = (1 - ((1 / input_df['odd_1']) + (1 / input_df['odd_X']) + (1 / input_df['odd_2']))) / 3
    input_df['odd_1'] = (1 / input_df['odd_1']) + tmp
    input_df['odd_X'] = (1 / input_df['odd_X']) + tmp
    input_df['odd_2'] = (1 / input_df['odd_2']) + tmp
    return input_df


def drop_odds_col(df):
    return df.drop(columns=['odd_over', 'odd_under','odd_1', 'odd_X', 'odd_2'])

def get_training_df():
    file_path = os.path.dirname(os.path.abspath(__file__))
    #import dataset
    all_files = sorted(glob.glob(file_path + "/../csv/*.csv"), key=lambda x: int(x[15:-4]))
    li = [pd.read_csv(filename, index_col=None, header=0) for filename in all_files[:-1]]
    df = pd.concat(li, axis=0, ignore_index=True)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
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

    df.reset_index(drop = True).to_csv(filepath + "/../dfs_pp/training_result.csv")
    return df.reset_index(drop = True)


def process_input_data(input_df, training_df):

    if 'home_final_score' in input_df.columns and 'away_final_score' in input_df.columns:
        input_df.drop(columns = ['home_final_score', 'away_final_score'], inplace = True)

    input_df = utils.nan_imputation(training_df, input_df)

    #introduce target variables
    input_df['actual_result'] = np.where(input_df['home_score'] > input_df['away_score'], 1, np.where(input_df['home_score'] == input_df['away_score'], 2, 3))
    input_df['result_strongness'] = (input_df['home_score'] - input_df['away_score']) * input_df['minute']   
    return input_df


#drop_y_column = ['home_final_score', 'away_final_score', 'result', 'final_total_goals']
def train_and_save_model(train):
    """
    Create model and save it with joblib
    """
    file_path = os.path.dirname(os.path.abspath(__file__))
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_total_goals']
    train_y = train['result'].values
    train_X = train.drop(columns = ['result'] + cat_col + outcome_cols)

    xgb = XGBClassifier(n_estimators = 2000)
    xgb.fit(train_X, train_y)
    joblib.dump(xgb, file_path + "/../models_pp/result.joblib")


def get_model():
    return joblib.load(file_path + "/../models_pp/result.joblib")


def get_predict_proba(model, input_df):
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    predictions = model.predict(input_df.drop(columns = cat_col))
    probabilities = model.predict_proba(input_df.drop(columns = cat_col))
    return predictions, probabilities


def get_predictions_table_nodraws(input_df, predictions, probabilities, threshold = 0.9):
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
    input_df['probability_1'] = probabilities[:,0]
    input_df['probability_X'] = probabilities[:,1]
    input_df['probability_2'] = probabilities[:,2]
    final_df = input_df.loc[:, ['id_partita','home', 'away', 'minute', 'home_score',\
         'away_score', 'probability_1','probability_X', 'probability_2', 'predictions']]\
             .sort_values(by = ['minute'], ascending = False, inplace = False)
    return final_df

def get_prior_posterior_predictions(input_pred_df, input_odds_df):
    # al 15 minuto probabilità pesate 50-50
    rate = 0.6 / 90
    res_df = input_pred_df.merge(input_odds_df, on = ['id_partita', 'minute'])
    res_df['probability_final_result_1'] = ((0.4 + (rate*res_df['minute'])) * res_df['probability_1'])\
                                         + ((0.6 - (rate*res_df['minute'])) * res_df['odd_1'])
    res_df['probability_final_result_X'] = ((0.4 + (rate*res_df['minute'])) * res_df['probability_X'])\
                                         + ((0.6 - (rate*res_df['minute'])) * res_df['odd_X'])
    res_df['probability_final_result_2'] = ((0.4 + (rate*res_df['minute'])) * res_df['probability_2'])\
                                         + ((0.6 - (rate*res_df['minute'])) * res_df['odd_2'])
    res_df['prediction_final_result'] = np.argmax(res_df[['probability_final_result_1','probability_final_result_X', 'probability_final_result_2']].values, axis = 1) + 1
    return res_df                    
