import pandas as pd
import numpy as np
import glob
import os
from xgboost import XGBClassifier
import joblib
from matches_predictor import utils


def get_input_data():
    file_path = os.path.dirname(os.path.abspath(__file__))
    all_files = sorted(glob.glob(file_path + "/../csv/*.csv"), key=lambda x: int(x[15:-4]))
    input_df = pd.read_csv(all_files[-1], index_col=None, header=0)
    if 'Unnamed: 0' in input_df.columns:
        input_df.drop(columns=['Unnamed: 0'], inplace=True)
    return input_df.sort_values(by=['id_partita', 'minute'], ascending=[True, False]).groupby(['id_partita']).first().reset_index() 


def normalize_odds(input_df):
    tmp = (1 - ((1 / input_df['odd_over']) + (1 / input_df['odd_under']))) / 2
    input_df['odd_over'] = (1 / input_df['odd_over']) + tmp
    input_df['odd_under'] = (1 / input_df['odd_under']) + tmp
    return input_df


def pop_input_odds_data(input_df):
    odds_input = input_df.loc[:,['id_partita', 'minute', 'odd_under', 'odd_over']]
    input_df.drop(columns=['odd_1', 'odd_2', 'odd_X', 'odd_over', 'odd_under'], inplace = True)
    return odds_input


def drop_odds_col(df):
    return df.drop(columns = ['odd_over', 'odd_under','odd_1', 'odd_X', 'odd_2'])


def get_training_df():
    file_path = os.path.dirname(os.path.abspath(__file__))
    #import dataset
    all_files = sorted(glob.glob(file_path + "/../csv/*.csv"), key = lambda x: int(x[15:-4]))
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
    df['final_total_goals'] = np.where(df['home_final_score'] + df['away_final_score'] > 2, 0, 1)

    #input columns
    df['actual_total_goals'] = df['home_score'] + df['away_score']

    df.reset_index(drop = True).to_csv(file_path + "/../dfs_pp/training_goals.csv")
    return df.reset_index(drop = True)


def process_input_data(input_df, training_df):
    if 'home_final_score' in input_df.columns and 'away_final_score' in input_df.columns:
        input_df.drop(columns = ['home_final_score', 'away_final_score'], inplace = True)

    input_df = utils.nan_imputation(training_df, input_df) 
    input_df['actual_total_goals'] = input_df['home_score'] + input_df['away_score']
    return input_df


def train_and_save_model(train):
    """
    Create model and save it with joblib
    """
    file_path = os.path.dirname(os.path.abspath(__file__))
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'result']
    train_y = train['final_total_goals'].values
    train_X = train.drop(columns = ['final_total_goals'] + cat_col + outcome_cols)
    xgb = XGBClassifier(n_estimators = 2000)
    xgb.fit(train_X, train_y)
    joblib.dump(xgb, file_path + "/../models_pp/goals.joblib")


def get_model():
    file_path = os.path.dirname(os.path.abspath(__file__))
    return joblib.load(file_path + "/../models_pp/goals.joblib")


def get_predict_proba(model, input_df):
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    predictions = model.predict(input_df.drop(columns = cat_col))
    probabilities = model.predict_proba(input_df.drop(columns = cat_col))
    return predictions, probabilities


def get_complete_predictions_table(input_df, predictions, probabilities):
    input_df['predictions'] = predictions
    input_df['probability_over'] = probabilities[:,0]
    final_df = input_df.loc[:, ['id_partita', 'home', 'away', 'minute', 'home_score','away_score','predictions', 'probability_over']]\
             .sort_values(by = ['minute'], ascending = False, inplace = False)
    return final_df


def get_prior_posterior_predictions(input_pred_df, input_odds_df):
    # al 15 minuto probabilità pesate 50-50
    rate = 0.6 / 90
    res_df = input_pred_df.merge(input_odds_df, on = ['id_partita', 'minute'])
    res_df['probability_final_over'] = ((0.4 + (rate*res_df['minute'])) * res_df['probability_over'])\
                                         + ((0.6 - (rate*res_df['minute'])) * res_df['odd_over'])
    res_df['probability_final_under'] = ((0.4 + (rate*res_df['minute'])) * (1-res_df['probability_over']))\
                                         + ((0.6 - (rate*res_df['minute'])) * res_df['odd_under'])
    res_df['prediction_final_over'] = np.argmax(res_df[['probability_final_over','probability_final_under']].values, axis = 1)
    res_df['prediction_final_over'] = np.where(res_df['prediction_final_over'] == 0, 'over', 'under')
    return res_df