import numpy as np
import pandas as pd
from matches_predictor.models.uo import train_set, input_stream
import os


class Prediction():

    def __init__(self, predictions_df):
        self.minute = predictions_df.loc[:, 'minute'][0]
        self.home = predictions_df.loc[:, 'home'][0]
        self.away = predictions_df.loc[:, 'away'][0]
        self.home_score = predictions_df.loc[:, 'home_score'][0]
        self.away_score = predictions_df.loc[:, 'away_score'][0]
        self.market_name = 'Over/Under 2.5 Goals'
        self.bet_type = 'uo'
        self.prediction = predictions_df.loc[:, 'prediction_final'][0]
        self.probability = predictions_df.loc[:, 'probability_final_over'][0] if predictions_df.loc[:, 'probability_final_over'][0] > 0.5 else 1 - predictions_df.loc[:, 'probability_final_over'][0]
        self.model_probability = predictions_df.loc[:, 'probability_over'][0] if predictions_df.loc[:, 'probability_over'][0] > 0.5 else 1 - predictions_df.loc[:, 'probability_over'][0]


def build_output_df(input_df):
    final_df = input_df.loc[:, ['id_partita', 'home', 'away', 'minute', 'home_score',
                                'away_score', 'prediction', 'probability_over']]\
        .sort_values(by='minute', ascending=False)\
        .groupby(['id_partita']).first().reset_index()
    return final_df


def prematch_odds_based(input_pred_df, input_prematch_odds_df):
    # al 15 minuto probabilità pesate 50-50
    rate = 0.6 / 90
    res_df = input_pred_df.merge(input_prematch_odds_df, on=['id_partita', 'minute'])
    res_df['probability_final_over'] = ((0.4 + (rate*res_df['minute'])) * res_df['probability_over'])\
        + ((0.6 - (rate*res_df['minute'])) * res_df['odd_over'])
    res_df['probability_final_under'] = ((0.4 + (rate*res_df['minute'])) * (1-res_df['probability_over']))\
        + ((0.6 - (rate*res_df['minute'])) * res_df['odd_under'])
    res_df['prediction_final_encoded'] = np.argmax(
        res_df[['probability_final_under', 'probability_final_over']].values, axis=1)
    res_df['prediction_final'] = np.where(
        res_df['prediction_final_encoded'] == 0, 'under', 'over')
    return res_df


def model_based(input_pred_df, input_prematch_odds_df):
    # al 15 minuto probabilità pesate 50-50
    res_df = input_pred_df
    res_df['probability_final_over'] = res_df['probability_over']
    res_df['probability_final_under'] = (1-res_df['probability_over'])
    res_df['prediction_final_encoded'] = np.argmax(res_df[['probability_final_under', 'probability_final_over']].values, axis=1)
    res_df['prediction_final'] = np.where(res_df['prediction_final_encoded'] == 0, 'under', 'over')
    return res_df


def get_predict_proba(clf, test_X, df):
    prediction = clf.predict(test_X)
    probabilities = clf.predict_proba(test_X)
    df['prediction'] = prediction
    df['probability_over'] = probabilities[:, 1]


def get_live_predictions(reprocess=False, retrain=False):

    file_path = os.path.dirname(os.path.abspath(__file__))
    cat_cols = ['home', 'away', 'campionato', 'date', 'id_partita']
    to_drop_cols = ['home', 'away', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_uo']
    api_missing_cols = ['home_punizioni', 'away_punizioni',
                        'home_rimesse_laterali', 'away_rimesse_laterali',
                        'home_contrasti', 'away_contrasti', 'home_attacchi',
                        'away_attacchi', 'home_attacchi_pericolosi',
                        'away_attacchi_pericolosi']

    if reprocess:
        train_df = train_set.Retrieving.starting_df(api_missing_cols, cat_cols)
        train_set.Preprocessing.execute(train_df, cat_cols, api_missing_cols)

    train_df = pd.read_csv(
        f"{file_path}/../res/dataframes/training_uo.csv", header=0, index_col=0)

    input_df = input_stream.Retrieving.starting_df(cat_cols, api_missing_cols)
    input_prematch_odds = input_stream.Preprocessing.execute(
        input_df, train_df, cat_cols)

    if retrain:
        clf = train_set.Modeling.get_dev_model()
        train_set.Modeling.train_model(
            train_df, clf, to_drop_cols, outcome_cols, prod=True)

    clf = train_set.Modeling.get_prod_model()
    test_X = input_df.drop(columns=cat_cols)
    get_predict_proba(clf, test_X, input_df)
    predictions_df = build_output_df(input_df)
    predictions_df = prematch_odds_based(predictions_df,
                                         input_prematch_odds)
    return predictions_df
