import numpy as np
import pandas as pd
from matches_predictor.models.uo import train_set, input_stream
import os


class Prediction():

    def __init__(self, minute, home, away, market_name, prediction, probability, model_probability):
        self.minute = minute
        self.home = home
        self.away = away
        self.market_name = market_name
        self.prediction = prediction
        self.probability = probability if probability > 0.5 else 1 - probability
        self.model_probability = model_probability if model_probability > 0.5 else 1 - model_probability


def build_output_df(input_df):
    final_df = input_df.loc[:, ['id_partita', 'home', 'away', 'minute', 'home_score',
                                'away_score', 'predictions', 'probability_over']]\
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
    predictions = clf.predict(test_X)
    probabilities = clf.predict_proba(test_X)
    df['predictions'] = predictions
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


def predictions_prod_cons(in_q, out_q, prob_threshold):
    file_path = os.path.dirname(os.path.abspath(__file__))
    cat_cols = ['home', 'away', 'campionato', 'date', 'id_partita']
    to_drop_cols = ['home', 'away', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_uo']
    api_missing_cols = ['home_punizioni', 'away_punizioni', 'home_rimesse_laterali',
                        'away_rimesse_laterali', 'home_contrasti', 'away_contrasti',
                        'home_attacchi', 'away_attacchi','home_attacchi_pericolosi',
                        'away_attacchi_pericolosi']
    train_df = train_set.Retrieving.starting_df(cat_cols, api_missing_cols)
    train_set.Preprocessing.execute(train_df, cat_cols, api_missing_cols)
    train_df = pd.read_csv(f"{file_path}/../res/dataframes/training_uo.csv", header=0, index_col=0)
    # get clf from cross validation (dev) and retrain on all the train set
    clf = train_set.Modeling.get_dev_model()
    cols_used = train_set.Modeling.train_model(train_df, clf, to_drop_cols, outcome_cols, prod=True)
    clf = train_set.Modeling.get_prod_model()

    while True:
        input_df = pd.DataFrame(in_q.get())
        input_df.drop(columns=['fixture_id'], inplace=True)
        input_prematch_odds = input_stream.Preprocessing.execute(input_df,
                                                                 train_df,
                                                                 cat_cols)
        test_X = input_df[cols_used]
        get_predict_proba(clf, test_X, input_df)
        predictions_df = prematch_odds_based(input_df, input_prematch_odds)
        minute = predictions_df.loc[:, 'minute'][0]
        home = predictions_df.loc[:, 'home'][0]
        away = predictions_df.loc[:, 'away'][0]
        market_name = 'Over/Under 2.5 Goals'
        prediction = predictions_df.loc[:, 'prediction_final'][0]
        probability = predictions_df.loc[:, 'probability_final_over'][0]
        model_probability = predictions_df.loc[:, 'probability_over'][0]
        prediction_obj = Prediction(minute, home, away, market_name, prediction, probability, model_probability)
        if prediction_obj.probability > prob_threshold:
            out_q.put(prediction_obj)
        print(f"{prediction_obj.home}-{prediction_obj.away}, \
              minute: {prediction_obj.minute}, \
              probability: {prediction_obj.probability}, \
              model_probability: {prediction_obj.model_probability}, \
              eventual prediction: {prediction_obj.prediction}\n")
