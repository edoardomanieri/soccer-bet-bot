import numpy as np
import pandas as pd
from matches_predictor import train_set, input_stream
import os


def build_output_df(input_df):
    final_df = input_df.loc[:, ['id_partita', 'home', 'away', 'minute', 'home_score',
                                'away_score', 'predictions', 'probability_over']]\
        .sort_values(by='minute', ascending=False)\
        .groupby(['id_partita']).first().reset_index()
    return final_df


def consumer_outuput(input_df):
    final_df = input_df.loc[:, ['minute', 'home', 'away', 'prediction_final', 'probability_final_over']]
    return final_df


def prematch_odds_based(input_pred_df, input_prematch_odds_df):
    # al 15 minuto probabilità pesate 50-50
    rate = 0.6 / 90
    res_df = input_pred_df.merge(input_prematch_odds_df, on=[
                                 'id_partita', 'minute'])
    res_df['probability_final_over'] = ((0.4 + (rate*res_df['minute'])) * res_df['probability_over'])\
        + ((0.6 - (rate*res_df['minute'])) * res_df['odd_over'])
    res_df['probability_final_under'] = ((0.4 + (rate*res_df['minute'])) * (1-res_df['probability_over']))\
        + ((0.6 - (rate*res_df['minute'])) * res_df['odd_under'])
    res_df['prediction_final_encoded'] = np.argmax(
        res_df[['probability_final_under', 'probability_final_over']].values, axis=1)
    res_df['prediction_final'] = np.where(
        res_df['prediction_final_encoded'] == 0, 'under', 'over')
    return res_df


def get_predict_proba(clf, test_X, df):
    predictions = clf.predict(test_X)
    probabilities = clf.predict_proba(test_X)
    df['predictions'] = predictions
    df['probability_over'] = probabilities[:, 1]


def get_live_predictions(reprocess=False, retrain=False, res_path="../res/csv"):

    file_path = os.path.dirname(os.path.abspath(__file__))
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_uo']

    if reprocess:
        train_df = train_set.Retrieving.starting_df(res_path)
        train_set.Preprocessing.execute(train_df, cat_col)

    train_df = pd.read_csv(
        f"{file_path}/../res/dataframes/training_goals.csv", header=0, index_col=0)

    input_df = input_stream.Retrieving.starting_df(res_path)
    input_prematch_odds = input_stream.Preprocessing.execute(
        input_df, train_df, cat_col)

    if retrain:
        clf = train_set.Modeling.get_dev_model()
        train_set.Modeling.train_model(
            train_df, clf, cat_col, outcome_cols, prod=True)

    clf = train_set.Modeling.get_prod_model()
    test_X = input_df.drop(columns=cat_col)
    get_predict_proba(clf, test_X, input_df)
    predictions_df = build_output_df(input_df)
    predictions_df = prematch_odds_based(predictions_df,
                                         input_prematch_odds)
    return predictions_df


def predictions_consumer(in_q, out_q):
    res_path = "../res/csv"
    file_path = os.path.dirname(os.path.abspath(__file__))
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_uo']
    train_df = train_set.Retrieving.starting_df(res_path)
    train_set.Preprocessing.execute(train_df, cat_col)
    train_df = pd.read_csv(f"{file_path}/../res/dataframes/training_goals.csv", header=0, index_col=0)
    clf = train_set.Modeling.get_dev_model()
    train_set.Modeling.train_model(train_df, clf, cat_col, outcome_cols, prod=True)

    while True:
        input_df = in_q.get()
        input_prematch_odds = input_stream.Preprocessing.execute(
            input_df, train_df, cat_col)
        clf = train_set.Modeling.get_prod_model()
        test_X = input_df.drop(columns=cat_col)
        get_predict_proba(clf, test_X, input_df)
        predictions_df = prematch_odds_based(input_df, input_prematch_odds)
        final_df = consumer_outuput(predictions_df)
        out_q.put(final_df)
        print(final_df)
        in_q.task_done()


