import pandas as pd
from matches_predictor import utils
from matches_predictor import train_set
from matches_predictor import input_streaming
import os


def get_live_predictions(reprocess=False, retrain=False, res_path="../../res/csv"):

    file_path = os.path.dirname(os.path.abspath(__file__))
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_uo']

    if reprocess:
        train_df = train_set.retrieving.get_df(res_path)
        train_set.preprocessing.execute(train_df, cat_col)

    train_df = pd.read_csv(
        f"{file_path}/../res/dataframes/training_goals.csv", header=0, index_col=0)

    input_df = input_streaming.retrieving.get_df(res_path)
    input_prematch_odds = input_streaming.preprocessing.execute(
        input_df, train_df, cat_col)

    if retrain:
        clf = train_set.modeling.get_dev_model()
        train_set.modeling.train_model(
            train_df, clf, cat_col, outcome_cols, prod=True)

    clf = train_set.modeling.get_prod_model()
    test_X = input_df.drop(columns=cat_col)
    input_df = train_set.modeling.get_predict_proba(clf, test_X, input_df)
    predictions_df = utils.get_complete_predictions_df(input_df)
    predictions_df = utils.get_posterior_predictions(predictions_df,
                                                     input_prematch_odds)
    return predictions_df
