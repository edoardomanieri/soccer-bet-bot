import pandas as pd
from matches_predictor import utils
from matches_predictor.classifiers.xgb import Xgb
from matches_predictor import train_set
from matches_predictor import input_streaming
import os


def get_live_predictions(clf='xgb', params=None, reprocess_train_data=False,
                         retrain_model=False, res_path="../../res/csv"):

    if params is None:
        params = {}

    file_path = os.path.dirname(os.path.abspath(__file__))
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_uo']

    if reprocess_train_data:
        train_df = train_set.retrieving.get_df(res_path)
        train_set.preprocessing.execute(train_df, cat_col)
    train_df = pd.read_csv(
        file_path + "/../res/dataframes/training_goals.csv", header=0, index_col=0)

    input_df = input_streaming.retrieving.get_df(res_path)
    input_prematch_odds = input_streaming.preprocessing.execute(
        input_df, train_df, cat_col)

    clf = Xgb(train_df, cat_col, outcome_cols)

    if retrain_model:
        train_set.modeling.train_model(clf, params)

    model = clf.get_model()
    test_X = clf.preprocess_input(input_df)
    input_df = clf.get_predict_proba(model, test_X, input_df)
    predictions_df = utils.get_complete_predictions_df(input_df)
    predictions_df = utils.get_posterior_predictions(predictions_df,
                                                     input_prematch_odds)
    return predictions_df
