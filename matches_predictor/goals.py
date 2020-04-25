import pandas as pd
from matches_predictor import utils, training, classifiers, input_data
import os


def get_processed_train(reprocess_train_data, cat_col, file_path, res_path):
    if reprocess_train_data:
        train_df = training.get_df(res_path)
        training.to_numeric(train_df, cat_col)
        training.drop_odds_cols(train_df)
        training.drop_nan(train_df)
        training.impute_nan(train_df)
        training.add_outcome_col(train_df)
        training.add_input_cols(train_df)
        training.save(train_df)
    else:
        train_df = pd.read_csv(
            file_path + "/../res/dataframes/training_goals.csv", header=0, index_col=0)
    return train_df


def get_live_predictions(clf='xgb', params=None, reprocess_train_data=False,
                         retrain_model=False, res_path="../res/csv"):

    file_path = os.path.dirname(os.path.abspath(__file__))
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_uo']

    input_df = input_data.get_df(res_path)
    input_data.normalize_prematch_odds(input_df)
    input_prematch_odds = input_data.pop_prematch_odds_data(input_df)
    input_live_odds = input_data.pop_live_odds_data(input_df)

    train_df = get_processed_train(
        reprocess_train_data, cat_col, file_path, res_path)

    input_data.drop_outcome_cols(input_df)
    input_data.to_numeric(input_df, cat_col)
    input_data.drop_nan(input_df)
    input_data.impute_nan(train_df, input_df)
    input_data.add_input_cols(input_df)

    clf = classifiers.xgb.xgb(train_df, cat_col, outcome_cols)

    if retrain_model:
        training.train_model(clf, params)

    model = clf.get_model()
    test_X = clf.preprocess_input(input_df)
    input_df = clf.get_predict_proba(model, test_X, input_df)
    predictions_df = utils.get_complete_predictions_df(input_df)
    predictions_df = utils.get_posterior_predictions(predictions_df,
                                                     input_prematch_odds)
    return predictions_df
