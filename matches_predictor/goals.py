import pandas as pd
from matches_predictor import utils, training, classifiers, input_data
import os


def get_live_predictions(clf='xgb', reprocess_train_data=False,
                         retrain_model=False):
    file_path = os.path.dirname(os.path.abspath(__file__))
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_uo']

    input_df = input_data.get_df()
    input_data.normalize_prematch_odds(input_df)
    input_prematch_odds = input_data.pop_prematch_odds_data(input_df)
    input_live_odds = input_data.pop_live_odds_data(input_df)

    if reprocess_train_data:
        train_df = training.get_df()
        training.to_numeric(train_df, cat_col)
        training.drop_odds_cols(train_df)
        training.drop_nan(train_df)
        training.impute_nan(train_df)
        training.add_outcome_col(train_df)
        training.add_input_cols(train_df)
        training.save(train_df)
    else:
        train_df = pd.read_csv(file_path + "/../dfs_pp/training_goals.csv",
                               header=0)
    if 'Unnamed: 0' in train_df.columns:
        train_df.drop(columns=['Unnamed: 0'], inplace=True)

    input_data.drop_outcome_cols(input_df)
    input_data.to_numeric(input_df, cat_col)
    input_data.drop_nan(input_df)
    input_data.impute_nan(train_df, input_df)
    input_data.add_input_cols(input_df)

    if clf == 'xgb':
        clf = classifiers.xgb(train_df, cat_col, outcome_cols)
    elif clf == 'lstm':
        clf = classifiers.lstm(train_df, cat_col, outcome_cols)

    if retrain_model:
        training.train_model(clf)

    model = clf.get_model()
    test_X = clf.preprocess_input(input_df)
    input_df = clf.get_predict_proba(model, test_X, input_df)
    predictions_df = utils.get_complete_predictions_df(input_df)
    predictions_df = utils.get_posterior_predictions(predictions_df,
                                                     input_prematch_odds)
    return predictions_df


get_live_predictions(clf='lstm', reprocess_train_data=True, retrain_model=True)