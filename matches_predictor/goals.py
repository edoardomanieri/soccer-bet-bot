import pandas as pd
from matches_predictor import goals_utils, classifiers
import os


def get_live_predictions(clf='xgb', reprocess_train_data=False,
                         retrain_model=False):
    file_path = os.path.dirname(os.path.abspath(__file__))
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_uo']
    input_df = goals_utils.get_input_data()
    input_df = goals_utils.normalize_prematch_odds(input_df)
    input_prematch_odds = goals_utils.pop_input_prematch_odds_data(input_df)
    input_live_odds = goals_utils.pop_input_live_odds_data(input_df)

    if reprocess_train_data:
        train_df = goals_utils.get_training_df()
    else:
        train_df = pd.read_csv(file_path + "/../dfs_pp/training_goals.csv",
                               header=0)
    if 'Unnamed: 0' in train_df.columns:
        train_df.drop(columns=['Unnamed: 0'], inplace=True)

    input_df = goals_utils.process_input_data(input_df, train_df)

    if clf == 'xgb':
        clf = classifiers.xgb(train_df, cat_col, outcome_cols)
    elif clf == 'lstm':
        clf = classifiers.lstm(train_df, cat_col, outcome_cols)

    if retrain_model:
        goals_utils.train_and_save_model(clf)

    model = clf.get_model()
    test_X = clf.preprocess_input(input_df)
    pred_goals, prob_goals = clf.get_predict_proba(model, test_X)
    predictions_df = goals_utils.get_complete_predictions_df(input_df,
                                                             pred_goals,
                                                             prob_goals)
    predictions_df = goals_utils.get_posterior_predictions(predictions_df,
                                                           input_prematch_odds)
    return predictions_df

#get_live_predictions(clf='lstm', reprocess_train_data=True, retrain_model=True)