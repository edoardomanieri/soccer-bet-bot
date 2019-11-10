import os
import pandas as pd
from matches_predictor import result_utils
import logging


def get_live_predictions(reprocess_train_data=False,
                         retrain_model=False):
    file_path = os.path.dirname(os.path.abspath(__file__))
    input_result_df = result_utils.get_input_data()
    input_result_df = result_utils.normalize_odds(input_result_df)
    input_result_odds = result_utils.pop_input_odds_data(input_result_df)

    if reprocess_train_data:
        train_result_df = result_utils.get_training_df()
    else:
        train_result_df = pd.read_csv(file_path + "/../dfs_pp/training_result.csv", header=0)
    if 'Unnamed: 0' in train_result_df.columns:
        train_result_df.drop(columns=['Unnamed: 0'], inplace=True)

    input_result_df = result_utils.process_input_data(input_result_df, train_result_df)

    if retrain_model:
        result_utils.train_and_save_model(train_result_df)

    m_result = result_utils.get_model()
    predictions_result, probabilities_result = result_utils.get_predict_proba(m_result, input_result_df)
    predictions_result_df = result_utils.get_complete_predictions_table(input_result_df, predictions_result,
        probabilities_result, threshold=0)
    predictions_result_df = result_utils.get_prior_posterior_predictions(predictions_result_df, input_result_odds)
    return predictions_result_df
