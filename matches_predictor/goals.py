import pandas as pd
from matches_predictor import goals_utils
import os

def get_live_predictions(reprocess_train_data=False,
                         retrain_model=False):
    file_path = os.path.dirname(os.path.abspath(__file__))
    input_goals_df = goals_utils.get_input_data()
    input_goals_df = goals_utils.normalize_odds(input_goals_df)
    input_goals_odds = goals_utils.pop_input_odds_data(input_goals_df)

    if reprocess_train_data:
        train_goals_df = goals_utils.get_training_df()
    else:
        train_goals_df = pd.read_csv(file_path + "/../dfs_pp/training_goals.csv", header = 0)
    if 'Unnamed: 0' in train_goals_df.columns:
        train_goals_df.drop(columns = ['Unnamed: 0'], inplace = True)

    input_goals_df = goals_utils.process_input_data(input_goals_df, train_goals_df)

    if retrain_model:
        goals_utils.train_and_save_model(train_goals_df)

    m_goals = goals_utils.get_model()
    predictions_goals, probabilities_goals = goals_utils.get_predict_proba(m_goals, input_goals_df)
    predictions_goals_df = goals_utils.get_complete_predictions_table(input_goals_df, predictions_goals, probabilities_goals)
    predictions_goals_df = goals_utils.get_prior_posterior_predictions(predictions_goals_df, input_goals_odds)
    return predictions_goals_df
