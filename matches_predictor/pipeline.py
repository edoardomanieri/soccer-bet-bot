import model_result
import model_goals
import pandas as pd
import numpy as np


def real_time_output(reprocess_train_result_data=False, reprocess_train_goals_data=False,
                     retrain_result_model=False, retrain_goals_model=False):
    ##########result###########
    file_path = os.path.dirname(os.path.abspath(__file__))
    input_result_df = model_result.get_input_data()
    input_result_df = model_result.normalize_odds(input_result_df)
    input_result_odds = model_result.pop_input_odds_data(input_result_df)

    if reprocess_train_result_data:
        train_result_df = model_result.get_training_df()
    else:
        train_result_df = pd.read_csv(file_path + "/../dfs_pp/training_result.csv", header = 0)
    if 'Unnamed: 0' in train_result_df.columns:
        train_result_df.drop(columns = ['Unnamed: 0'], inplace = True)

    input_result_df = model_result.process_input_data(input_result_df, train_result_df)

    if retrain_result_model:
        model_result.train_and_save_model(train_result_df)

    m_result = model_result.get_model()
    predictions_result, probabilities_result = model_result.get_predict_proba(m_result, input_result_df)
    predictions_result_df = model_result.get_complete_predictions_table(input_result_df, predictions_result, \
        probabilities_result, threshold = 0)
    predictions_result_df = model_result.get_prior_posterior_predictions(predictions_result_df, input_result_odds)

    ########goals############

    input_goals_df = model_goals.get_input_data()
    input_goals_df = model_goals.normalize_odds(input_goals_df)
    input_goals_odds = model_goals.pop_input_odds_data(input_goals_df)

    if reprocess_train_goals_data:
        train_goals_df = model_goals.get_training_df()
    else:
        train_goals_df = pd.read_csv(file_path + "/../dfs_pp/training_goals.csv", header = 0)
    if 'Unnamed: 0' in train_goals_df.columns:
        train_goals_df.drop(columns = ['Unnamed: 0'], inplace = True)

    input_goals_df = model_goals.process_input_data(input_goals_df, train_goals_df)

    if retrain_goals_model:
        model_goals.train_and_save_model(train_goals_df)

    m_goals = model_goals.get_model()
    predictions_goals, probabilities_goals = model_goals.get_predict_proba(m_goals, input_goals_df)
    predictions_goals_df = model_goals.get_complete_predictions_table(input_goals_df, predictions_goals, probabilities_goals)
    predictions_goals_df = model_goals.get_prior_posterior_predictions(predictions_goals_df, input_goals_odds)

    ####renaming
    
    final_df = predictions_result_df.merge(predictions_goals_df.loc[:, ["id_partita","minute", "predictions",
                        "probability_over", 'prediction_final_over', 'probability_final_over']], on = ['id_partita', 'minute'])
    final_df.rename(columns = {'predictions_x': 'predictions_result', 'predictions_y': 'predictions_goals'}, inplace = True)

    final_df.drop(columns={'predictions_result', 'predictions_goals'}, inplace=True)
    final_df.loc[final_df['prediction_final_result'] == 2,'prediction_final_result'] = 'X'
    final_df.loc[final_df['prediction_final_result'] == 3,'prediction_final_result'] = 2

    return final_df


final_df = real_time_output(False,False,False,False)
final_df = final_df[final_df['minute'] < 85]
final_df['probability_final_result'] = np.max(final_df[['probability_final_result_1', 'probability_final_result_X', 'probability_final_result_2']].values, axis = 1) 
final_df.loc[:,['home', 'away', 'minute', 'home_score', 'away_score','probability_final_result', 'prediction_final_result','probability_final_over', 'prediction_final_over']]

#peso dei risultati dati-prior
#si parte da un 50-50
#si arriva a un 100-0
#rate di aumento 0.5 / 90 al minuto