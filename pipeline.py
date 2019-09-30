import model_result
import model_goals
import pandas as pd

def real_time_output(reprocess_train_result_data = False, reprocess_train_goals_data = False,retrain_result_model = False, retrain_goals_model = False):
    
    ##########result###########
    input_result_df = model_result.get_input_data()
    input_result_df = model_result.normalize_odds(input_result_df)
    input_result_odds = model_result.pop_input_odds_data(input_result_df)

    if reprocess_train_result_data:
        train_result_df = model_result.get_training_df()
    else:
        train_result_df = pd.read_csv("../dfs/training_result.csv", header = 0)

    input_result_df = model_result.process_input_data(input_result_df, train_result_df)

    if retrain_result_model:
        model_result.train_and_save_model(train_result_df)

    m_result = model_result.get_model()
    predictions_result, probabilities_result = model_result.get_predict_proba(m_result, input_result_df)
    predictions_result_df = model_result.get_complete_predictions_table(input_result_df, predictions_result, \
        probabilities_result, threshold = 0)

    ########goals############

    input_goals_df = model_goals.get_input_data()
    input_goals_df = model_result.normalize_odds(input_goals_df)
    input_goals_odds = model_result.pop_input_odds_data(input_goals_df)

    if reprocess_train_goals_data:
        train_goals_df = model_goals.get_training_df()
    else:
        train_goals_df = pd.read_csv("../dfs/training_goals.csv", header = 0)

    input_goals_df = model_goals.process_input_data(input_goals_df, train_goals_df)

    if retrain_goals_model:
        model_goals.train_and_save_model(train_goals_df)

    m_goals = model_goals.get_model()
    predictions_goals = model_goals.get_predict(m_goals, input_goals_df)
    predictions_goals_df = model_goals.get_complete_predictions_table(input_goals_df, predictions_goals)

    ####renaming
    
    final_df = predictions_result_df.merge(predictions_goals_df.loc[:, ["id_partita","minute", "predictions"]], on = ['id_partita', 'minute'])
    final_df.rename(columns = {'predictions_x': 'predictions_result', 'predictions_y': 'predictions_goals'}, inplace = True)

    final_df.loc[final_df['predictions_result'] == 2,'predictions_result'] = 'X'
    final_df.loc[final_df['predictions_result'] == 3,'predictions_result'] = 2

    return final_df


real_time_output(False,False,False,False)

#peso dei risultati dati-prior
#si parte da un 50-50
#si arriva a un 100-0
#rate di aumento 0.5 / 90 al minuto

#calcolare probabilità a partire da regressione
#ad esempio se l'algoritmo predice 4 si avrà un over 2.5 con l'80 %
#rate (0.8/1.6)