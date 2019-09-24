import model_result
import model_goals

def real_time_output(reprocess_train_result_data = False, reprocess_train_goals_data = False,retrain_result_model = False, retrain_goals_model = False):
    
    ##########result###########
    input_result_df = model_result.get_input_data()

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