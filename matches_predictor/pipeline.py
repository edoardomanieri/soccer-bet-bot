from matches_predictor import result, goals


def run(reprocess_train_result_data=False, retrain_result_model=False,
        reprocess_train_goals_data=False, retrain_goals_model=False):
    predictions_result_df = result.get_live_predictions(reprocess_train_result_data, retrain_result_model)
    predictions_goals_df = goals.get_live_predictions(reprocess_train_goals_data, retrain_goals_model)
    final_df = predictions_result_df.merge(predictions_goals_df.loc[:, ["id_partita","minute", "predictions",
                        "probability_over", 'prediction_final', 'probability_final_over']], on = ['id_partita', 'minute'])
    final_df.rename(columns={'predictions_x': 'predictions_result', 'predictions_y': 'predictions_goals'}, inplace = True)
    final_df.drop(columns={'predictions_result', 'predictions_goals'}, inplace=True)
    final_df.loc[final_df['prediction_final_result'] == 2, 'prediction_final_result'] = 'X'
    final_df.loc[final_df['prediction_final_result'] == 3, 'prediction_final_result'] = 2
    return final_df
