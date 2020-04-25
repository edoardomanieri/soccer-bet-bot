from matches_predictor import goals


def test_goal():
    df = goals.get_live_predictions(
        clf='xgb', reprocess_train_data=True, retrain_model=True, res_path='../res/csv_test')
    assert len(df) != 0
    df = goals.get_live_predictions(
        clf='xgb', reprocess_train_data=False, retrain_model=False, res_path='../res/csv_test')
    assert len(df) != 0
