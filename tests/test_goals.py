from matches_predictor import prediction


def test_not_empty_df():
    df = prediction.get_live_predictions(
        reprocess=True, retrain=True, res_path='../../res/csv_test')
    assert len(df) != 0
    df = prediction.get_live_predictions(
        reprocess=False, retrain=False, res_path='../../res/csv_test')
    assert len(df) != 0
