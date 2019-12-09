from matches_predictor import test_set, training, classifiers, utils


def get_live_predictions(clf='xgb'):
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_uo']

    df = test_set.get_df()
    training.to_numeric(df, cat_col)

    train_df, test_df = test_set.split_test_train(df)
    training.drop_odds_cols(train_df)
    training.drop_nan(train_df)
    training.impute_nan(train_df)
    training.add_outcome_col(train_df)
    training.add_input_cols(train_df)

    test_set.drop_nan(test_df)
    test_set.impute_nan(train_df, test_df)
    test_set.add_input_cols(test_df)
    test_set.normalize_prematch_odds(test_df)
    test_prematch_odds = test_set.pop_prematch_odds_data(test_df)
    test_live_odds = test_set.pop_live_odds_data(test_df)

    test_set.add_outcome_col(test_df)
    test_y = test_df[['id_partita', 'minute', 'final_uo']].copy()
    test_set.drop_outcome_cols(test_df)

    if clf == 'xgb':
        clf = classifiers.xgb(train_df, cat_col, outcome_cols)
    elif clf == 'lstm':
        clf = classifiers.lstm(train_df, cat_col, outcome_cols, epochs=30)

    test_X, test_y_values = clf.preprocess_input(test_df, test_y=test_y)
    training.train_model(clf, test_X, test_y_values)

    model = clf.get_model()
    test_df = clf.get_predict_proba(model, test_X, test_df)
    predictions_df = utils.get_complete_predictions_df(test_df)
    predictions_df = utils.get_posterior_predictions(predictions_df,
                                                     test_prematch_odds)
    predictions_df = predictions_df.merge(test_y, on=['id_partita', 'minute'])
    predictions_df = predictions_df.merge(test_live_odds, on=['minute', 'id_partita'])
    test_set.get_insights(predictions_df)
    return predictions_df

get_live_predictions(clf='lstm')