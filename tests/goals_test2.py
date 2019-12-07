import numpy as np
from sklearn.metrics import accuracy_score
from matches_predictor import goals_utils, classifiers


def get_ids_for_test(df, prematch_odds=True, live_odds=True):
    if prematch_odds and live_odds:
        total_odds_mask = (df['live_odd_1'] != 0) & (df['odd_1'] != 0)
        id_partita_total_odds = df.loc[total_odds_mask, 'id_partita'].unique()
        if len(id_partita_total_odds) > len(df['id_partita'].unique()) // 4:
            id_partita_test = np.random.choice(id_partita_total_odds, size=len(df['id_partita'].unique()) // 4, replace=False)
        else:
            id_partita_test = id_partita_total_odds
    elif prematch_odds:
        prematch_odds_mask = df['odd_1'] != 0
        id_partita_prematch_odds = df.loc[prematch_odds_mask, 'id_partita'].unique()
        if len(id_partita_prematch_odds) > len(df['id_partita'].unique()) // 4:
            id_partita_test = np.random.choice(id_partita_prematch_odds, size=len(df['id_partita'].unique()) // 4, replace=False)
        else:
            id_partita_test = id_partita_prematch_odds
    elif live_odds:
        live_odds_mask = df['live_odd_1'] != 0
        id_partita_live_odds = df.loc[live_odds_mask, 'id_partita'].unique()
        if len(id_partita_live_odds) > len(df['id_partita'].unique()) // 4:
            id_partita_test = np.random.choice(id_partita_live_odds, size=len(df['id_partita'].unique()) // 4, replace=False)
        else:
            id_partita_test = id_partita_live_odds
    # splittare test and train in modo che nel train ci siano alcune partite, nel test altre
    else:
        id_partita_test = np.random.choice(df['id_partita'].unique(), size=len(df['id_partita'].unique()) // 4, replace=False)
    return id_partita_test


def split_test_train(df, prematch_odds=True, live_odds=True, minute=80):
    id_partita_test = get_ids_for_test(df, prematch_odds=prematch_odds, live_odds=live_odds)
    test_mask = df['id_partita'].isin(id_partita_test)
    test = df.loc[test_mask, :].copy(deep=True)
    train = df.loc[~(test_mask), :].copy(deep=True)
    test = drop_easy_predictions(test, minute)
    return train, test


def drop_easy_predictions(test, minute=80):
    # drop too easy predictions
    under_minute_mask = test['minute'] <= minute
    return test.loc[under_minute_mask, :]


def get_revenues(df, thresh=0.75):
    res = 0
    for _, row in df.iterrows():
        if row['probability_final_under'] >= thresh or row['probability_final_under'] <= 1-thresh:
            if row['final_uo'] == row['prediction_final_over']:
                if row['prediction_final'] == 'under':
                    if row['live_odd_under'] > 1:
                        res += (row['live_odd_under'] - 1)
                else:
                    if row['live_odd_over'] > 1:
                        res += (row['live_odd_over'] - 1)
            else:
                res -= 1
    return res


def get_insights(df):
    predictions_final = df['prediction_final_over']
    true_y = df['final_uo']
    acc = accuracy_score(true_y, predictions_final)
    print(f'Accuracy: {acc:.2f} \n')
    rev = get_revenues(df)
    print(f'Revenues: {rev:.2f} \n')
    print(df)


def get_live_predictions(clf='xgb'):
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_uo']

    df = goals_utils.get_training_df()

    for col in df.columns:
        if col not in cat_col:
            df[col] = pd.to_numeric(df[col])

    train, test = split_test_train(df)
    train = goals_utils.process_train(train)

    test['final_uo'] = np.where(test['home_final_score'] + test['away_final_score'] > 2, 0, 1)
    test_y = test[['id_partita', 'minute', 'final_uo']].copy()
    #test.drop(columns=outcome_cols, inplace=True)

    test = goals_utils.normalize_prematch_odds(test)
    test_prematch_odds = goals_utils.pop_input_prematch_odds_data(test)
    test_live_odds = goals_utils.pop_input_live_odds_data(test)

    test = goals_utils.process_input_data(test, train, cat_col)

    if clf == 'xgb':
        clf = classifiers.xgb(train, cat_col, outcome_cols)
    elif clf == 'lstm':
        clf = classifiers.lstm(train, cat_col, outcome_cols)

    test_X, test_y_values = clf.preprocess_input(test, test_y=True)
    goals_utils.train_and_save_model(clf, test_X, test_y_values)

    model = clf.get_model()
    test = clf.get_predict_proba(model, test_X, test)
    predictions_df = goals_utils.get_complete_predictions_df(test)
    predictions_df = goals_utils.get_posterior_predictions(predictions_df,
                                                           test_prematch_odds)
    predictions_df = predictions_df.merge(test_y)
    predictions_df = predictions_df.merge(test_live_odds, on=['minute', 'id_partita'])
    get_insights(predictions_df)

get_live_predictions(clf='lstm')