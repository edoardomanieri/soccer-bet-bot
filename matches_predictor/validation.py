from sklearn.metrics import accuracy_score
from matches_predictor import test_set, train_set, prediction
import random
import itertools
from sklearn.base import clone


def _test_mask(df):
    minute_mask = (df['minute'] < 70) & (df['minute'] > 30)
    goal_mask = (df['home_score'] + df['away_score']) == 1
    total_mask = minute_mask & goal_mask
    return total_mask


def _get_ids_for_test(df, prematch_odds=True, live_odds=True):
    # splittare test and train in modo che nel train ci siano alcune partite, nel test altre
    mask = _test_mask(df)
    total_odds_mask = [True] * len(df)  # ?
    if prematch_odds and live_odds:
        total_odds_mask = (df['live_odd_1'] != 0) & (df['odd_1'] != 0)
    elif prematch_odds:
        total_odds_mask = df['odd_1'] != 0
    elif live_odds:
        total_odds_mask = df['live_odd_1'] != 0
    total_mask = (total_odds_mask & mask)
    id_partita_test = df.loc[total_mask, 'id_partita'].unique()
    return id_partita_test, total_mask


def _split_test_train(df, id_partita_test):
    test_mask = df['id_partita'].isin(id_partita_test)
    test = df.loc[test_mask, :].copy()
    train = df.loc[~(test_mask), :].copy()
    return train, test


def _drop_easy_predictions(test, minute=80):
    # drop too easy predictions
    under_minute_mask = test['minute'] <= minute
    return test.loc[under_minute_mask, :].copy()


def _get_revenues(df, thresh=0.75):
    res = 0
    for _, row in df.iterrows():
        if row['probability_final_under'] >= thresh or row['probability_final_under'] <= 1-thresh:
            if row['final_uo'] == row['prediction_final_encoded']:
                if row['prediction_final'] == 'under':
                    if row['live_odd_under'] > 1:
                        res += (row['live_odd_under'] - 1)
                else:
                    if row['live_odd_over'] > 1:
                        res += (row['live_odd_over'] - 1)
            else:
                res -= 1
    return res


def _get_insights(df):
    predictions_final = df['prediction_final_encoded']
    true_y = df['final_uo']
    acc = accuracy_score(true_y, predictions_final)
    print(f'Accuracy: {acc:.2f} \n')
    rev = _get_revenues(df)
    print(f'Revenues: {rev:.2f} \n')


def _get_accuracy(df, threshold=0.5):
    df = df.loc[(df['probability_final_over'] > threshold) |
                (df['probability_final_over'] < (1-threshold)), :]
    predictions_final = df['prediction_final_encoded']
    true_y = df['final_uo']
    acc = accuracy_score(true_y, predictions_final)
    return acc


def _partition(list_in, n):
    random.seed(12)
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def _show_df(input_df):
    final_df = input_df.loc[:, ['minute', 'home_score',
                                'away_score', 'probability_final_over', 'prediction_final_encoded', 'final_uo']]
    return final_df


def full_CV_pipeline(df, clf, cat_col, outcome_cols, cv=5, threshold=0.5):
    id_partita_test, total_mask = _get_ids_for_test(df, False, False)
    cv_lists = _partition(id_partita_test, cv)
    mae_folds = []
    for sublist in cv_lists:
        df_temp = df.copy()
        # drop all the matches that doesn't satisfy the condition
        dropping_mask = df_temp['id_partita'].isin(sublist) & ~total_mask
        df_temp = df_temp.drop(df_temp[dropping_mask].index)
        train_df, test_df = _split_test_train(df_temp, sublist)
        train_set.Preprocessing.execute(train_df, cat_col, prod=False)
        test_y, test_prematch_odds, test_live_odds = test_set.Preprocessing.execute(
            test_df, train_df)
        test_X = test_df.drop(columns=cat_col)
        train_set.Modeling.train_model(
            train_df, clf, cat_col, outcome_cols, prod=False)
        clf = train_set.Modeling.get_dev_model()
        prediction.get_predict_proba(clf, test_X, test_df)
        predictions_df = prediction.prematch_odds_based(
            test_df, test_prematch_odds)
        predictions_df = predictions_df.merge(
            test_y, on=['id_partita', 'minute'])
        predictions_df = predictions_df.merge(
            test_live_odds, on=['minute', 'id_partita'])
        print(_show_df(predictions_df).head(5))
        accuracy = _get_accuracy(predictions_df, threshold)
        mae_folds.append(accuracy)
        avg_accuracy = sum(mae_folds)/cv
    return mae_folds, avg_accuracy


def randomizedsearch_CV(df, estimator, cat_col, outcome_cols, param_dist, cv=5, threshold=0.5, trials=20):
    m = 0
    best_params = {}
    param_dict_list = []
    best_estimator = None
    for list_of_params in itertools.product(*param_dist.values()):
        param_dict = {x: y for x, y in zip(param_dist.keys(), list_of_params)}
        param_dict_list.append(param_dict)
    for _ in range(trials):
        param_dict = random.choice(param_dict_list)
        param_dict_list.remove(param_dict)
        estimator.set_params(**param_dict)
        selected_estimator = clone(estimator)
        _, res = full_CV_pipeline(
            df, selected_estimator, cat_col, outcome_cols, cv=cv, threshold=threshold)
        print(param_dict)
        print(res)
        if res > m:
            m = res
            best_params = param_dict
            best_estimator = clone(selected_estimator)
    train_set.Modeling.save_model(best_estimator)
    return m, best_params
