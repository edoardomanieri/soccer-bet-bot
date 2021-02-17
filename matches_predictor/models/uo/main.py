from matches_predictor.models.uo import test_set, validation, prediction
from xgboost import XGBClassifier
import sqlite3
import os

if __name__ == "__main__":

    def mask_minute3070_1goal(df):
        minute_mask = (df['minute'] < 70) & (df['minute'] > 30)
        goal_mask = (df['home_score'] + df['away_score']) == 1
        total_mask = minute_mask & goal_mask
        return total_mask

    def mask_all(df):
        total_mask = df['home_score'] >= 0
        return total_mask

    file_path = os.path.dirname(os.path.abspath(__file__))
    conn = sqlite3.connect(f"{file_path}/../../../res/football.db")

    cat_cols = ['home', 'away', 'campionato', 'date', 'id_partita']
    to_drop = ['home', 'away', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_uo']
    api_missing_cols = ['home_punizioni', 'away_punizioni', 'home_rimesse_laterali',
                        'away_rimesse_laterali', 'home_contrasti', 'away_contrasti',
                        'home_attacchi', 'away_attacchi', 'home_attacchi_pericolosi',
                        'away_attacchi_pericolosi']
    params = {
        'learning_rate': [0.05, 0.1, 0.15, 0.20, 0.3],
        'n_estimators': [200, 300, 500, 700],
        'min_child_weight': [3, 5, 10, 15],
        'gamma': [0.3, 0.5, 0.7, 1, 1.5],
        'subsample': [0.4, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        'max_depth': [2, 3, 4, 5]
    }
    prob_threshold = 0.7
    clf = XGBClassifier(**params)
    df = test_set.Retrieving.starting_df(cat_cols, api_missing_cols, conn)

    do_mask_all = False
    if do_mask_all:
        best_acc, best_params = validation.randomizedsearch_CV(
            df, mask_all, prediction.model_based, clf, cat_cols, to_drop,
            api_missing_cols, outcome_cols, params, force_saving=False,
            cv=5, trials=1, threshold=prob_threshold)
        print(f"Best threshold {prob_threshold} accuracy without mask: {best_acc}")

    best_acc, best_params = validation.randomizedsearch_CV(
        df, mask_minute3070_1goal, prediction.prematch_odds_based,
        clf, cat_cols, to_drop, api_missing_cols,
        outcome_cols, params, force_saving=False, cv=5, trials=20,
        threshold=prob_threshold)
    print(f"Best threshold {prob_threshold} accuracy 1 goal and 30-70 minute mask: {best_acc}")

    connection.close()
