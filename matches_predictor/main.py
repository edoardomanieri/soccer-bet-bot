from matches_predictor import test_set, validation
from xgboost import XGBClassifier

if __name__ == "__main__":
    cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_uo']
    api_missing_cols = ['home_punizioni', 'away_punizioni', 'home_rimesse_laterali', 'away_rimesse_laterali',
                        'home_contrasti', 'away_contrasti', 'home_attacchi', 'away_attacchi',
                        'home_attacchi_pericolosi', 'away_attacchi_pericolosi']
    params = {
        'learning_rate': [0.05, 0.1, 0.15, 0.20, 0.3],
        'n_estimators': [200, 300, 500],
        'min_child_weight': [3, 5, 10, 15],
        'gamma': [0.3, 0.5, 0.7, 1, 1.5],
        'subsample': [0.4, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        'max_depth': [2, 3, 4, 5]
    }
    clf = XGBClassifier(**params)
    df = test_set.Retrieving.starting_df(
        res_path='../res/csv', cat_col=cat_col)

    best_acc, best_params = validation.randomizedsearch_CV(
        df, clf, cat_col, api_missing_cols, outcome_cols, params, cv=5, trials=5, threshold=0.7)
    print(best_acc)
