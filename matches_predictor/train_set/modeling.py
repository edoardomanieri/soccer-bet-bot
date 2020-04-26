import joblib
import os


def train_model(train_df, clf, cat_col, outcome_cols, prod=True):
    """
    Create model and save it with joblib
    """
    train_y = train_df['final_uo'].values
    to_drop = cat_col + outcome_cols
    train_X = train_df.drop(columns=to_drop)
    clf.fit(train_X, train_y)
    file_path = os.path.dirname(os.path.abspath(__file__))
    prod_path = "production" if prod else "development"
    path = f"{file_path}/../../res/models/{prod_path}/goals.joblib"
    joblib.dump(clf, path)


def save_model(clf):
    file_path = os.path.dirname(os.path.abspath(__file__))
    path = f"{file_path}/../../res/models/development/goals.joblib"
    joblib.dump(clf, path)


def get_prod_model():
    file_path = os.path.dirname(os.path.abspath(__file__))
    path = f"{file_path}/../../res/models/production/goals.joblib"
    return joblib.load(path)


def get_dev_model():
    file_path = os.path.dirname(os.path.abspath(__file__))
    path = f"{file_path}/../../res/models/development/goals.joblib"
    return joblib.load(path)


def get_predict_proba(clf, test_X, input_df):
    predictions = clf.predict(test_X)
    probabilities = clf.predict_proba(test_X)
    input_df['predictions'] = predictions
    input_df['probability_over'] = probabilities[:, 0]
    return input_df
