from xgboost import XGBClassifier
import os
import joblib
from model import Model


class xgb(Model):

    def __init__(self, train_df, cat_col, outcome_cols):
        self.train_df = train_df
        self.cat_col = cat_col
        self.outcome_cols = outcome_cols

    def preprocess_train(self):
        train_y = self.train_df['final_uo'].values
        to_drop = self.cat_col + self.outcome_cols
        train_X = self.train_df.drop(columns=to_drop)
        return train_X, train_y

    def preprocess_input(self, input_df, test_y=None):
        # preprocessing test
        test_X = input_df.drop(columns=self.cat_col)
        if test_y is not None:
            test_y_values = test_y['final_uo'].values
            return test_X, test_y_values
        return test_X

    def build_model(self, params=None):
        if params is None:
            params = {}
        model = XGBClassifier(**params)
        return model

    def train(self, model, train_X, train_y, test_X=None, test_y=None):
        model.fit(train_X, train_y)

    def save_model(self, model):
        file_path = os.path.dirname(os.path.abspath(__file__))
        joblib.dump(model, file_path + "/../models_pp/goals.joblib")

    def get_model(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        return joblib.load(file_path + "/../models_pp/goals.joblib")

    def get_predict_proba(self, model, test_X, input_df):
        predictions = model.predict(test_X)
        probabilities = model.predict_proba(test_X)
        input_df['predictions'] = predictions
        input_df['probability_over'] = probabilities[:, 0]
        return input_df
