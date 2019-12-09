from xgboost import XGBClassifier
import os
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Masking
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np
import abc
import joblib


class Model(metaclass=abc.ABCMeta):
    """
    Abstract class that is the super class of all models
    """

    @abc.abstractmethod
    def preprocess_train(self):
        pass

    @abc.abstractmethod
    def preprocess_input(self):
        pass

    @abc.abstractmethod
    def build_model(self):
        pass

    @abc.abstractmethod
    def train(self, model, train_X, train_y):
        pass

    @abc.abstractmethod
    def save_model(self):
        pass

    @abc.abstractmethod
    def get_predict_proba(self):
        pass


class lstm(Model):

    def __init__(self, train_df, cat_col, outcome_cols, special_value=-2,
                 batch_size=16, epochs=2):
        self.special_value = special_value
        self.train_df = train_df
        self.train_groups = train_df.groupby(['id_partita'])
        self.mx = self.train_groups['id_partita'].size().max()
        self.not_ts_cols = outcome_cols + cat_col + ['minute']
        self.n_cols = len(train_df.columns) - len(self.not_ts_cols)
        self.batch_size = batch_size
        self.epochs = epochs

    def preprocess_train(self):
        # preprocessing lstm train
        train_X = np.array([np.pad(frame[[col for col in self.train_df.columns
                                          if col not in self.not_ts_cols]].values,
                            pad_width=[(0, self.mx-len(frame)), (0, 0)],
                            mode='constant',
                            constant_values=self.special_value)
                            for _,frame in self.train_groups]).reshape(-1,
                                                                       self.mx,
                                                                       self.n_cols)
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_X = scaler.fit_transform(train_X.reshape(-1, self.n_cols)).reshape(-1, self.mx, self.n_cols)
        train_y = self.train_groups.first()['final_uo'].values
        train_y = to_categorical(train_y)
        return train_X, train_y

    def preprocess_input(self, input_df, test_y=None):
        # preprocessing test
        self.input_groups = input_df.sort_values(by='minute', ascending=False).groupby(['id_partita'])
        test_X = np.array([np.pad(frame[[col for col in input_df.columns
                                         if col not in self.not_ts_cols]].values,
                           pad_width=[(0, self.mx-len(frame)), (0, 0)],
                           mode='constant',
                           constant_values=self.special_value)
                           for _, frame in self.input_groups]).reshape(-1, self.mx, self.n_cols)
        scaler = MinMaxScaler(feature_range=(0, 1))
        test_X = scaler.fit_transform(test_X.reshape(-1, self.n_cols)).reshape(-1, self.mx, self.n_cols)
        if test_y is not None:
            test_y_values = test_y.sort_values(by='minute', ascending=False).groupby(['id_partita']).first()['final_uo'].values
            test_y_values = to_categorical(test_y_values)
            return test_X, test_y_values
        return test_X

    def build_model(self):
        model = Sequential()
        model.add(Masking(mask_value=self.special_value,
                          input_shape=(self.mx, self.n_cols)))
        model.add(LSTM(30, input_shape=(self.mx, self.n_cols),
                  return_sequences=True))
        model.add(Dropout(0.1))
        model.add(LSTM(20, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(LSTM(10, return_sequences=False))
        model.add(Dropout(0.1))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        print(model.summary())
        return model

    def train(self, model, train_X, train_y, test_X=None, test_y=None):
        if test_X is None and test_y is None:
            model.fit(train_X, train_y, batch_size=self.batch_size,
                      epochs=self.epochs, verbose=1)
        else:
            model.fit(train_X, train_y, batch_size=self.batch_size,
                      validation_data=(test_X, test_y),
                      epochs=self.epochs, verbose=1)

    def save_model(self, model):
        file_path = os.path.dirname(os.path.abspath(__file__))
        model.save(file_path + "/../models_pp/goals.h5")

    def get_model(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        return load_model(file_path + "/../models_pp/goals.h5")

    def get_predict_proba(self, model, test_X, input_df):
        predictions = np.argmax(model.predict(test_X), axis=1)
        probabilities = model.predict_proba(test_X)
        tmp = self.input_groups.first()
        tmp['predictions'] = predictions
        tmp['probability_over'] = probabilities[:, 0]
        merged = input_df.merge(tmp, on=['id_partita', 'minute'], suffixes=('', '_y'))
        return merged


class xgb(Model):

    def __init__(self, train_df, cat_col, outcome_cols, n_estimators=2000):
        self.train_df = train_df
        self.cat_col = cat_col
        self.outcome_cols = outcome_cols
        self.n_estimators = n_estimators

    def preprocess_train(self):
        # preprocessing lstm train
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

    def build_model(self):
        model = XGBClassifier(n_estimators=2000)
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
