from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Masking
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np


def preprocessing_train(train, test, cat_col):
    # preprocessing lstm train
    special_value = -2
    outcome_cols = ['home_final_score', 'away_final_score', 'final_uo']

    train.sort_values(by=['minute'], inplace=True)
    gb = train.groupby(['id_partita'])
    test.sort_values(by=['minute'], inplace=True)
    gt = test.groupby(['id_partita'])
    mx = max(gb['id_partita'].size().max(), gt['id_partita'].size().max())
    not_ts_cols = outcome_cols + cat_col + ['minute']
    n_cols = len(train.columns) - len(not_ts_cols)

    train_X = np.array([np.pad(frame[[col for col in train.columns if col not in not_ts_cols]].values,
                        pad_width=[(0, mx-len(frame)), (0, 0)],
                        mode='constant',
                        constant_values=special_value)
                        for _,frame in gb]).reshape(-1, mx, n_cols)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X = scaler.fit_transform(train_X.reshape(-1, n_cols)).reshape(-1, mx, n_cols)
    train_y = gb.first()['final_uo'].values
    train_y = to_categorical(train_y)
    return train_X, train_y


def preprocessing_test(train, test, cat_col):
    special_value = -2
    outcome_cols = ['home_final_score', 'away_final_score', 'final_uo']

    train.sort_values(by=['minute'], inplace=True)
    gb = train.groupby(['id_partita'])
    test.sort_values(by=['minute'], inplace=True)
    gt = test.groupby(['id_partita'])
    mx = max(gb['id_partita'].size().max(), gt['id_partita'].size().max())
    not_ts_cols = outcome_cols + cat_col + ['minute']
    n_cols = len(train.columns) - len(not_ts_cols)
    # preprocessing test
    test_X = np.array([np.pad(frame[[col for col in test.columns if col not in not_ts_cols]].values,
                       pad_width=[(0, mx-len(frame)), (0, 0)],
                       mode='constant',
                       constant_values=special_value)
                       for _,frame in gt]).reshape(-1, mx, n_cols)
    scaler = MinMaxScaler(feature_range=(0, 1))
    test_X = scaler.fit_transform(test_X.reshape(-1, n_cols)).reshape(-1, mx, n_cols)
    test_y = gt.first()['final_uo'].values
    test_y = to_categorical(test_y)
    return test_X, test_y, gt.last(), mx, n_cols


def build_model(special_value, mx, n_cols):
    model = Sequential()
    model.add(Masking(mask_value=special_value, input_shape=(mx, n_cols)))
    model.add(LSTM(30, input_shape=(mx, n_cols), return_sequences=True))
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


def train(model, train_X, train_y, test_X, test_y):
    history = model.fit(train_X, train_y, validation_data=(test_X, test_y),
                        batch_size=16, epochs=30,  verbose=1)
    return history.history
