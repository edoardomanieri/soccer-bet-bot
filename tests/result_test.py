import numpy as np
import pandas as pd
import glob
from sklearn.metrics import accuracy_score
from matches_predictor import result_utils, utils
from matches_predictor.classifiers import xgb
from sklearn.preprocessing import MinMaxScaler
from keras import optimizers
from keras.layers import GlobalAveragePooling2D, BatchNormalization, LSTM, Masking
from keras import Model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical

pd.set_option('display.float_format', lambda x: '%.3f' % x)


def get_ids_for_test(df, prematch_odds=True, live_odds=True):
    if prematch_odds and live_odds:
        # to be modified
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


# import dataset
all_files = sorted(glob.glob("./csv/*.csv"), key=lambda x: int(x[x.index('/csv/') + 10:-4]))
li = [pd.read_csv(filename, index_col=None, header=0) for filename in all_files]
df = pd.concat(li, axis=0, ignore_index=True)
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']

# change data type
for col in df.columns:
    if col not in cat_col:
        df[col] = pd.to_numeric(df[col])

df = result_utils.nan_drop_train(df)
id_partita_test = get_ids_for_test(df, prematch_odds=True, live_odds=False)

# split test and train
test_mask = df['id_partita'].isin(id_partita_test)
test = df.loc[test_mask, :].copy(deep=True)
train = df.loc[~(test_mask), :].copy(deep=True)


# nan imputation
train = utils.nan_impute_train(train)
test = utils.nan_impute_test(train, test)

# adding train outcome columns
train['result'] = np.where(train['home_final_score'] > train['away_final_score'], 1, np.where(train['home_final_score'] == train['away_final_score'], 2, 3))

# adding train additional information
# one hot encoding
# train['actual_result'] = np.where(train['home_score'] > train['away_score'], 1, np.where(train['home_score'] == train['away_score'], 2, 3))
train['result_strongness'] = (train['home_score'] - train['away_score']) * train['minute']

# adding test outcome columns
test['result'] = np.where(test['home_final_score'] > test['away_final_score'], 1, np.where(test['home_final_score'] == test['away_final_score'], 2, 3))

# adding test additional information
# one hot encoding
# test['actual_result'] = np.where(test['home_score'] > test['away_score'], 1, np.where(test['home_score'] == test['away_score'], 2, 3))
test['result_strongness'] = (test['home_score'] - test['away_score']) * test['minute']

# drop too easy predictions
result_strong_mask = abs(test['result_strongness']) > 160
test = test.loc[result_strong_mask, :]

test = result_utils.normalize_prematch_odds(test)
test_result_prematch_odds = result_utils.pop_input_prematch_odds_data(test)
test_result_live_odds = result_utils.pop_input_live_odds_data(test)
train = result_utils.drop_odds_col(train)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

outcome_cols = ['home_final_score', 'away_final_score', 'result']
test_y = test[['id_partita', 'minute', 'result']]
test.drop(columns=outcome_cols, inplace=True)
test_X = test.drop(columns=cat_col)

train_y = train['result'].values
train_X = train.drop(columns=outcome_cols + cat_col)

model, accuracy = xgb(train_X, train_y, test_X, test_y['result'].values)
print(accuracy)

predictions_result, probabilities_result = result_utils.get_predict_proba(model, test)
predictions_result_df = result_utils.get_complete_predictions_table(test, predictions_result,
                                                                    probabilities_result, threshold=0)
predictions_result_df = result_utils.get_prior_posterior_predictions(predictions_result_df, test_result_prematch_odds)


# test settings
predictions_result_df = predictions_result_df.merge(test_y)
predictions_final = predictions_result_df['prediction_final_result']
test_y = predictions_result_df['result']
accuracy_score(test_y, predictions_final)
predictions_result_df.loc[predictions_result_df['minute'] < 60, :]

# preprocessing lstm train
special_value = -2
outcome_cols = ['home_final_score', 'away_final_score', 'result']
train.sort_values(by=['minute'], inplace=True)
train['result'] = train['result'] - 1
gb = train.groupby(['id_partita'])

test.sort_values(by=['minute'], inplace=True)
test['result'] = test['result'] - 1
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
train_y = gb.first()['result'].values
train_y = to_categorical(train_y)


# preprocessing test
test_X = np.array([np.pad(frame[[col for col in test.columns if col not in not_ts_cols]].values,
                    pad_width=[(0, mx-len(frame)), (0, 0)],
                    mode='constant',
                    constant_values=special_value)
                    for _,frame in gt]).reshape(-1, mx, n_cols)
scaler = MinMaxScaler(feature_range=(0, 1))
test_X = scaler.fit_transform(test_X.reshape(-1, n_cols)).reshape(-1, mx, n_cols)
test_y = gt.first()['result'].values
test_y = to_categorical(test_y)


model = Sequential()
model.add(Masking(mask_value=special_value, input_shape=(mx, n_cols)))
model.add(LSTM(30, input_shape=(mx, n_cols), return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(20, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(10, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(train_X, train_y, validation_data=(test_X, test_y), batch_size=16, epochs=30,  verbose=1)
history.history
model.predict(test_X)