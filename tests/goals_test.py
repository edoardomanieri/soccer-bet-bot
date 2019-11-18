import numpy as np
import pandas as pd
import glob
from sklearn.metrics import accuracy_score, mean_squared_error
from matches_predictor import result_utils, goals_utils, utils
from matches_predictor.classifiers import xgb
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def get_ids_for_splitting(df, prematch_odds=True, live_odds=True):
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

id_partita_test = get_ids_for_splitting(df,prematch_odds=True, live_odds=False)

# splitting
test_mask = df['id_partita'].isin(id_partita_test)
test = df.loc[test_mask, :].copy(deep=True)
train = df.loc[~(test_mask), :].copy(deep=True)

# nan imputation
train = utils.nan_imputation(train, train)
test = utils.nan_imputation(train, test)

# adding outcome columns
train['result'] = np.where(train['home_final_score'] > train['away_final_score'], 1, np.where(train['home_final_score'] == train['away_final_score'], 2, 3))
train['final_total_goals'] = train['home_final_score'] + train['away_final_score']

# adding additional information
train['actual_total_goals'] = train['home_score'] + train['away_score']
train['actual_result'] = np.where(train['home_score'] > train['away_score'], 1, np.where(train['home_score'] == train['away_score'], 2, 3))
train['result_strongness'] = (train['home_score'] - train['away_score']) * train['minute']

# adding outcome columns
test['result'] = np.where(test['home_final_score'] > test['away_final_score'], 1, np.where(test['home_final_score'] == test['away_final_score'], 2, 3))
test['final_total_goals'] = test['home_final_score'] + test['away_final_score']

# adding additional information
test['actual_total_goals'] = test['home_score'] + test['away_score']
test['actual_result'] = np.where(test['home_score'] > test['away_score'], 1, np.where(test['home_score'] == test['away_score'], 2, 3))
test['result_strongness'] = (test['home_score'] - test['away_score']) * test['minute']

###############################1x2#####################################
test = result_utils.normalize_prematch_odds(test)
test_result_prematch_odds = result_utils.pop_input_prematch_odds_data(test)
test_result_live_odds = result_utils.pop_input_live_odds_data(test)
train = result_utils.drop_odds_col(train)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

outcome_cols = ['home_final_score', 'away_final_score', 'final_total_goals', 'result']
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
predictions_result_df
predictions_final = predictions_result_df['prediction_final_result']
test_y = predictions_result_df['result']
accuracy_score(test_y, predictions_final)

predictions_result_df.loc[predictions_result_df['minute'] < 60, :]


########################total_goals##################
test_y = test['final_total_goals'].values
test_X = test.drop(columns = drop_y_column)
#test_X = test_X.drop(columns = tmp_y_col_to_be_dropped)

train_y = train['final_total_goals'].values
train_X = train.drop(columns = drop_y_column)
#train_X = train_X.drop(columns = tmp_y_col_to_be_dropped)

xgb = XGBRegressor(n_estimators = 2000)
xgb.fit(train_X, train_y)
predictions = xgb.predict(test_X)
mean_squared_error(test_y, predictions)

test['predictions'] = predictions
test.loc[test['predictions'] < test['actual_total_goals'],'predictions'] = test['actual_total_goals']
test['true_values'] = test_y
new_df = test.merge(df, how = "left")
minute_max_mask = new_df['minute'] < 60
minute_min_mask = new_df['minute'] > 20

final_df = new_df.loc[(minute_max_mask & minute_min_mask), ['id_partita','home', 'away', 'minute', 'home_score', 'away_score','home_final_score', 'away_final_score', 'true_values', 'predictions']]
len(final_df[final_df['predictions'] == final_df['true_values']]) /len(final_df)
final_df.sort_values(by = ['id_partita', 'minute'], ascending = [True, False]).groupby(['id_partita']).first().reset_index() 

