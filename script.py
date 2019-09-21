import numpy as np
import pandas as pd
import glob
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import model





#import dataset
all_files = glob.glob("../csv/*.csv")
li = [pd.read_csv(filename, index_col=None, header=0) for filename in all_files]
df = pd.concat(li, axis=0, ignore_index=True)
cat_col = ['home', 'away', 'campionato', 'date', 'id_partita']


#change data type
for col in df.columns:
    if col not in cat_col:
        df[col] = pd.to_numeric(df[col])

#nan imputation
nan_cols = [i for i in df.columns if df[i].isnull().any()]
for col in nan_cols:
    df = model.nan_imputation(df[(~df['home_' + col[5:]].isnull()) & (~df['away_' + col[5:]].isnull())], df,col)


#dropna
# nan_col = ['home_rimesse_laterali', 'away_rimesse_laterali', 'home_tiri_fermati', 'away_tiri_fermati',\
#     'home_punizioni', 'away_punizioni', 'home_passaggi_totali', 'away_passaggi_totali', 'home_passaggi_completati', 'away_passaggi_completati', 'home_contrasti', 'away_contrasti']
# df.drop(columns = nan_col, inplace = True)
# df.dropna(inplace = True)

#splittare test and train in modo che nel train ci siano alcune partite, nel test altre
id_partita_test = np.random.choice(df['id_partita'].unique(), size = len(df['id_partita'].unique()) // 4, replace = False)
test_mask = df['id_partita'].isin(id_partita_test)
test = df.loc[test_mask, :].copy(deep =  True)
train = df.loc[~(test_mask), :].copy(deep = True)

#adding outcome columns
train['result'] = np.where(train['home_final_score'] > train['away_final_score'], 1, np.where(train['home_final_score'] == train['away_final_score'], 2, 3))
train['final_total_goals'] = train['home_final_score'] + train['away_final_score']

#adding additional information
train['actual_result'] = np.where(train['home_score'] > train['away_score'], 1, np.where(train['home_score'] == train['away_score'], 2, 3))
train['result_strongness'] = (train['home_score'] - train['away_score']) * train['minute']

campionati = train['campionato'].unique()
train['avg_camp_goals'] = 0

df_matches = train[['home', 'away', 'campionato', 'home_final_score', 'away_final_score', 'id_partita', 'final_total_goals']].groupby('id_partita').first().reset_index()
for camp in campionati:
    train.loc[train['campionato'] == camp,'avg_camp_goals'] = df_matches.loc[df_matches['campionato'] == camp,'final_total_goals'].mean()

train['home_avg_goal_fatti'] = 0
train['away_avg_goal_fatti'] = 0

train['home_avg_goal_subiti'] = 0
train['away_avg_goal_subiti'] = 0

squadre = set((train['home'].unique().tolist() + train['away'].unique().tolist()))

for team in squadre:
    n_match_home = len(df_matches[df_matches['home'] == team])
    n_match_away = len(df_matches[df_matches['away'] == team])

    sum_home_fatti = df_matches.loc[(df_matches['home'] == team),'home_final_score'].sum()
    sum_away_fatti = df_matches.loc[(df_matches['away'] == team),'away_final_score'].sum()

    #divide by 0
    if (n_match_home + n_match_away) == 0:
        n_match_away += 1

    train.loc[train['home'] == team,'home_avg_goal_fatti'] = (sum_home_fatti + sum_away_fatti) / (n_match_home + n_match_away)
    train.loc[train['away'] == team,'away_avg_goal_fatti'] = (sum_home_fatti + sum_away_fatti) / (n_match_home + n_match_away)

    sum_home_subiti = df_matches.loc[(df_matches['home'] == team),'away_final_score'].sum()
    sum_away_subiti = df_matches.loc[(df_matches['away'] == team),'home_final_score'].sum()

    train.loc[train['home'] == team,'home_avg_goal_subiti'] = (sum_home_subiti + sum_away_subiti) / (n_match_home + n_match_away)
    train.loc[train['away'] == team,'away_avg_goal_subiti'] = (sum_home_subiti + sum_away_subiti) / (n_match_home + n_match_away)

test = model.process_test_data(train, test)

test.drop(columns = cat_col, inplace = True)
train.drop(columns = cat_col, inplace = True)

drop_y_column = ['home_final_score', 'away_final_score', 'result', 'final_total_goals']
tmp_y_col_to_be_dropped = ['home_avg_goal_fatti', 'away_avg_goal_fatti', 'home_avg_goal_subiti', 'away_avg_goal_subiti', 'avg_camp_goals']

test_y = test['result'].values
test_X = test.drop(columns = drop_y_column)
test_X = test_X.drop(columns = tmp_y_col_to_be_dropped)

train_y = train['result'].values
train_X = train.drop(columns = drop_y_column)
train_X = train_X.drop(columns = tmp_y_col_to_be_dropped)

#baseline model
lr = LogisticRegression(solver = 'lbfgs', multi_class = 'auto', max_iter = 10000)
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)
probabilities = lr.predict_proba(test_X)
accuracy_score(test_y, predictions)

#randomforest
rf = RandomForestClassifier(n_estimators = 500)
rf.fit(train_X, train_y)
predictions = rf.predict(test_X)
probabilities = rf.predict_proba(test_X)
accuracy_score(test_y, predictions)

#xgb
xgb = XGBClassifier(n_estimators = 2000)
xgb.fit(train_X, train_y)
predictions = xgb.predict(test_X)
probabilities = xgb.predict_proba(test_X)
accuracy_score(test_y, predictions)

test['predictions'] = predictions
test['probability'] = np.max(probabilities, axis = 1)
test['true_values'] = test_y 
new_df = test.merge(df, how = "left")
camp_mask = new_df['campionato'] == 'serie_a'
prob_mask = new_df['probability'] >= 0.90
minute_max_mask = new_df['minute'] < 60
minute_min_mask = new_df['minute'] > 20
score_mask = new_df['home_score'] == new_df['away_score']
no_draws_mask = new_df['predictions'] != 2

final_df = new_df.loc[(prob_mask & minute_max_mask & minute_min_mask & score_mask & no_draws_mask), ['home', 'away', 'minute', 'home_score', 'away_score','home_final_score', 'away_final_score', 'probability', 'true_values', 'predictions']].sort_values(by = 'probability', ascending = False, inplace = False)
len(final_df[final_df['predictions'] == final_df['true_values']]) /len(final_df)
final_df

final_df2 = new_df.loc[prob_mask, ['home', 'away', 'minute', 'home_score', 'away_score','home_final_score', 'away_final_score', 'probability', 'true_values', 'predictions']].sort_values(by = 'probability', ascending = False, inplace = False)
len(final_df2[final_df2['predictions'] == final_df2['true_values']]) /len(final_df2)
final_df2




    # for m in test_df.loc[nan_mask, 'minute'].values:
    #     lower = m - 10 if m - 10 >= 0 else 0
    #     upper = m + 10 if m + 10 <= 90 else 90
    #     mask_min = train_df['minute'] > lower
    #     mask_max = train_df['minute'] < upper
    #     if home_away_col:
    #         test_df.loc[(test_df['minute'] == m) & (nan_mask), 'home_' + col] = train_df.loc[mask_min & mask_max, ['home_' + col, 'away_' + col]].mean().mean()
    #         test_df.loc[(test_df['minute'] == m) & (nan_mask), 'away_' + col] = train_df.loc[mask_min & mask_max, ['home_' + col, 'away_' + col]].mean().mean()
    #     else:
    #         test_df.loc[(test_df['minute'] == m) & (nan_mask), col] = train_df.loc[mask_min & mask_max, col].mean()
    # return test_df

