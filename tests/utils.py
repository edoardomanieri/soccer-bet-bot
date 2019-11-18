import numpy as np


def nan_imputation(train_df, test_df, thresh = 'half'):

    #eliminate duplicate rows
    subset = [col for col in test_df.columns if col != 'minute']
    test_df.drop_duplicates(subset = subset, inplace = True)

    #eliminate rows with a lot of nans
    if thresh == 'half':
        thresh = len(test_df.columns) // 2
    test_df.dropna(axis = 0, thresh = thresh, inplace = True)

    #handling odds cols
    if all(['odd_over', 'odd_under','odd_1', 'odd_X', 'odd_2']) in test_df.columns:
        test_df.loc[test_df['odd_under'] == 0, 'odd_under'] = 2
        test_df.loc[test_df['odd_over'] == 0, 'odd_over'] = 2
        test_df.loc[test_df['odd_1'] == 0, 'odd_1'] = 3
        test_df.loc[test_df['odd_X'] == 0, 'odd_X'] = 3
        test_df.loc[test_df['odd_2'] == 0, 'odd_2'] = 3

    #imputing the other nans
    nan_cols = [i for i in test_df.columns if test_df[i].isnull().any() if i not in ['home_final_score', 'away_final_score']]
    for col in nan_cols:

        col_df = train_df[(~train_df['home_' + col[5:]].isnull()) & (~train_df['away_' + col[5:]].isnull())]
    
        if 'away' in col:
            continue

        col = col[5:]
        nan_mask = test_df['home_' + col].isnull() | test_df['away_' + col].isnull()

        if "possesso_palla" in col:
            test_df.loc[nan_mask, 'home_possesso_palla'] = 50
            test_df.loc[nan_mask, 'away_possesso_palla'] = 50
            continue

        for m in np.arange(5, 90, 5):
            mask_min_test = test_df['minute'] >= m
            mask_max_test = test_df['minute'] <= m + 5
            mask_min_train = col_df['minute'] >= m
            mask_max_train = col_df['minute'] <= m + 5
            test_df.loc[(mask_min_test) & (mask_max_test) & (nan_mask), 'home_' + col] = col_df.loc[mask_min_train & mask_max_train, ['home_' + col, 'away_' + col]].mean().mean()
            test_df.loc[(mask_min_test) & (mask_max_test) & (nan_mask), 'away_' + col] = col_df.loc[mask_min_train & mask_max_train, ['home_' + col, 'away_' + col]].mean().mean()
    
    test_df.dropna(inplace = True)
    return test_df

