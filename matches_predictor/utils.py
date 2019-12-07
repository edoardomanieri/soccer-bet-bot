import numpy as np


def get_complete_predictions_df(input_df):
    final_df = input_df.loc[:, ['id_partita', 'home', 'away', 'minute', 'home_score',
                                'away_score', 'predictions', 'probability_over']]\
                       .sort_values(by=['id_partita', 'minute'], ascending=[True, False])\
                       .groupby(['id_partita']).first().reset_index().copy()
    return final_df


def get_posterior_predictions(input_pred_df, input_prematch_odds_df):
    # al 15 minuto probabilità pesate 50-50
    rate = 0.6 / 90
    res_df = input_pred_df.merge(input_prematch_odds_df, on=['id_partita', 'minute'])
    res_df['probability_final_over'] = ((0.4 + (rate*res_df['minute'])) * res_df['probability_over'])\
                                        + ((0.6 - (rate*res_df['minute'])) * res_df['odd_over'])
    res_df['probability_final_under'] = ((0.4 + (rate*res_df['minute'])) * (1-res_df['probability_over']))\
                                        + ((0.6 - (rate*res_df['minute'])) * res_df['odd_under'])
    res_df['prediction_final_over'] = np.argmax(res_df[['probability_final_over', 'probability_final_under']].values, axis=1)
    res_df['prediction_final'] = np.where(res_df['prediction_final_over'] == 0, 'over', 'under')
    return res_df
