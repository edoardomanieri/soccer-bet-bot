from matches_predictor import API_connect
from matches_predictor import betfair
from matches_predictor.models import uo, ef
import pandas as pd
import numpy as np
import os
import glob
import time
from queue import Queue
from threading import Thread
import signal
import sqlite3


def live_matches_producer(out_q1, out_q2, minute_threshold, conn, curs):
    n_api_call = 0
    label_dict = API_connect.get_label_dict()
    n_api_call += 1
    while True:
        matches_list = API_connect.get_basic_info()
        n_api_call += 1
        for match in matches_list:
            if match['minute'] < minute_threshold:
                continue
            if match['home_score'] + match['away_score'] > 2:
                continue
            print(f"match: {match['home']}-{match['away']}\n")
            resp = API_connect.get_match_statistics(match['fixture_id'])
            stat_present = API_connect.stat_to_dict(resp, match)
            if not stat_present:
                print("no statistics available")
                continue
            resp = API_connect.get_prematch_odds(match['fixture_id'], label_dict['Goals Over/Under'])
            API_connect.prematch_odds_uo_to_dict(resp, match)
            resp = API_connect.get_prematch_odds(match['fixture_id'], label_dict['Match Winner'])
            n_api_call += 3
            API_connect.prematch_odds_1x2_to_dict(resp, match)
            df = pd.DataFrame([match])
            out_q1.put(df)
            out_q2.put(df)
            df_to_save = df.copy()
            API_connect.save(df_to_save, conn, curs)
        if len(matches_list) == 0:
            print("no matches available \n")
        print("pause..............\n")
        time.sleep(301)


def predictions_prod_cons_uo(in_q, out_q, prob_threshold, conn):
    file_path = os.path.dirname(os.path.abspath(__file__))
    cat_cols = ['home', 'away', 'campionato', 'date', 'id_partita']
    to_drop_cols = ['home', 'away', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_uo']
    api_missing_cols = ['home_punizioni', 'away_punizioni', 'home_rimesse_laterali', 'away_rimesse_laterali',
                        'home_contrasti', 'away_contrasti', 'home_attacchi', 'away_attacchi',
                        'home_attacchi_pericolosi', 'away_attacchi_pericolosi']
    train_df = uo.train_set.Retrieving.starting_df(cat_cols, api_missing_cols, conn)
    uo.train_set.Preprocessing.execute(train_df, cat_cols, api_missing_cols)
    train_df = pd.read_csv(f"{file_path}/../res/dataframes/training_uo.csv", header=0, index_col=0)
    # get clf from cross validation (dev) and retrain on all the train set
    clf = uo.train_set.Modeling.get_dev_model()
    cols_used = uo.train_set.Modeling.train_model(train_df, clf, to_drop_cols, outcome_cols, prod=True)
    clf = uo.train_set.Modeling.get_prod_model()

    while True:
        input_df = pd.DataFrame(in_q.get())
        input_df.drop(columns=['fixture_id'], inplace=True)
        prematch_odds = uo.input_stream.Preprocessing.execute(input_df,
                                                              train_df,
                                                              cat_cols)
        input_X = input_df[cols_used]
        uo.prediction.get_predict_proba(clf, input_X, input_df)
        predictions_df = uo.prediction.prematch_odds_based(input_df, prematch_odds)
        prediction_obj = uo.prediction.Prediction(predictions_df)
        output_df = predictions_df.loc[:, ['minute', 'home', 'away', 'prediction_final']]
        output_df['probability_final'] = np.where(predictions_df['probability_final_over'] > 0.5,
                                                  predictions_df['probability_final_over'],
                                                  1 - predictions_df['probability_final_over'])
        output_df['probability'] = np.where(predictions_df['probability_over'] > 0.5,
                                            predictions_df['probability_over'],
                                            1 - predictions_df['probability_over'])
        output_df['bet_type'] = 'uo'
        output_df.to_csv(f"{file_path}/../dash/uo{prediction_obj.minute}{prediction_obj.home}.csv")
        if prediction_obj.probability > prob_threshold:
            out_q.put(prediction_obj)
        print(f"{prediction_obj.home}-{prediction_obj.away}, \
              minute: {prediction_obj.minute}, \
              probability: {prediction_obj.probability}, \
              model_probability: {prediction_obj.model_probability}, \
              eventual prediction: {prediction_obj.prediction}\n")


def predictions_prod_cons_ef(in_q, out_q, prob_threshold, conn):
    file_path = os.path.dirname(os.path.abspath(__file__))
    cat_cols = ['home', 'away', 'campionato', 'date', 'id_partita']
    to_drop_cols = ['home', 'away', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_ef']
    api_missing_cols = ['home_punizioni', 'away_punizioni', 'home_rimesse_laterali',
                        'away_rimesse_laterali', 'home_contrasti', 'away_contrasti',
                        'home_attacchi', 'away_attacchi', 'home_attacchi_pericolosi',
                        'away_attacchi_pericolosi']
    train_df = ef.train_set.Retrieving.starting_df(cat_cols, api_missing_cols, conn)
    ef.train_set.Preprocessing.execute(train_df, cat_cols, api_missing_cols)
    train_df = pd.read_csv(f"{file_path}/../res/dataframes/training_ef.csv", header=0, index_col=0)
    # get clf from cross validation (dev) and retrain on all the train set
    clf = ef.train_set.Modeling.get_dev_model()
    cols_used = ef.train_set.Modeling.train_model(train_df, clf, to_drop_cols, outcome_cols, prod=True)
    clf = ef.train_set.Modeling.get_prod_model()

    while True:
        input_df = pd.DataFrame(in_q.get())
        input_df.drop(columns=['fixture_id'], inplace=True)
        prematch_odds = ef.input_stream.Preprocessing.execute(input_df,
                                                              train_df,
                                                              cat_cols)
        input_X = input_df[cols_used]
        ef.prediction.get_predict_proba(clf, input_X, input_df)
        predictions_df = ef.prediction.prematch_odds_based(input_df, prematch_odds)
        prediction_obj = ef.prediction.Prediction(predictions_df)
        output_df = predictions_df.loc[:, ['minute', 'home', 'away',
                                           'prediction_final', 'probability_final',
                                           'probability']]
        output_df['bet_type'] = 'ef'
        output_df.to_csv(f"{file_path}/../dash/ef{prediction_obj.minute}{prediction_obj.home}.csv")
        if prediction_obj.probability > prob_threshold:
            out_q.put(prediction_obj)
        print(f"{prediction_obj.home}-{prediction_obj.away}, \
              minute: {prediction_obj.minute}, \
              probability: {prediction_obj.probability}, \
              model_probability: {prediction_obj.model_probability}, \
              eventual prediction: {prediction_obj.prediction}\n")



def main(in_q, max_exposure, bets_dict_init, risk_level_high):
    runner_name_dict = {
        'under': 'Under 2.5 Goals',
        'over': 'Over 2.5 Goals',
        'X': 'The Draw'
    }
    bets_dict = dict(bets_dict_init)
    trading = betfair.login()
    balance = trading.account.get_account_funds().available_to_bet_balance
    number_bets = sum(bets_dict.values())
    # divide total exposure on number of bets
    bet_size = max_exposure / number_bets
    while True:
        if not betfair.check_exposure(trading, balance, max_exposure):
            # Empty the queue
            with in_q.mutex:
                in_q.queue.clear()
            time.sleep(120)
            continue
        bets_dict = betfair.restore_dict(trading, bets_dict, bets_dict_init, balance)
        prediction_obj = in_q.get()
        soccer_df = betfair.get_soccer_df(trading, in_play_only=True)
        event_id = betfair.get_event_id(soccer_df, runner_name_dict, prediction_obj)
        if event_id == 'ERR':
            print('not event id')
            continue
        selection_df = betfair.get_selection_df(trading, event_id, prediction_obj)
        if selection_df is None:
            print('not market id')
            continue
        market_id = selection_df.reset_index()['Market ID'][0]
        runners_df = betfair.get_runners_df(trading, market_id)
        execute_bet, selection_id, odd, size, side = betfair.bet_algo(bet_size, bets_dict,
                                                                      runner_name_dict,
                                                                      risk_level_high,
                                                                      runners_df, selection_df,
                                                                      prediction_obj)
        if execute_bet:
            print(f"Bet to be placed on {prediction_obj.home}-{prediction_obj.away}, \
                  minute: {prediction_obj.minute}, prediction: {prediction_obj.prediction}, \
                  odd: {odd}, money: {size}\n")
            res = betfair.place_order(trading, odd, side, size, selection_id, market_id)
            if res:
                betfair.update_bets_dict(trading, bets_dict, odd, risk_level_high, balance, max_exposure)

if __name__ == "__main__":
    # Create the shared queue and launch both threads
    file_path = os.path.dirname(os.path.abspath(__file__))
    conn = sqlite3.connect(f"{file_path}/../../res/football.db")
    curs = conn.cursor()
    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    # params
    minute_threshold = 20
    probability_threshold = 0.7
    bets_dict = {'high': 2, 'low': 8}
    risk_level_high = 1.45
    max_exposure = 100
    try:
        # remove previous files
        files = glob.glob(f"{file_path}/../dash/*")
        for f in files:
            os.remove(f)
        live_matches_thread = Thread(target=live_matches_producer, args=(q1, q2, minute_threshold, conn, curs, ))
        predictions_uo_thread = Thread(target=predictions_prod_cons_uo, args=(q1, q3, probability_threshold, conn ))
        predictions_ef_thread = Thread(target=predictions_prod_cons_ef, args=(q2, q3, probability_threshold, conn ))
        # betfair_thread = Thread(target=betfair_consumer, args=(q3, max_exposure, bets_dict, risk_level_high, ))
        live_matches_thread.start()
        predictions_uo_thread.start()
        predictions_ef_thread.start()
        # betfair_thread.start()
        signal.pause()
    except KeyboardInterrupt:
        print('\n! Received keyboard interrupt, quitting threads.\n')
        conn.close()
        # API_connect.ended_matches()
