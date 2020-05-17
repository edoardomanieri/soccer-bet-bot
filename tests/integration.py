from matches_predictor import API_connect
from matches_predictor.prediction import prematch_odds_based, Prediction, get_predict_proba
from matches_predictor import betfair
from matches_predictor.model import train_set, input_stream
import pandas as pd
import os
import time
from queue import Queue
from threading import Thread
import signal


def live_matches_producer(out_q, minute_threshold):
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
            out_q.put(df)
            # df_to_save = df.copy()
            # API_connect.save(df_to_save)
        print("pause..............\n")
        time.sleep(301)


def predictions_prod_cons(in_q, out_q, prob_threshold):
    file_path = os.path.dirname(os.path.abspath(__file__))
    cat_cols = ['home', 'away', 'campionato', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_uo']
    api_missing_cols = ['home_punizioni', 'away_punizioni', 'home_rimesse_laterali', 'away_rimesse_laterali',
                        'home_contrasti', 'away_contrasti', 'home_attacchi', 'away_attacchi',
                        'home_attacchi_pericolosi', 'away_attacchi_pericolosi']
    train_df = train_set.Retrieving.starting_df(cat_cols, api_missing_cols)
    train_set.Preprocessing.execute(train_df, cat_cols, api_missing_cols)
    train_df = pd.read_csv(f"{file_path}/../res/dataframes/training_goals.csv", header=0, index_col=0)
    # get clf from cross validation (dev) and retrain on all the train set
    clf = train_set.Modeling.get_dev_model()
    train_set.Modeling.train_model(train_df, clf, cat_cols, outcome_cols, prod=True)
    clf = train_set.Modeling.get_prod_model()

    while True:
        input_df = pd.DataFrame(in_q.get())
        input_df.drop(columns=['fixture_id'], inplace=True)
        input_prematch_odds = input_stream.Preprocessing.execute(input_df,
                                                                 train_df,
                                                                 cat_cols)
        test_X = input_df.drop(columns=cat_cols)
        get_predict_proba(clf, test_X, input_df)
        predictions_df = prematch_odds_based(input_df, input_prematch_odds)
        minute = predictions_df.loc[:, 'minute'][0]
        home = predictions_df.loc[:, 'home'][0]
        away = predictions_df.loc[:, 'away'][0]
        market_name = 'over 2.5'
        prediction = predictions_df.loc[:, 'prediction_final'][0]
        probability = predictions_df.loc[:, 'probability_final_over'][0]
        model_probability = predictions_df.loc[:, 'probability_over'][0]
        prediction_obj = Prediction(minute, home, away, market_name, prediction, probability, model_probability)
        if prediction_obj.probability > prob_threshold:
            out_q.put(prediction_obj)
        print(f"{prediction_obj.home}-{prediction_obj.away}, \
              minute: {prediction_obj.minute}, \
              probability: {prediction_obj.probability}, \
              model_probability: {prediction_obj.model_probability}, \
              eventual prediction: {prediction_obj.prediction}\n")


def betfair_consumer(in_q, max_exposure, bets_dict_init, risk_level_high,
                     risk_level_medium):
    bets_dict = dict(bets_dict_init)
    trading = betfair.login()
    balance = trading.account.get_account_funds().available_to_bet_balance
    number_bets = sum(bets_dict.values())
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
        print("trying to bet.....\n")
        soccer_df = betfair.get_soccer_df(trading, in_play_only=True)
        event_id = betfair.get_event_id(soccer_df, prediction_obj)
        if event_id == 'ERR':
            print('not event id')
            continue
        market_id = betfair.get_market_id(trading, event_id, prediction_obj)
        if market_id == 'ERR':
            print('not market id')
            continue
        runners_df = betfair.get_runners_df(trading, market_id)
        execute_bet, selection_id, odd, size = betfair.bet_algo(bet_size, bets_dict,
                                                                risk_level_high, risk_level_medium,
                                                                runners_df, prediction_obj)
        if execute_bet:
            print(f"Bet to be placed on {prediction_obj.home}-{prediction_obj.away}, \
                  minute: {prediction_obj.minute}, prediction: {prediction_obj.prediction}, \
                  odd: {odd}, money: {size}\n")
            betfair.update_bets_dict(trading, bets_dict, odd, risk_level_high, risk_level_medium, balance, max_exposure)


if __name__ == "__main__":
    # Create the shared queue and launch both threads
    q1 = Queue()
    q2 = Queue()
    # params
    minute_threshold = 20
    probability_threshold = 0.7
    bets_dict = {'high': 1, 'medium': 2, 'low': 4}
    risk_level_high = 1.6
    risk_level_medium = 1.2
    max_exposure = 14
    try:
        live_matches_thread = Thread(target=live_matches_producer, args=(q1, minute_threshold, ))
        predictions_thread = Thread(target=predictions_prod_cons, args=(q1, q2, probability_threshold, ))
        betfair_thread = Thread(target=betfair_consumer, args=(q2, max_exposure, bets_dict, risk_level_high, risk_level_medium, ))
        live_matches_thread.start()
        predictions_thread.start()
        betfair_thread.start()
        signal.pause()
    except KeyboardInterrupt:
        print('\n! Received keyboard interrupt, quitting threads.\n')
        # API_connect.ended_matches()
