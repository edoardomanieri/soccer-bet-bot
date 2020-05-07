from queue import Queue
from threading import Thread
import signal
from matches_predictor.API_connect import live_matches_producer
from matches_predictor.prediction import predictions_consumer
from matches_predictor.betfair import main


if __name__ == "__main__":
    # Create the shared queue and launch both threads
    q1 = Queue()
    q2 = Queue()
    # params
    minute_threshold = 30
    probability_threshold = 0.7
    bets_dict = {'high': 1, 'medium': 2, 'low': 3}
    risk_level_high = 1.6
    risk_level_medium = 1.2
    max_exposure = 12
    try:
        live_matches_thread = Thread(target=live_matches_producer, args=(q1, minute_threshold, ))
        predictions_thread = Thread(target=predictions_consumer, args=(q1, q2, probability_threshold, ))
        betfair_thread = Thread(target=main, args=(q2, max_exposure, bets_dict, risk_level_high, risk_level_medium, ))
        live_matches_thread.start()
        predictions_thread.start()
        betfair_thread.start()
        signal.pause()
    except KeyboardInterrupt:
        print('\n! Received keyboard interrupt, quitting threads.\n')

