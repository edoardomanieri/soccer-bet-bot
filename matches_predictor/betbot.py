from queue import Queue
from threading import Thread
import signal
from matches_predictor.API_connect import live_matches_producer, ended_matches
from matches_predictor import prediction_thread


def start_threads(threads):
    for thread in threads:
        thread.start()
    signal.pause()


def main():
    q1 = Queue()
    minute_threshold = 30
    probability_threshold = 0.01
    try:
        live_matches_thread = Thread(target=live_matches_producer, args=(q1, minute_threshold, ))
        predictions_thread = Thread(target=prediction_thread.run, args=(q1,  probability_threshold, ))
        start_threads([live_matches_thread, predictions_thread])
    except KeyboardInterrupt:
        print('\n! Received keyboard interrupt, quitting threads.\n')


if __name__ == "__main__":
    main()
