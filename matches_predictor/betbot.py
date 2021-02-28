from queue import Queue
from threading import Thread
import signal
from matches_predictor import prediction_thread, apifootball_thread


def start_threads(threads):
    for thread in threads:
        thread.start()
    signal.pause()


def main():
    queue = Queue()
    minute_threshold = 30
    probability_threshold = 0.7
    try:
        live_matches_thread = Thread(target=apifootball_thread.run, args=(queue, minute_threshold, ))
        predictions_thread = Thread(target=prediction_thread.run, args=(queue,  probability_threshold, ))
        start_threads([live_matches_thread, predictions_thread])
    except KeyboardInterrupt:
        print('\n! Received keyboard interrupt, quitting threads.\n')


if __name__ == "__main__":
    main()
