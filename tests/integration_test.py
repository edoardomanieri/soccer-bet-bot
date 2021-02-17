import pandas as pd
import os
import time
from queue import Queue
from threading import Thread
import signal
from matches_predictor import prediction_thread


def mock_live_matches_producer(out_q, minute_threshold):
    file_path = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(f"{file_path}/api_df.csv", index_col=False)
    while True:
        for _ in range(5):
            out_q.put(df)
        time.sleep(301)

def _start_threads(threads):
    for thread in threads:
        thread.start()
    signal.pause()


def main():
    q1 = Queue()
    minute_threshold = 30
    probability_threshold = 0.7
    try:
        live_matches_thread = Thread(target=mock_live_matches_producer, args=(q1, minute_threshold, ))
        predictions_thread = Thread(target=prediction_thread.run, args=(q1, probability_threshold, ))
        _start_threads([live_matches_thread, predictions_thread])
    except KeyboardInterrupt:
        print('\n! Received keyboard interrupt, quitting threads.\n')


def test_main():
    pass
    

if __name__ == "__main__":
    main()
