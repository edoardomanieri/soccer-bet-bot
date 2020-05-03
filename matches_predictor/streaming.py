from queue import Queue
from threading import Thread
from matches_predictor.API_connect import live_matches_producer
from matches_predictor.prediction import predictions_consumer


if __name__ == "__main__":
    # Create the shared queue and launch both threads 
    q1 = Queue()
    q2 = Queue()
    live_matches_thread = Thread(target=live_matches_producer, args=(q1, ))
    predictions_thread = Thread(target=predictions_consumer, args=(q1, q2))
    betfair_thread = Thread(target=producer, args=(q2, ))
    live_matches_thread.start()
    predictions_thread.start()
    betfair_thread.start()

    # Wait for all produced items to be consumed 
    q1.join()
    q2.join()
