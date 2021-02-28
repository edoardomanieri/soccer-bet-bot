import pandas as pd
import os
import time
from queue import Queue
from threading import Thread
import signal
from unittest import TestCase, main, mock
from matches_predictor import betbot, telegram


class BetBotTestCase(TestCase):

    def live_matches_producer(self, out_q, minute_threshold):
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

    def test_telegram(self):
        telegram.telegram_bot_sendtext("ciao")


    @mock.patch('matches_predictor.apifootball_thread.run')
    def test_main(self, mock_run):
        mock_run.side_effect = self.live_matches_producer
        betbot.main()


if __name__ == "__main__":
    main()
