from matches_predictor.apifootball import utils
import pandas as pd
import time
import logging


def filter(match, minute_threshold):
    if match['minute'] < minute_threshold:
        return False
    if match['home_score'] + match['away_score'] > 2:
        return False
    return True


def run(queue, minute_threshold):
    label_dict = utils.get_label_dict()

    while True:
        matches_list = utils.get_basic_info()

        for match in matches_list:

            logging.info(f"match: {match['home']}-{match['away']}\n")
            resp = utils.get_match_statistics(match['fixture_id'])
            stat_present = utils.stat_to_dict(resp, match)

            if not stat_present:
                logging.info("no statistics available")
                continue

            resp = utils.get_prematch_odds(match['fixture_id'], label_dict['Goals Over/Under'])
            utils.prematch_odds_uo_to_dict(resp, match)

            resp = utils.get_prematch_odds(match['fixture_id'], label_dict['Match Winner'])
            utils.prematch_odds_1x2_to_dict(resp, match)

            df = pd.DataFrame([match])

            # put the api on the queue
            queue.put(df)
        logging.info("pause..............\n")
        time.sleep(301)
