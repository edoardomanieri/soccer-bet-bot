import betfairlightweight
from betfairlightweight import filters
import json
import pandas as pd
import os
import time
import math


def login():
    file_path = os.path.dirname(os.path.abspath(__file__))
    certs_path = f"{file_path}/../certs"

    # create trading instance
    keys = json.load(open(f"{file_path}/../keys.json"))
    trading = betfairlightweight.APIClient(username=keys['betfair-username'],
                                           password=keys['betfair-password'],
                                           app_key=keys['betfair-key-delay'],
                                           certs=certs_path,
                                           locale='italy')
    # login
    trading.login()
    return trading


def check_exposure(trading, balance, max_exposure):
    current_balance = trading.account.get_account_funds().available_to_bet_balance
    return current_balance + max_exposure > balance


def get_soccer_df(trading, in_play_only=True):
    # Filter for just soccer
    soccer_live_filter = filters.market_filter(text_query='Soccer', in_play_only=in_play_only)

    # list events -> from event get id
    # Get a list of all thoroughbred events as objects
    soccer_events = trading.betting.list_events(filter=soccer_live_filter)

    # Create a DataFrame with all the events by iterating over each event object
    soccer_events_df = pd.DataFrame({
        'Event Name': [event_object.event.name.lower() for event_object in soccer_events],
        'Home': [event_object.event.name[:event_object.event.name.find(' v ')].trim() for event_object in soccer_events],
        'Away': [event_object.event.name[event_object.event.name.find(' v ') + 3:].trim() for event_object in soccer_events],
        'Event ID': [event_object.event.id for event_object in soccer_events],
        'Event Venue': [event_object.event.venue for event_object in soccer_events],
        'Country Code': [event_object.event.country_code for event_object in soccer_events],
        'Time Zone': [event_object.event.time_zone for event_object in soccer_events],
        'Open Date': [event_object.event.open_date for event_object in soccer_events],
        'Market Count': [event_object.market_count for event_object in soccer_events]
    })

    return soccer_events_df


def get_event_id(soccer_df, runner_name_dict, prediction_obj):
    # this part is tricky because the name will not be the same
    event_id_df = soccer_df.loc[soccer_df['Event Name'].str.contains(prediction_obj.home), 'Event ID']
    if len(event_id_df) == 0:
        event_id_df = soccer_df.loc[soccer_df['Event Name'].str.contains(prediction_obj.away), 'Event ID']
    if len(event_id_df) == 0:
        event_id = 'ERR'
    else:
        event_id = event_id_df.reset_index()['Event ID'][0]

    # popolate runner name dict for 1X2
    runner_name_dict['1'] = soccer_df.loc[soccer_df['Event ID'] == event_id, 'Home']
    runner_name_dict['2'] = soccer_df.loc[soccer_df['Event ID'] == event_id, 'Away']
    return event_id


def get_selection_df(trading, event_id, prediction_obj):
    market_catalogue_filter = filters.market_filter(event_ids=[event_id])

    market_catalogues = trading.betting.list_market_catalogue(
        filter=market_catalogue_filter,
        max_results='1000',
        market_projection=['RUNNER_DESCRIPTION']
    )
    market_catalogues_list = [m for m in market_catalogues if m.market_name == prediction_obj.market_name]
    if len(market_catalogues_list) == 0:
        return None

    market_catalogue = market_catalogues_list[0]
    selection_df = pd.DataFrame({
        'Selection ID': [runner.selection_id for runner in market_catalogue.runners],
        'Runner Name': [runner.runner_name for runner in market_catalogue.runners]
    })
    selection_df['Market Name'] = prediction_obj.market_name
    selection_df['Market ID'] = market_catalogue.market_id
    return selection_df


def _process_runner_books(runner_books):
    '''
    This function processes the runner books and returns a DataFrame
    with the best back/lay prices + vol for each runner
    :param runner_books:
    :return:
    '''
    best_back_prices = [runner_book.ex.available_to_back[0].price
                        if runner_book.ex.available_to_back[0].price
                        else 1.01
                        for runner_book
                        in runner_books]
    best_back_sizes = [runner_book.ex.available_to_back[0].size
                       if runner_book.ex.available_to_back[0].size
                       else 1.01
                       for runner_book
                       in runner_books]

    best_lay_prices = [runner_book.ex.available_to_lay[0].price
                       if runner_book.ex.available_to_lay[0].price
                       else 1000.0
                       for runner_book
                       in runner_books]
    best_lay_sizes = [runner_book.ex.available_to_lay[0].size
                      if runner_book.ex.available_to_lay[0].size
                      else 1.01
                      for runner_book
                      in runner_books]

    selection_ids = [runner_book.selection_id for runner_book in runner_books]
    last_prices_traded = [runner_book.last_price_traded for runner_book in runner_books]
    total_matched = [runner_book.total_matched for runner_book in runner_books]
    statuses = [runner_book.status for runner_book in runner_books]
    scratching_datetimes = [runner_book.removal_date for runner_book in runner_books]
    adjustment_factors = [runner_book.adjustment_factor for runner_book in runner_books]

    df = pd.DataFrame({
        'Selection ID': selection_ids,
        'Best Back Price': best_back_prices,
        'Best Back Size': best_back_sizes,
        'Best Lay Price': best_lay_prices,
        'Best Lay Size': best_lay_sizes,
        'Last Price Traded': last_prices_traded,
        'Total Matched': total_matched,
        'Status': statuses,
        'Removal Date': scratching_datetimes,
        'Adjustment Factor': adjustment_factors
    })
    return df


def get_runners_df(trading, market_id):
    # Create a price filter. Get all traded and offer data
    price_filter = filters.price_projection(
        price_data=['EX_BEST_OFFERS']
    )

    # Request market books
    market_books = trading.betting.list_market_book(
        market_ids=[market_id],
        price_projection=price_filter
    )

    # Grab the first market book from the returned list as we only requested one market
    market_book = market_books[0]
    runners_df = _process_runner_books(market_book.runners)
    return runners_df


def get_bet_params(runners_df, selection_df, runner_name_dict, prediction_obj):
    # if prediction is 1 we lay 2 (we bet 1X)
    team_to_lay = '1' if prediction_obj.prediction == '2' else '2'

    selection_id = selection_df.loc[selection_df['Runner Name'] == runner_name_dict[team_to_lay], :].at[0, 'Selection ID']
    if prediction_obj.prediction in ['1', 'X', '2']:
        side = 'Lay'
    else:
        side = 'Back'
    odd = runners_df.loc[runners_df['Selection ID'] == selection_id, :].at[0, f'Best {side} Price']
    size_available = runners_df.loc[runners_df['Selection ID'] == selection_id, :].at[0, f'Best {side} Size']
    return selection_id, odd, size_available, side


def bet_algo(bet_size, bets_dict, runner_name_dict, risk_level_high,
             runners_df, selection_df, prediction_obj):
    # algo thing to modify is to avoid to bet multiple time on the same match
    selection_id, odd, size_available, side = get_bet_params(runners_df, selection_df,
                                                       runner_name_dict, prediction_obj)

    # see if there are still bets available for today (budget based)
    if odd > risk_level_high and bets_dict['high'] == 0:
        return False, 0, 0
    if odd <= risk_level_high and bets_dict['low'] == 0:
        return False, 0, 0

    # see if the odd is worth it
    print(f"weighted probability: {prediction_obj.probability}, only model probability: {prediction_obj.model_probability}, odd probability: {1/odd}")

    if prediction_obj.bet_type == 'uo':
        if prediction_obj.prediction == 'under':
            if prediction_obj.home_score + prediction_obj.away_score >= 1:
                print(f"under bet {prediction_obj.home}-{prediction_obj.away} rejected because too many goals\n")
                return False, 0, 0
            if prediction_obj.minute < 45:
                print(f"under bet {prediction_obj.home}-{prediction_obj.away} rejected because too early\n")
                return False, 0, 0
        if prediction_obj.prediction == 'over':
            if prediction_obj.home_score + prediction_obj.away_score <= 1:
                print(f"over bet {prediction_obj.home}-{prediction_obj.away} rejected because too few goals\n")
                return False, 0, 0
            if prediction_obj.minute > 45:
                print(f"over bet {prediction_obj.home}-{prediction_obj.away} rejected because too late\n")
                return False, 0, 0

    if prediction_obj.bet_type == 'ef':
        if prediction_obj.prediction == 'X':
            print(f"X bet {prediction_obj.home}-{prediction_obj.away} rejected\n")
            return False, 0, 0
        if prediction_obj.prediction == '1':
            if prediction_obj.home_score <= prediction_obj.away_score:
                print(f"1 bet {prediction_obj.home}-{prediction_obj.away} rejected because team not in advantage\n")
                return False, 0, 0
        if prediction_obj.prediction == '2':
            if prediction_obj.home_score >= prediction_obj.away_score:
                print(f"2 bet {prediction_obj.home}-{prediction_obj.away} rejected because team not in advantage\n")
                return False, 0, 0

    execute_bet = True
    # true_odd represent the true odd also for Laying bet. I will use it to
    # understand the risk associated with a bet
    if side == 'Back':
        size = size_available if size_available < bet_size else bet_size
        true_odd = odd
    else:
        # inverse formula for laying matches
        size = math.ceil(bet_size / (odd - 1))
        size = size_available if size_available < size else size
        true_odd = (size / bet_size) + 1
    size = math.floor(size)
    return execute_bet, selection_id, odd, size, side, true_odd


def place_order(trading, price, side, size, selection_id, market_id):

    # Define a limit order filter
    limit_order_filter = filters.limit_order(
        size=size,
        price=price,
        # if not filled the bet, kill it
        time_in_force='FILL_OR_KILL',
        persistence_type='LAPSE'
    )

    instructions_filter = filters.place_instruction(
        selection_id=selection_id,
        order_type="LIMIT",
        side=side.upper(),
        limit_order=limit_order_filter
    )

    # Place the order
    order = trading.betting.place_orders(
        market_id=market_id,
        instructions=[instructions_filter]
    )

    status = order.__dict__['_data']['instructionReports'][0]['status'] == 'SUCCESS'
    sizeMatched = order.__dict__['_data']['instructionReports'][0]['sizeMatched'] > 0
    return status and sizeMatched


# Decrease the number of bets remaining
def update_bets_dict(trading, bets_dict, true_odd, risk_level_high, balance, max_exposure):
    if true_odd > risk_level_high:
        bets_dict['high'] -= 1
    else:
        bets_dict['low'] -= 1
    bets_remaining = sum(bets_dict.values())
    current_balance = trading.account.get_account_funds().available_to_bet_balance
    if bets_remaining == 0 and current_balance + max_exposure > balance:
        bets_dict['high'] = 1
        bets_dict['low'] = 2


def restore_dict(trading, bets_dict, bets_dict_init, balance):
    current_balance = trading.account.get_account_funds().available_to_bet_balance
    if current_balance > balance:
        return bets_dict_init
    else:
        return bets_dict


def main(in_q, max_exposure, bets_dict_init, risk_level_high):
    runner_name_dict = {
        'under': 'Under 2.5 Goals',
        'over': 'Over 2.5 Goals',
        'X': 'The Draw'
    }
    bets_dict = dict(bets_dict_init)
    trading = login()
    balance = trading.account.get_account_funds().available_to_bet_balance
    number_bets = sum(bets_dict.values())
    # divide total exposure on number of bets
    bet_size = max_exposure / number_bets
    while True:
        if not check_exposure(trading, balance, max_exposure):
            # Empty the queue
            with in_q.mutex:
                in_q.queue.clear()
            time.sleep(120)
            continue
        bets_dict = restore_dict(trading, bets_dict, bets_dict_init, balance)
        prediction_obj = in_q.get()
        soccer_df = get_soccer_df(trading, in_play_only=True)
        event_id = get_event_id(soccer_df, runner_name_dict, prediction_obj)
        if event_id == 'ERR':
            print('not event id')
            continue
        selection_df = get_selection_df(trading, event_id, prediction_obj)
        if selection_df is None:
            print('not market id')
            continue
        market_id = selection_df.reset_index()['Market ID'][0]
        runners_df = get_runners_df(trading, market_id)
        execute_bet, selection_id, odd, size, side, true_odd = bet_algo(bet_size, bets_dict,
                                                                        runner_name_dict,
                                                                        risk_level_high,
                                                                        runners_df, selection_df,
                                                                        prediction_obj)
        if execute_bet:
            print(f"Bet to be placed on {prediction_obj.home}-{prediction_obj.away}, \
                  minute: {prediction_obj.minute}, prediction: {prediction_obj.prediction}, \
                  odd: {odd}, money: {size}\n")
            res = place_order(trading, odd, side, size, selection_id, market_id)
            if res:
                update_bets_dict(trading, bets_dict, true_odd, risk_level_high, balance, max_exposure)
