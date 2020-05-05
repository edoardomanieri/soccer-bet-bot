import betfairlightweight
from betfairlightweight import filters
import json
import pandas as pd
import os
import time


def login():
    # Change this certs path to wherever you're storing your certificates
    file_path = os.path.dirname(os.path.abspath(__file__))
    certs_path = f"{file_path}/../certs"

    # create trading instance
    keys = json.load(open("../keys.js"))
    trading = betfairlightweight.APIClient(username=keys['betfair-username'],
                                           password=keys['betfair-password'],
                                           app_key=keys['betfair-key-delay'],
                                           certs=certs_path,
                                           locale='italy')
    # login
    trading.login()
    return trading


def check_exposure(trading, max_exposure):
    exposure = trading.account.get_account_funds().exposure
    return exposure < max_exposure


def get_soccer_df(trading, in_play_only=True):
    # Filter for just soccer
    soccer_live_filter = filters.market_filter(text_query='Soccer', in_play_only=False)

    # list events -> from event get id
    # Get a list of all thoroughbred events as objects
    soccer_events = trading.betting.list_events(filter=soccer_live_filter)

    # Create a DataFrame with all the events by iterating over each event object
    soccer_events_df = pd.DataFrame({
        'Event Name': [event_object.event.name for event_object in soccer_events],
        'Event ID': [event_object.event.id for event_object in soccer_events],
        'Event Venue': [event_object.event.venue for event_object in soccer_events],
        'Country Code': [event_object.event.country_code for event_object in soccer_events],
        'Time Zone': [event_object.event.time_zone for event_object in soccer_events],
        'Open Date': [event_object.event.open_date for event_object in soccer_events],
        'Market Count': [event_object.market_count for event_object in soccer_events]
    })
    return soccer_events_df


def get_event_id(soccer_df, prediction_obj):
    event_id_df = soccer_df.loc[soccer_df['Event Name'].str.lower().contains(prediction_obj.home), 'Event ID']
    if len(event_id_df) == 0:
        event_id_df = soccer_df.loc[soccer_df['Event Name'].str.lower().contains(prediction_obj.away), 'Event ID']
    if len(event_id_df) == 0:
        event_id = 'ERR'
    else:
        event_id = event_id_df[0]
    return event_id


def get_market_id(trading, event_id, prediction_obj):
    # list market catalogue -> from market name get Market id
    market_catalogue_filter = filters.market_filter(event_ids=[event_id])

    market_catalogues = trading.betting.list_market_catalogue(
        filter=market_catalogue_filter,
        max_results='100'
    )

    # Create a DataFrame for each market catalogue
    market_df = pd.DataFrame({
        'Market Name': [market_cat_object.market_name for market_cat_object in market_catalogues],
        'Market ID': [market_cat_object.market_id for market_cat_object in market_catalogues],
        'Total Matched': [market_cat_object.total_matched for market_cat_object in market_catalogues],
    })

    market_id_df = market_df.loc[market_df['Market Name'].str.lower().contains(prediction_obj.market_name), 'Market ID']
    if len(market_id_df) == 0:
        market_id = 'ERR'
    else:
        market_id = market_id_df[0]
    return market_id


def process_runner_books(runner_books):
    '''
    This function processes the runner books and returns a DataFrame with the best back/lay prices + vol for each runner
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
    runners_df = process_runner_books(market_book.runners)
    return runners_df


def bet_algo(bet_size, runners_df, prediction_obj):
    selection_id = ''
    odd = runners_df.loc[runners_df['Selection ID'] == selection_id, 'Best Back Price'][0]
    size_available = runners_df.loc[runners_df['Selection ID'] == selection_id, 'Best Back Size'][0]
    execute_bet = True
    size = 0

    return execute_bet, selection_id, odd, size


def place_order(trading, price, size, selection_id, market_id):

    # Define a limit order filter
    limit_order_filter = filters.limit_order(
        size=size,
        price=price,
        time_in_force='FILL_OR_KILL',
        persistence_type='LAPSE'
    )

    instructions_filter = filters.place_instruction(
        selection_id=selection_id,
        order_type="LIMIT",
        side="BACK",
        limit_order=limit_order_filter
    )

    # Place the order
    order = trading.betting.place_orders(
        market_id=market_id, # The market id we obtained from before
        instructions=[instructions_filter] # This must be a list
    )

    status = order.__dict__['_data']['instructionReports'][0]['status'] == 'SUCCESS'
    sizeMatched = order.__dict__['_data']['instructionReports'][0]['sizeMatched'] > 0
    return status and sizeMatched


def main(in_q, max_exposure, number_bets):
    trading = login()
    bet_size = max_exposure / number_bets
    while True:
        if not check_exposure(trading, max_exposure):
            with in_q.mutex:
                in_q.queue.clear()
            time.sleep(120)
            continue
        prediction_obj = in_q.get()
        soccer_df = get_soccer_df(trading, in_play_only=True)
        event_id = get_event_id(soccer_df, prediction_obj)
        if event_id == 'ERR':
            continue
        market_id = get_market_id(trading, event_id, prediction_obj)
        if market_id == 'ERR':
            continue
        runners_df = get_runners_df(trading, market_id)
        execute_bet, selection_id, odd, size = bet_algo(bet_size, runners_df, prediction_obj)
        if execute_bet:
            res = place_order(trading, odd, size, selection_id, market_id)
            if res:
                print("bet placed")