import requests
import json
import pandas as pd
import numpy as np
import time
import random
import os


def extract_values(obj, key):
    """Pull all values of specified key from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results


def get_basic_info():
    file_path = os.path.dirname(os.path.abspath(__file__))
    keys = json.load(open(f"{file_path}/../../keys.json"))
    url = f"http://{keys['x-rapidapi-host']}/{keys['version']}fixtures/live"
    headers = {
        'x-rapidapi-host': keys['x-rapidapi-host'],
        'x-rapidapi-key': keys['x-rapidapi-key']
        }
    response = requests.request("GET", url, headers=headers)
    list_of_matches = basic_info_to_dict(response.content)
    return list_of_matches


def basic_info_to_dict(response):
    res = []
    resp_dict = json.loads(response)
    for resp in resp_dict['api']['fixtures']:
        out_dict = {}
        out_dict['fixture_id'] = resp['fixture_id']
        # campionato identified as league name-country
        out_dict['campionato'] = f"{resp['league']['name']}-{resp['league']['country']}"
        out_dict['date'] = resp['event_date']
        out_dict['minute'] = resp['elapsed']
        out_dict['home'] = resp['homeTeam']['team_name'].lower()
        out_dict['away'] = resp['awayTeam']['team_name'].lower()
        out_dict['home_score'] = int(resp['goalsHomeTeam'])
        out_dict['away_score'] = int(resp['goalsAwayTeam'])
        out_dict['id_partita'] = f"{resp['event_date']}-{resp['fixture_id']}"
        res.append(out_dict)
    return res


def get_label_dict():
    file_path = os.path.dirname(os.path.abspath(__file__))
    keys = json.load(open(f"{file_path}/../../keys.json"))
    url = f"http://{keys['x-rapidapi-host']}/{keys['version']}odds/labels/"
    headers = {
        'x-rapidapi-host': keys['x-rapidapi-host'],
        'x-rapidapi-key': keys['x-rapidapi-key']
        }
    response = requests.request("GET", url, headers=headers)
    resp_dict = json.loads(response.content)
    labels = resp_dict['api']['labels']
    label_dict = {}
    for label in labels:
        label_dict[label['label']] = label['id']
    return label_dict


def get_prematch_odds(fixture_id, label_id):
    file_path = os.path.dirname(os.path.abspath(__file__))
    keys = json.load(open(f"{file_path}/../../keys.json"))
    url = f"http://{keys['x-rapidapi-host']}/{keys['version']}odds/fixture/{fixture_id}/label/{label_id}"
    headers = {
        'x-rapidapi-host': keys['x-rapidapi-host'],
        'x-rapidapi-key': keys['x-rapidapi-key']
        }
    response = requests.request("GET", url, headers=headers)
    return response.content


def prematch_odds_uo_to_dict(response, match_dict):
    resp_dict = json.loads(response)
    if 'api' in resp_dict.keys() and 'odds' in resp_dict['api'].keys():
        bets = resp_dict['api']['odds'][0]['bookmakers'][0]['bets'][0]['values']
        for bet in bets:
            if bet['value'] == 'Over 2.5':
                match_dict['odd_over'] = float(bet['odd'])
            if bet['value'] == 'Under 2.5':
                match_dict['odd_under'] = float(bet['odd'])
    if 'odd_over' not in match_dict.keys():
        print("Over 2.5 odd not found \n")
        # 0 is nan value for odds
        match_dict['odd_over'] = 0
    if 'odd_under' not in match_dict.keys():
        print("Under 2.5 odd not found \n")
        # 0 is nan value for odds
        match_dict['odd_under'] = 0


def prematch_odds_1x2_to_dict(response, match_dict):
    resp_dict = json.loads(response)
    if 'api' in resp_dict.keys() and 'odds' in resp_dict['api'].keys():
        bets = resp_dict['api']['odds'][0]['bookmakers'][0]['bets'][0]['values']
        for bet in bets:
            if bet['value'] == 'Home':
                match_dict['odd_1'] = float(bet['odd'])
            if bet['value'] == 'Draw':
                match_dict['odd_X'] = float(bet['odd'])
            if bet['value'] == 'Away':
                match_dict['odd_2'] = float(bet['odd'])
    if 'odd_1' not in match_dict.keys():
        print("1 odd not found \n")
        # 0 is nan value for odds
        match_dict['odd_1'] = 0
    if 'odd_X' not in match_dict.keys():
        print("X odd not found \n")
        # 0 is nan value for odds
        match_dict['odd_X'] = 0
    if 'odd_2' not in match_dict.keys():
        print("2 odd not found \n")
        # 0 is nan value for odds
        match_dict['odd_2'] = 0


def get_match_statistics(fixture_id):
    file_path = os.path.dirname(os.path.abspath(__file__))
    keys = json.load(open(f"{file_path}/../../keys.json"))
    url = f"http://{keys['x-rapidapi-host']}/{keys['version']}statistics/fixture/{fixture_id}"
    headers = {
        'x-rapidapi-host': keys['x-rapidapi-host'],
        'x-rapidapi-key': keys['x-rapidapi-key']
        }
    response = requests.request("GET", url, headers=headers)
    return response.content


def stat_to_dict(response, match_dict):
    resp_dict = json.loads(response)
    if 'api' not in resp_dict.keys() or 'statistics' not in resp_dict['api'].keys():
        return False
    stats = resp_dict['api']['statistics']
    if len(stats) == 0:
        return False
    value = stats['Ball Possession']['home']
    match_dict['home_possesso_palla'] = 50 if value is None else int(value.replace("%", ""))
    value = stats['Ball Possession']['away']
    match_dict['away_possesso_palla'] = 50 if value is None else int(value.replace("%", ""))
    value = stats['Total Shots']['home']
    match_dict['home_tiri'] = 0 if value is None else int(value)
    value = stats['Total Shots']['away']
    match_dict['away_tiri'] = 0 if value is None else int(value)
    value = stats['Shots on Goal']['home']
    match_dict['home_tiri_in_porta'] = 0 if value is None else int(value)
    value = stats['Shots on Goal']['away']
    match_dict['away_tiri_in_porta'] = 0 if value is None else int(value)
    value = stats['Shots off Goal']['home']
    match_dict['home_tiri_fuori'] = 0 if value is None else int(value)
    value = stats['Shots off Goal']['away']
    match_dict['away_tiri_fuori'] = 0 if value is None else int(value)
    value = stats['Blocked Shots']['home']
    match_dict['home_tiri_fermati'] = 0 if value is None else int(value)
    value = stats['Blocked Shots']['away']
    match_dict['away_tiri_fermati'] = 0 if value is None else int(value)
    value = stats['Corner Kicks']['home']
    match_dict['home_calci_d_angolo'] = 0 if value is None else int(value)
    value = stats['Corner Kicks']['away']
    match_dict['away_calci_d_angolo'] = 0 if value is None else int(value)
    value = stats['Offsides']['home']
    match_dict['home_fuorigioco'] = 0 if value is None else int(value)
    value = stats['Offsides']['away']
    match_dict['away_fuorigioco'] = 0 if value is None else int(value)
    value = stats['Goalkeeper Saves']['home']
    match_dict['home_parate'] = 0 if value is None else int(value)
    value = stats['Goalkeeper Saves']['away']
    match_dict['away_parate'] = 0 if value is None else int(value)
    value = stats['Fouls']['home']
    match_dict['home_falli'] = 0 if value is None else int(value)
    value = stats['Fouls']['away']
    match_dict['away_falli'] = 0 if value is None else int(value)
    value = stats['Red Cards']['home']
    match_dict['home_cartellini_rossi'] = 0 if value is None else int(value)
    value = stats['Red Cards']['away']
    match_dict['away_cartellini_rossi'] = 0 if value is None else int(value)
    value = stats['Yellow Cards']['home']
    match_dict['home_cartellini_gialli'] = 0 if value is None else int(value)
    value = stats['Yellow Cards']['away']
    match_dict['away_cartellini_gialli'] = 0 if value is None else int(value)
    value = stats['Total passes']['home']
    match_dict['home_passaggi_totali'] = 0 if value is None else int(value)
    value = stats['Total passes']['away']
    match_dict['away_passaggi_totali'] = 0 if value is None else int(value)
    value = stats['Passes accurate']['home']
    match_dict['home_passaggi_completati'] = 0 if value is None else int(value)
    value = stats['Passes accurate']['away']
    match_dict['away_passaggi_completati'] = 0 if value is None else int(value)
    return True


def save(df, conn, curs):
    ''' save match for future training '''
    df['home_final_score'] = np.nan
    df['away_final_score'] = np.nan
    df.to_sql('match', conn, if_exists='append')
    conn.commit()


def ended_matches(conn, curs):
    file_path = os.path.dirname(os.path.abspath(__file__))
    fixture_id_df = pd.read_sql_query("select fixture_id from match where home_final_score IS NULL;", conn)
    keys = json.load(open(f"{file_path}/../../keys.json"))
    headers = {
        'x-rapidapi-host': keys['x-rapidapi-host'],
        'x-rapidapi-key': keys['x-rapidapi-key']
        }
    # fixture id of finished matches
    fixture_ids = fixture_id_df.loc[:, 'fixture_id'].values.tolist()
    for fixture_id in fixture_ids:
        url = f"http://{keys['x-rapidapi-host']}/{keys['version']}fixtures/id/{fixture_id}"
        response = requests.request("GET", url, headers=headers)
        resp_dict = json.loads(response)
        status = resp_dict['api']['status']  # double check
        if status == 'Match Finished':
            values = (int(resp_dict['api']['goalsHomeTeam']), int(resp_dict['api']['goalsAwayTeam']), fixture_id)
            curs.execute("update match set home_final_score=?, away_final_score=? where fixture_id=?", values)
            conn.commit()


