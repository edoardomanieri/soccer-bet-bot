import requests
import json
import pandas as pd
import numpy as np
import time
import os

'''date', 'id_partita', 'minute', 'home', 'away', 'campionato',
       'home_score', 'away_score', 'home_possesso_palla',
       'away_possesso_palla', 'home_tiri', 'away_tiri', 'home_tiri_in_porta',
       'away_tiri_in_porta', 'home_tiri_fuori', 'away_tiri_fuori',
       'home_tiri_fermati', 'away_tiri_fermati', 'home_punizioni',
       'away_punizioni', 'home_calci_d_angolo', 'away_calci_d_angolo',
       'home_fuorigioco', 'away_fuorigioco', 'home_rimesse_laterali',
       'away_rimesse_laterali', 'home_parate', 'away_parate', 'home_falli',
       'away_falli', 'home_cartellini_rossi', 'away_cartellini_rossi',
       'home_cartellini_gialli', 'away_cartellini_gialli',
       'home_passaggi_totali', 'away_passaggi_totali',
       'home_passaggi_completati', 'away_passaggi_completati',
       'home_contrasti', 'away_contrasti', 'home_attacchi', 'away_attacchi',
       'home_attacchi_pericolosi', 'away_attacchi_pericolosi', 'odd_1',
       'odd_X', 'odd_2', 'odd_over', 'odd_under', 'live_odd_1', 'live_odd_X',
       'live_odd_2', 'live_odd_over', 'live_odd_under', 'home_final_score',
       'away_final_score'''


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
    keys = json.load(open("../keys.js"))
    url = "https://api-football-v1.p.rapidapi.com/v2/fixtures/live"
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
        out_dict['campionato'] = resp['league']['name']
        out_dict['date'] = resp['event_date']
        out_dict['minute'] = resp['elapsed']
        out_dict['home'] = resp['homeTeam']['team_name']
        out_dict['away'] = resp['awayTeam']['team_name']
        out_dict['home_score'] = resp['goalsHomeTeam']
        out_dict['away_score'] = resp['goalsAwayTeam']
        out_dict['id_partita'] = f"{resp['event_date']}-{resp['fixture_id']}"
        res.append(out_dict)
    return res


def get_label_dict():
    keys = json.load(open("../keys.js"))
    url = f"https://api-football-v1.p.rapidapi.com/v2/odds/labels/"
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
    keys = json.load(open("../keys.js"))
    url = f"https://api-football-v1.p.rapidapi.com/v2/odds/fixture/{fixture_id}/label/{label_id}"
    headers = {
        'x-rapidapi-host': keys['x-rapidapi-host'],
        'x-rapidapi-key': keys['x-rapidapi-key']
        }
    response = requests.request("GET", url, headers=headers)
    return response.content


def prematch_odds_uo_to_dict(response, match_dict):
    resp_dict = json.loads(response)
    bets = resp_dict['api']['odds'][0]['bookmakers'][0]['bets'][0]['values']
    for bet in bets:
        if bet['value'] == 'Over 2.5':
            match_dict['odd_over'] = float(bet['odd'])
        if bet['value'] == 'Under 2.5':
            match_dict['odd_under'] = float(bet['odd'])
    if 'odd_over' not in match_dict.keys():
        print("Over 2.5 odd not found \n")
        match_dict['odd_over'] = np.nan
    if 'odd_under' not in match_dict.keys():
        print("Under 2.5 odd not found \n")
        match_dict['odd_under'] = np.nan


def prematch_odds_1x2_to_dict(response, match_dict):
    resp_dict = json.loads(response)
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
        match_dict['odd_1'] = np.nan
    if 'odd_X' not in match_dict.keys():
        print("X odd not found \n")
        match_dict['odd_X'] = np.nan
    if 'odd_2' not in match_dict.keys():
        print("2 odd not found \n")
        match_dict['odd_2'] = np.nan


def get_match_statistics(fixture_id):
    keys = json.load(open("../keys.js"))
    url = f"https://api-football-v1.p.rapidapi.com/v2/statistics/fixture/{fixture_id}"
    headers = {
        'x-rapidapi-host': keys['x-rapidapi-host'],
        'x-rapidapi-key': keys['x-rapidapi-key']
        }
    response = requests.request("GET", url, headers=headers)
    return response.content


def stat_to_dict(response, match_dict):
    resp_dict = json.loads(response)
    match_dict['home_tiri_in_porta'] = int(resp_dict['api']['statistics']['Shots on Goal']['home'])
    match_dict['away_tiri_in_porta'] = int(resp_dict['api']['statistics']['Shots on Goal']['away'])
    match_dict['home_tiri_fuori'] = int(resp_dict['api']['statistics']['Shots off Goal']['home'])
    match_dict['away_tiri_fuori'] = int(resp_dict['api']['statistics']['Shots off Goal']['away'])
    match_dict['home_tiri'] = int(resp_dict['api']['statistics']['Total Shots']['home'])
    match_dict['away_tiri'] = int(resp_dict['api']['statistics']['Total Shots']['away'])
    match_dict['home_tiri_fermati'] = int(resp_dict['api']['statistics']['Blocked Shots']['home'])
    match_dict['away_tiri_fermati'] = int(resp_dict['api']['statistics']['Blocked Shots']['away'])
    match_dict['home_falli'] = int(resp_dict['api']['statistics']['Fouls']['home'])
    match_dict['away_falli'] = int(resp_dict['api']['statistics']['Fouls']['away'])
    match_dict['home_calci_d_angoli'] = int(resp_dict['api']['statistics']['Corner Kicks']['home'])
    match_dict['away_calci_d_angoli'] = int(resp_dict['api']['statistics']['Corner Kicks']['away'])
    match_dict['home_fuorigioco'] = int(resp_dict['api']['statistics']['Offsides']['home'])
    match_dict['away_fuorigioco'] = int(resp_dict['api']['statistics']['Offsides']['away'])
    match_dict['home_possesso_palla'] = int(resp_dict['api']['statistics']['Ball Possession']['home'].replace("%", ""))
    match_dict['away_possesso_palla'] = int(resp_dict['api']['statistics']['Ball Possession']['away'].replace("%", ""))
    match_dict['home_cartellini_gialli'] = int(resp_dict['api']['statistics']['Yellow Cards']['home'])
    match_dict['away_cartellini_gialli'] = int(resp_dict['api']['statistics']['Yellow Cards']['away'])
    match_dict['home_cartellini_rossi'] = len(resp_dict['api']['statistics']['Red Cards']['home'])
    match_dict['away_cartellini_rossi'] = len(resp_dict['api']['statistics']['Red Cards']['away'])
    match_dict['home_parate'] = int(resp_dict['api']['statistics']['Goalkeeper Saves']['home'])
    match_dict['away_parate'] = int(resp_dict['api']['statistics']['Goalkeeper Saves']['away'])
    match_dict['home_passaggi_totali'] = int(resp_dict['api']['statistics']['Total passes']['home'])
    match_dict['away_passaggi_totali'] = int(resp_dict['api']['statistics']['Total passes']['away'])
    match_dict['home_passaggi_completati'] = int(resp_dict['api']['statistics']['Passes accurate']['home'])
    match_dict['away_passaggi_completati'] = int(resp_dict['api']['statistics']['Passes accurate']['away'])


# save match on df for future training
def save(df):
    file_path = os.path.dirname(os.path.abspath(__file__))
    fixture_id = df.loc[:, 'fixture_id'][0]
    with open(f"{file_path}/../res/fixture_ids", "r") as f:
        fixture_ids = [line.replace("\n", "").strip() for line in f.readlines()]
    if fixture_id not in fixture_ids:
        with open(f"{file_path}/../res/fixture_ids", "a") as f:
            f.write(f"{fixture_id}\n")
    df_before = pd.read_csv(f'{file_path}/../res/temp.csv', index_col=0)
    df['home_final_score'] = np.nan
    df['away_final_score'] = np.nan
    df_new = pd.concat([df_before, df])
    df_new.to_csv(f'{file_path}/../res/temp.csv')


def ended_matches():
    file_path = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(f'{file_path}/../res/temp.csv', index_col=0)
    with open(f"{file_path}/../res/fixture_ids", "r") as f:
        fixture_ids = [line.replace("\n", "").strip() for line in f.readlines()]
    keys = json.load(open("../keys.js"))
    headers = {
        'x-rapidapi-host': keys['x-rapidapi-host'],
        'x-rapidapi-key': keys['x-rapidapi-key']
        }
    # fixture id of finished matches
    finished = []
    for fixture_id in fixture_ids:
        url = f"https://api-football-v1.p.rapidapi.com/v2/fixtures/id/{fixture_id}"
        response = requests.request("GET", url, headers=headers)
        resp_dict = json.loads(response)
        status = resp_dict['api']['status'] # double check
        if status == 'Match Finished':
            df.loc[df['fixture_id'] == fixture_id, 'home_final_score'] = int(resp_dict['api']['goalsHomeTeam'])
            df.loc[df['fixture_id'] == fixture_id, 'away_final_score'] = int(resp_dict['api']['goalsAwayTeam'])
            # remove fixture id from list
            finished.append(fixture_id)
    # write not finished matches on the fixture id file and temp file
    not_finished = [f for f in fixture_id if f not in fixture_ids]
    with open(f"{file_path}/../res/fixture_ids", "w") as f:
        for f_id in not_finished:
            f.write(f"{f_id}\n")
    df_temp_remaining = df.loc[df['fixture_id'].isin(not_finished), :]
    df_temp_remaining.to_csv(f'{file_path}/../res/temp.csv')
    # save file in res folder total df file
    df_to_save = df.loc[~df['fixture_id'].isin(not_finished), :]
    df_before = pd.read_csv(f"{file_path}/../res/df_api.csv", index_col=0)
    df = pd.concat([df_before, df_to_save], axis=0, ignore_index=True)
    df.to_csv(f"{file_path}/../res/df_api.csv")


def live_matches_producer(out_q, minute_threshold):
    n_api_call = 0
    label_dict = get_label_dict()
    n_api_call += 1
    while True:
        matches_list = get_basic_info()
        n_api_call += 1
        for match in matches_list:
            if match['minute'] < minute_threshold:
                continue
            resp = get_match_statistics(match['fixture_id'])
            stat_to_dict(resp, match)
            resp = get_prematch_odds(match['fixture_id'], label_dict['Goals Over/Under'])
            prematch_odds_uo_to_dict(resp, match)
            resp = get_prematch_odds(match['fixture_id'], label_dict['Match Winner'])
            n_api_call += 3
            prematch_odds_1x2_to_dict(resp, match)
            df = pd.DataFrame(data=match)
            out_q.put(df)
            save(df)
        time.sleep(301)
