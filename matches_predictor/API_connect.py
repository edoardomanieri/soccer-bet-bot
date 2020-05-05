import requests
import json
import pandas as pd
import time

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
    list_of_matches = basic_info_to_dict(response)
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


def get_match_statistics(fixture_id):
    keys = json.load(open("../keys.js"))
    url = f"https://api-football-v1.p.rapidapi.com/v2/statistics/fixture/{fixture_id}"
    headers = {
        'x-rapidapi-host': keys['x-rapidapi-host'],
        'x-rapidapi-key': keys['x-rapidapi-key']
        }
    response = requests.request("GET", url, headers=headers)
    return response

''' missing:
home_punizioni', 'away_punizioni', 'home_rimesse_laterali','away_rimesse_laterali'
'home_contrasti', 'away_contrasti', 'home_attacchi', 'away_attacchi',
'home_attacchi_pericolosi', 'away_attacchi_pericolosi'''
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


def live_matches_producer(out_q):
    n_api_call = 0
    while True:
        matches_list = get_basic_info()
        n_api_call += 1
        for match in matches_list:
            resp = get_match_statistics(match['fixture_id'])
            n_api_call += 1
            stat_to_dict(resp, match)
            df = pd.DataFrame(data=match)
            out_q.put(df)
        time.sleep(301)
