import requests
import json


def get_fixture_ids():
    keys = json.load(open("../keys.js"))
    url = "https://api-football-v1.p.rapidapi.com/v2/fixtures/live"
    headers = {
        'x-rapidapi-host': keys['x-rapidapi-host'],
        'x-rapidapi-key': keys['x-rapidapi-key']
        }
    response = requests.request("GET", url, headers=headers)
    return fixtures


def get_match_statistics(fixture_id):
    keys = json.load(open("../keys.js"))
    url = f"https://api-football-v1.p.rapidapi.com/v2/statistics/fixture/{fixture_id}"
    headers = {
        'x-rapidapi-host': keys['x-rapidapi-host'],
        'x-rapidapi-key': keys['x-rapidapi-key']
        }
    response = requests.request("GET", url, headers=headers)
    return response


def from_json_to_df(response):
    pass

