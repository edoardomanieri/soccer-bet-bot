from matches_predictor import API_connect
import pandas as pd

EXPECTED_COLS = ['date', 'id_partita', 'minute', 'home', 'away', 'campionato',
       'home_score', 'away_score', 'home_possesso_palla',
       'away_possesso_palla', 'home_tiri', 'away_tiri', 'home_tiri_in_porta',
       'away_tiri_in_porta', 'home_tiri_fuori', 'away_tiri_fuori',
       'home_tiri_fermati', 'away_tiri_fermati', 'home_calci_d_angolo', 'away_calci_d_angolo',
       'home_fuorigioco', 'away_fuorigioco', 'home_parate', 'away_parate', 'home_falli',
       'away_falli', 'home_cartellini_rossi', 'away_cartellini_rossi',
       'home_cartellini_gialli', 'away_cartellini_gialli',
       'home_passaggi_totali', 'away_passaggi_totali',
       'home_passaggi_completati', 'away_passaggi_completati', 'odd_1',
       'odd_X', 'odd_2', 'odd_over', 'odd_under']


def check_df_correctness(df):
    assert len(df.columns) == len(EXPECTED_COLS)
    for col in df.columns:
        assert col in EXPECTED_COLS


def test_api_connect():
    label_dict = API_connect.get_label_dict()
    matches_list = API_connect.get_basic_info()
    match = matches_list[0]
    resp = API_connect.get_match_statistics(match['fixture_id'])
    API_connect.stat_to_dict(resp, match)
    resp = API_connect.get_prematch_odds(match['fixture_id'], label_dict['Goals Over/Under'])
    API_connect.prematch_odds_uo_to_dict(resp, match)
    resp = API_connect.get_prematch_odds(match['fixture_id'], label_dict['Match Winner'])
    API_connect.prematch_odds_1x2_to_dict(resp, match)
    df = pd.DataFrame([match])
    check_df_correctness(df)
