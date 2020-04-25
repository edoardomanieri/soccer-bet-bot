import pandas as pd
import glob
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


def get_df(res_path):
    file_path = os.path.dirname(os.path.abspath(__file__))
    # import dataset
    all_files = sorted(glob.glob(f"{file_path}/{res_path}/*.csv"),
                       key=lambda x: int(x[x.index('stats') + 5:-4]))
    li = [pd.read_csv(filename, index_col=None, header=0)
          for filename in all_files[:-1]]
    df = pd.concat(li, axis=0, ignore_index=True)
    return df.reset_index(drop=True)
