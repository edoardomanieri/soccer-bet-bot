import sqlite3
import os

file_path = os.path.dirname(os.path.abspath(__file__))
connection = sqlite3.connect(f"{file_path}/../../res/football.db")
cursor = connection.cursor()


sql_command = """
CREATE TABLE match (
id_partita TEXT PRIMARY KEY,
fixture_id INTEGER,
date TEXT,
minute INTEGER,
home TEXT,
away TEXT,
campionato TEXT,
home_score INTEGER,
away_score INTEGER,
home_possesso_palla INTEGER,
away_possesso_palla INTEGER,
home_tiri INTEGER,
away_tiri INTEGER,
home_tiri_in_porta INTEGER,
away_tiri_in_porta INTEGER,
home_tiri_fuori INTEGER,
away_tiri_fuori INTEGER,
home_tiri_fermati INTEGER,
away_tiri_fermati INTEGER,
home_calci_d_angolo INTEGER,
away_calci_d_angolo INTEGER,
home_fuorigioco INTEGER,
away_fuorigioco INTEGER,
home_parate INTEGER,
away_parate INTEGER,
home_falli INTEGER,
away_falli INTEGER,
home_cartellini_rossi INTEGER,
away_cartellini_rossi INTEGER,
home_cartellini_gialli INTEGER,
away_cartellini_gialli INTEGER,
home_passaggi_totali INTEGER,
away_passaggi_totali INTEGER,
home_passaggi_completati INTEGER,
away_passaggi_completati INTEGER,
odd_1 REAL,
odd_X REAL,
odd_2 REAL,
odd_over REAL,
odd_under REAL,
home_final_score INTEGER,
away_final_score INTEGER);
"""

cursor.execute(sql_command)
connection.commit()
connection.close()