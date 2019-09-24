#!/usr/bin/python3

from selenium import webdriver
from bs4 import BeautifulSoup
import time
import io
import numpy as np
import pandas as pd
import csv
from fake_useragent import UserAgent
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import os
import glob
from datetime import date, timedelta
import traceback
import signal
from functools import partial
import calendar
import locale
import model
import scrape_statistics as ss
import sys


if __name__ == "__main__":

    predictions = False

    files = glob.glob("../csv/stats*")
    day = len(files) + 1
    
    with open("../csv/stats1.csv", "r") as f:
        line = f.readline()
        columns = [col.replace("\n", "").replace(" ", "")
                   for col in line.split(',')]
    with open("../csv/stats" + str(day) + ".csv", "w+") as f:
        f.write(line)

    campionati = {}
    with open("./teams.csv", "r") as f:
        champs = f.readlines()
        for champ in champs:
            line = champ.split(",")
            team_list = [team.replace(
                "\n", "") for team in line[1:] if team.replace("\n", "") != ""]
            campionati[line[0].replace(" ", "")] = team_list

    i = 0
    while "possesso_palla" not in columns[i]:
        i += 1

    print(os.getpid())
    
    with open("./kill_process.sh", "w") as f:
        f.write("#!/bin/bash\nkill -15 {}".format(os.getpid()))

    signal.signal(signal.SIGTERM, partial( 
        ss.signal_handler, day, campionati, columns))

    with open("./discard", "w") as f:
        f.write("")
    previous_len = 0
    while True:
        try:
            with open("./discard", "r") as f1:
                discard_list = [line.replace("\n", "").strip() for line in f1.readlines()]
            if predictions:
                while i < 5:
                    print("getting matches statistics, don't stop the process...")
                    ss.get_match_statistics_for_predictions(day, columns, campionati, i, discard_list)
                    previous_len += ss.get_input_stream(previous_len)
                    print("Now you can safely stop the process...")
                    time.sleep(60)
                print("getting matches statistics, don't stop the process...")
                ss.get_match_statistics(day, columns, campionati, i, discard_list)
                print("Now you can safely stop the process...")
            else:
                print("getting matches statistics, don't stop the process...")
                ss.get_match_statistics(day, columns, campionati, i, discard_list)
                #print(model.real_time_output(retrain_model = True))
                print("Now you can safely stop the process...")
                
                time.sleep(300)
        except KeyboardInterrupt:
            print("getting final scores....")
            ss.get_ended_matches(day, campionati, columns)
            # in order to avoid spurious rows
            ss.filter_matches(day, columns)
            print("killing process...")
            break