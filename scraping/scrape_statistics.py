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
import joblib

def get_match_statistics(day, columns, campionati, possesso_palla_index, discard_list):
    file_path = os.path.dirname(os.path.abspath(__file__))
    d = date.today().strftime("%d/%m/%Y")

    geckodriver_path = file_path + "/geckodriver"

    options = webdriver.FirefoxOptions()
    options.add_argument('-headless')
    fire = webdriver.FirefoxProfile()
    fire.set_preference("http.response.timeout", 3)
    fire.set_preference("dom.max_script_run_time", 3)
    driver = webdriver.Firefox(
        executable_path=geckodriver_path, firefox_profile=fire, options=options)
    url = "https://www.diretta.it"
    f = open(file_path + "/../csv/stats" + str(day) + ".csv", "a")
    try:
        driver.get(url)
        content_initial_page = driver.page_source
        soup_initial_page = BeautifulSoup(content_initial_page, "lxml")
        matches = soup_initial_page.find_all("div", class_="event__match event__match--live event__match--oneLine") + \
            soup_initial_page.find_all(
                "div", class_="event__match event__match--live event__match--last event__match--oneLine")
        
        for match in matches:
            teamA = match.find(
                "div", class_="event__participant event__participant--home").get_text().lower()
            teamB = match.find(
                "div", class_="event__participant event__participant--away").get_text().lower()

            if (teamA not in [item for sublist in campionati.values() for item in sublist]) and (teamB not in [item for sublist in campionati.values() for item in sublist]):
                continue

            if teamA in discard_list or teamB in discard_list:
                continue

            t = match.find(
                "div", class_="event__stage").get_text().lower().strip()
            t = t.replace("'", "")

            try:
                int(t)
            except ValueError:
                continue

            if int(t) in [0, 1, 2, 3, 4]:
                continue

            print("Accepted: {}-{}".format(teamA, teamB))
            href = match['id']
            f.write(d + "," + href[4:] + "," + t +
                    "," + teamA + "," + teamB + ",")

            found = False
            for key, value in campionati.items():
                if teamA in value or teamB in value:
                    f.write(key + ",")
                    found = True
                    break

            if not found:
                f.write("other,")
            
            documentNull = True
            trial = 0
            while documentNull:
                try:
                    newurl = url + "/partita/" + href[4:] + "/#statistiche-partite;0"
                    driver.get(newurl)
                    
                    j = 1
                    while driver.current_url != newurl and j != 5:
                        #print("old url...trying to reload on match: {}-{}, trial:{}".format(teamA, teamB, j))
                        driver.get(newurl)
                        time.sleep(1)
                        j += 1
                    
                    if j == 5:
                        print("didn't get the statistics of: {}-{}".format(teamA, teamB))

                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "statBox")))

                    content_statistic_page = driver.page_source
                    soup_statistic_page = BeautifulSoup(content_statistic_page, "lxml")
                    documentNull = False

                except WebDriverException:
                    print(traceback.format_exc())
                    print("On match : {}-{}".format(teamA, teamB))
                    trial += 1
                    if trial == 5:
                        with open("./discard", "a") as f3:
                            f3.write(teamA + "\n" + teamB + "\n")
                        raise
            

            scores = match.find(
                    "div", class_="event__scores fontBold").find_all("span")
            if scores != None:
                score_team_A = scores[0].get_text()
                score_team_B = scores[1].get_text()

            else:
                scores = soup_statistic_page.find("div", class_="match-info")
                score_team_A = scores.find_all(
                    "span", class_="scoreboard")[0].get_text()
                score_team_B = scores.find_all(
                    "span", class_="scoreboard")[1].get_text()

            f.write(score_team_A + "," + score_team_B)

            statistics = soup_statistic_page.find("div", class_="statBox").find(
                "div", class_="statContent").find_all("div", class_="statRow")

            current_index = possesso_palla_index
            for stat in statistics:
                home = stat.find(
                    "div", class_="statText statText--homeValue").get_text()
                name = stat.find("div", class_="statText statText--titleValue").get_text(
                ).lower().strip().replace(" ", "_").replace("'", "_")
                away = stat.find(
                    "div", class_="statText statText--awayValue").get_text()

                if name == "possesso_palla":
                    home = home[:-1]
                    away = away[:-1]

                while name not in columns[current_index]:

                    if "cartellini_rossi" in columns[current_index] or "cartellini_gialli" in columns[current_index]:
                        f.write(',0,0')
                        current_index += 2
                        continue
                    
                    f.write(",nan")
                    current_index += 1
            
                f.write("," + home + "," + away)
                current_index += 2
            
            odds = get_odds(url, driver, match)
            f.write(",{},{},{},{},{}".format(odds[0],odds[1],odds[2],odds[3],odds[4]))
            f.write("\n")

    except Exception:
        f.write("\n")
        print(traceback.format_exc())

    f.close()
    driver.close()




def get_ended_matches(day, campionati, columns):
    try:
        file_path = os.path.dirname(os.path.abspath(__file__))
        geckodriver_path = file_path + "/geckodriver"
        options = webdriver.FirefoxOptions()
        options.add_argument('-headless')
        fire = webdriver.FirefoxProfile()
        fire.set_preference("http.response.timeout", 3)
        fire.set_preference("dom.max_script_run_time", 3)
        driver = webdriver.Firefox(
            executable_path=geckodriver_path, firefox_profile=fire, options = options)
        url = "https://www.diretta.it"
        driver.get(url)

        locale.setlocale(locale.LC_ALL, 'it_IT.UTF-8')
        d = date.today() - timedelta(days = 1)
        d_name = calendar.day_name[d.weekday()][:3].capitalize()
        d = d.strftime("%d/%m")
        total_day = d + " " + d_name
        
        old_match = ""
        for index in range(2):

            if index == 1:
                driver.find_element_by_xpath('//*[@id="live-table"]/div[1]/div[2]/div[1]/div').click()
                trial = 0
                while driver.find_element_by_xpath('//*[@id="live-table"]/div[1]/div[2]/div[2]').text != total_day:
                    time.sleep(3)
                    print("not found yet...")
                    trial += 1
                    if trial == 5:
                        break
                wait = WebDriverWait(driver, 10)
                wait.until(EC.text_to_be_present_in_element((By.XPATH, '//*[@id="live-table"]/div[1]/div[2]/div[2]'),total_day))

            content = driver.page_source
            soup = BeautifulSoup(content, "lxml")

            matches = soup.find_all("div", class_="event__match event__match--oneLine") + \
                soup.find_all("div", class_="event__match event__match--last event__match--oneLine")

            
            #check that the program has changed date
            if old_match == "":
                old_match = matches
            elif old_match == matches:
                break
            
            for match in matches:

                stage = match.find("div", class_="event__stage").get_text().lower().strip()

                if stage == "finale":
                    teamA_bold = match.find(
                        "div", class_="event__participant event__participant--home fontBold")
                    teamA_notbold = match.find(
                        "div", class_="event__participant event__participant--home")
                    if teamA_bold is None:
                        teamA = teamA_notbold.get_text().lower().strip()
                    else:
                        teamA = teamA_bold.get_text().lower().strip()

                    teamB_bold = match.find(
                        "div", class_="event__participant event__participant--away fontBold")
                    teamB_notbold = match.find(
                        "div", class_="event__participant event__participant--away")
                    if teamB_bold is None:
                        teamB = teamB_notbold.get_text().lower().strip()
                    else:
                        teamB = teamB_bold.get_text().lower().strip()

                    if (teamA not in [item for sublist in campionati.values() for item in sublist]) and (teamB not in [item for sublist in campionati.values() for item in sublist]):
                        continue
                    
                    #print("{}-{}".format(teamA, teamB))
                    scores = match.find(
                        "div", class_="event__scores fontBold").find_all("span")
                    scoreA = scores[0].get_text()
                    scoreB = scores[1].get_text()
                    
                    with open(file_path + "/../csv/stats" + str(day) + ".csv", "r") as f:
                        lines = f.readlines()
                    
                    with open(file_path + "/../csv/stats" + str(day) + ".csv", "w") as f:
                        for line in lines:
                            if teamA in line and teamB in line:
                                print("writing final score of: {}-{}".format(teamA, teamB))
                                f.write(line.replace("\n", ",") +
                                        scoreA + "," + scoreB + "\n")
                            else:
                                f.write(line)
        driver.close()

    except Exception:
        print(traceback.format_exc())
        driver.close()



    

def filter_matches(day, columns):
    file_path = os.path.dirname(os.path.abspath(__file__))
    with open(file_path + "/../csv/stats" + str(day) + ".csv", "r") as f:
        lines = f.readlines()
    with open(file_path + "/../csv/stats" + str(day) + "final.csv", "w") as f:
        for line in lines:
            l = len([i for i in line.split(',') if i != ""])
            if l == (len(columns) - 2) or l == len(columns):
                f.write(line)


def scrape_teams(campionato):
    file_path = os.path.dirname(os.path.abspath(__file__))
    geckodriver_path = file_path + "/geckodriver"
    fire = webdriver.FirefoxProfile()
    fire.set_preference("http.response.timeout", 3)
    fire.set_preference("dom.max_script_run_time", 3)
    driver = webdriver.Firefox(
        executable_path=geckodriver_path, firefox_profile=fire)
    url = "https://www.diretta.it/" + campionato + "/squadre/"
    driver.get(url)
    content = driver.page_source
    soup = BeautifulSoup(content, "lxml")

    matches = soup.find_all("div", class_="leagueTable__teams")

    f = open("./teams.csv", "a")
    for match in matches:
        teamA = match.find(
            "a", class_="leagueTable__team").get_text().lower()
        f.write(teamA + ",")

    f.write("\n")
    f.close()
    driver.close()


def signal_handler(day, campionati, columns, signalNumber, frame):
    print("getting final scores....")
    get_ended_matches(day, campionati, columns)
    # in order to avoid spurious rows
    #filter_matches(day, columns)
    print("killing process...")
    os.kill(os.getpid(), signal.SIGKILL)



def get_odds(url, driver, match):
    try:
        href = match['id']
        newurl = url + "/partita/" + href[4:] + "/#comparazione-quote;quote-1x2;finale"
        j = 0
        while driver.current_url != newurl and j != 5:
            driver.get(newurl)
            time.sleep(1)
            j += 1
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "bookmaker")))
        content_odds_page = driver.page_source
        soup_odds_page = BeautifulSoup(content_odds_page, "lxml")
        odds = soup_odds_page.find_all("td", class_="kx")
        odds_res = [odds[i].get_text() for i in range(3)]
    except Exception:
        odds_res = [0,0,0]
        print("not captured odds result")
    try:
        newurl = url + "/partita/" + href[4:] + "/#comparazione-quote;over-under;finale"
        j = 0
        while driver.current_url != newurl and j != 5:
            driver.get(newurl)
            time.sleep(1)
            j += 1
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "bookmaker")))
        content_odds_page = driver.page_source
        soup_odds_page = BeautifulSoup(content_odds_page, "lxml")
        odds = soup_odds_page.find("table", id="odds_ou_2.5").find_all("td", class_="kx")
        odds_goals = [odds[i].get_text() for i in range(2)]

        return odds_res + odds_goals
    except Exception:
        print("not captured odds goals")
        return odds_res + [0,0]


# def adapt_files(day):
#     with open("/home/edoardo/Desktop/csv/stats1.csv", "r") as f:
#         header = f.readline()
#     with open("/home/edoardo/Desktop/csv/stats" + str(day) + ".csv", "r") as f:
#         lines = f.readlines()
#     with open("/home/edoardo/Desktop/csv/stats" + str(day) + ".csv", "w") as f:
#         f.write(header)
#         for line in lines[1:]:
#             elements = line.strip('\n').split(',')
#             for el in elements[:-2]:
#                 f.write(el + ',')
#             f.write('0,0,0,0,0,')
#             f.write(elements[-2] + ',' + elements[-1] + '\n')


# for i in range(2,32):
#     adapt_files(i)
#adapt_files(2)
# geckodriver_path = file_path + "/geckodriver"
# options = webdriver.FirefoxOptions()
# options.add_argument('-headless')
# fire = webdriver.FirefoxProfile()
# fire.set_preference("http.response.timeout", 3)
# fire.set_preference("dom.max_script_run_time", 3)
# driver = webdriver.Firefox(
#     executable_path=geckodriver_path, firefox_profile=fire, options = options)
# url = "https://www.diretta.it"
# get_odds(url, driver, "UDFgrSAP")

#scrape_teams("calcio/europa/europei")
# crontab -e */5 15-22 * 1-5,9-12 6,7 /home/script.sh

# 24 17 30 * * ./run_scrape.sh

# 0 19 * 1-5,9-12 1-5
# 0 19 * 1-5,9-12 1-5
# 0 14 * 1-5,9-12 6,7
# 0 14 * 1-5,9-12 6,7
# test


# file_path = os.path.dirname(os.path.abspath(__file__))
# with open(file_path + "/../csv/stats1.csv", "r") as f:
#     line = f.readline()
#     columns = [col.replace("\n", "").replace(" ", "")
#             for col in line.split(',')]
# campionati = {}
# with open("./teams.csv", "r") as f:
#     champs = f.readlines()
#     for champ in champs:
#         line = champ.split(",")
#         team_list = [team.replace(
#             "\n", "") for team in line[1:] if team.replace("\n", "") != ""]
#         campionati[line[0].replace(" ", "")] = team_list
    
# get_ended_matches(35, campionati, columns)
#filter_matches(25, columns)


        # geckodriver_path = file_path + "/geckodriver"
        # options = webdriver.FirefoxOptions()
        # options.add_argument('-headless')
        # fire = webdriver.FirefoxProfile()
        # fire.set_preference("http.response.timeout", 3)
        # fire.set_preference("dom.max_script_run_time", 3)
        # driver = webdriver.Firefox(
        #     executable_path=geckodriver_path, firefox_profile=fire)
        # url = "https://www.diretta.it"
        # driver.get(url)

        # locale.setlocale(locale.LC_ALL, 'it_IT.UTF-8')
        # d = date.today() - timedelta(days = 1)
        # d_name = calendar.day_name[d.weekday()][:3].capitalize()
        # d = d.strftime("%d/%m")
        # total_day = d + " " + d_name
        
        
        # #the day before
        # #if int(time.ctime()[11:13]) < 14:
        # for index in range(2):

        #     if index == 1:
        #         driver.find_element_by_xpath('//*[@id="live-table"]/div[1]/div/div[1]/div').click()
        #         wait = WebDriverWait(driver, 10)
        #         wait.until(EC.text_to_be_present_in_element((By.XPATH, '//*[@id="live-table"]/div[1]/div/div[2]'),total_day))
        #         time.sleep(2)

        #     content = driver.page_source
        #     soup = BeautifulSoup(content, "lxml")

        #     matches = soup.find_all("div", class_="event__match event__match--oneLine") + \
        #         soup.find_all("div", class_="event__match event__match--last event__match--oneLine")

        #     print(matches[0])
        
        # driver.close()

