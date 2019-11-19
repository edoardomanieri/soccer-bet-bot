from selenium import webdriver
from bs4 import BeautifulSoup
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import os


def setup():
    file_path = os.path.dirname(os.path.abspath(__file__))
    geckodriver_path = file_path + "/geckodriver"
    options = webdriver.FirefoxOptions()
    options.add_argument('-headless')
    fire = webdriver.FirefoxProfile()
    fire.set_preference("http.response.timeout", 3)
    fire.set_preference("dom.max_script_run_time", 3)
    driver = webdriver.Firefox(
        executable_path=geckodriver_path, firefox_profile=fire, options=options)
    return driver


def get_matches_dict():
    driver = setup()
    url = "https://www.eurobet.it/it/scommesse-live/#!"
    driver.get(url)
    time.sleep(2)
    content = driver.page_source
    soup_initial_page = BeautifulSoup(content, "lxml")
    matches = soup_initial_page.find_all("div", class_="event-players")
    match_dict = dict()
    for match in matches:
        match_a = match.find("a")
        href = match_a['href']
        teams = match_a.get_text().lower()
        teamA, teamB = teams.split('-')
        teamA = teamA.strip()
        teamB = teamB.strip()
        match_dict[(teamA, teamB)] = href
    driver.close()
    return match_dict


def get_live_odds(driver, d, teamA, teamB):
    href = get_match_href(d, teamA, teamB)
    if href == 0:
        print("not captured odds, not in dict")
        return [0, 0, 0, 0, 0]
    try:
        bet_types = ['u/o 2.5', '1X2']
        newurl = 'https://www.eurobet.it' + href
        driver.get(newurl)
        while driver.current_url != newurl:
            driver.get(newurl)
            time.sleep(2)
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CLASS_NAME, "quotaType")))
        match_content = driver.page_source
        match_page = BeautifulSoup(match_content, "lxml")
        boxes = match_page.find_all("div", class_="box-container")
        odds_res = []
        for box in boxes:
            bet = box.find('h2', class_='title-event info-match').find("div").get_text()
            if bet in bet_types:
                odds = box.find_all("div", class_='containerQuota')
                for odd in odds:
                    # quotatype = odd.find('p', class_='quotaType').get_text()
                    quota = odd.find("div", class_='quota').get_text().strip()
                    quota = float(quota)
                    odds_res.append(quota)
                    if len(odds_res) == 5:
                        # prima over poi under
                        tmp = odds_res[3]
                        odds_res[3] = odds_res[4]
                        odds_res[4] = tmp
                        return odds_res
    except Exception:
        odds_res = [-1, -1, -1, -1, -1]
        print(traceback.format_exc())
        print("not captured odds")
        return odds_res


def get_match_href(d, teamA, teamB):
    for teamA_d, teamB_d in d.keys():
        if teamA_d in teamA or teamA in teamA_d:
            return d[(teamA_d, teamB_d)]
        if teamB_d in teamB or teamB in teamB_d:
            return d[(teamA_d, teamB_d)]
    return 0


def test():
    d = get_matches_dict()
    driver = setup()
    odds = get_live_odds(driver, d, 'sarmiento', 'crucero del norte')
    print(odds)
    driver.close()

# test()