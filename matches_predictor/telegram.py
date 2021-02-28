import requests
import json
import os
import argparse

def telegram_bot_sendtext(bot_message):

    file_path = os.path.dirname(os.path.abspath(__file__))
    with open(f"{file_path}/../keys.json") as keys_file:
        keys = json.load(keys_file)
    bot_token = keys['telegram-bot-key']
    bot_chatID = keys['telegram-bot-id']
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)

    return response.json()


def create_keys(bot_token, bot_chatID, apifootball_key):
    file_path = os.path.dirname(os.path.abspath(__file__))
    d_keys = {'x-rapidapi-key': apifootball_key, 'telegram-bot-key': bot_token, 'telegram-bot-id': bot_chatID}
    with open(f"{file_path}/../keys.json", 'w') as keys_file:
        json.dump(d_keys, keys_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bot_token", help="bot key used to interact with the bot")
    parser.add_argument("bot_chatID", help="bot chat ID")
    parser.add_argument("apifootball_key", help="your secret football api key")
    args = parser.parse_args()
    create_keys(args.bot_token, args.bot_chatID, args.apifootball_key)



if __name__ == '__main__':
    main()
