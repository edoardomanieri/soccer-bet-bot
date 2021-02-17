import requests


def telegram_bot_sendtext(bot_message):
    
    bot_token = '1650840555:AAFZcULZIPn8q9eWnYRut7K6Zsrh_OWuZ10'
    bot_chatID = '249241585'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)

    return response.json()
