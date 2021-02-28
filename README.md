# Soccer bet bot

A multithreaded application able to retrieve live soccer data using API, run a machine learning model, provide live outcome predictions on football matches and give notification using Telegram bot.

## Usage

- Create your own bot following the first part of this guide [create-your-telegram-bot](https://medium.com/@ManHay_Hong/how-to-create-a-telegram-bot-and-send-messages-with-python-4cf314d9fa3e) and obtain your own bot token and chat ID.

- Register on [https://www.api-football.com/](https://www.api-football.com/) with the free subscription and obtain your api key.

- Download the source code.

- Install the requirements
    > pip install -r requirements.txt

- Install the package
    > pip install .

- Execute the following command to generate your secret keys.json file:
    > python telegram.py \<your-bot-token> \<your-chat-ID> \<your-apifootball-key>

- Run the app:
    > python betbot.py




