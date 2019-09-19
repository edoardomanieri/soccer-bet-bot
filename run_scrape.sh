#!/bin/bash
PATH=/home/edoardo/google-cloud-sdk/bin:/home/edoardo/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
cd "$(dirname "$0")";
CWD="$(pwd)"
echo $CWD
/usr/bin/python3 /home/edoardo/Desktop/Betting_live/scrape_statistics.py
