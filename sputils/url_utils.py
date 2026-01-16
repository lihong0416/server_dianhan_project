import requests
import json
import traceback


def send_json_data(url, json_data, mode='post'):
    try:
        headers = {'Content-Type': 'application/json'}
        if mode.upper() == 'POST':
            r = requests.post(url=url, headers=headers, data=json.dumps(json_data))
        elif mode.upper() == 'GET':
            r = requests.get(url, params=json_data)
        return r
    except:
        raise traceback.format_exc()
