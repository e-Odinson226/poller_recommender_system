import requests as re
import json

# root = "http://127.0.0.1:5000/"
root = "http://172.19.0.1:8000/"
payload = {
    "user_ID": "1",
    "interaction": [
        {"poll_ID": 8, "aciton": "vote | comment"},
        {"poll_ID": 4, "aciton": "vote | comment"},
        {"poll_ID": 2, "aciton": "vote | share"},
        {"poll_ID": 6, "aciton": "vote"},
        {"poll_ID": 16, "aciton": "vote | comment"},
        {"poll_ID": 17, "aciton": "vote"},
        {"poll_ID": 18, "aciton": "vote"},
        {"poll_ID": 31, "aciton": "vote"},
        {"poll_ID": 33, "aciton": "vote"},
        {"poll_ID": 34, "aciton": "vote | share"},
    ],
}


payload = json.dumps(payload)

print(payload)
headers = {"accept": "application/json"}

response = re.post(root + "get_rec/2", json=payload)
print(response.json())
