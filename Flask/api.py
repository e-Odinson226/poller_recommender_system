import requests as re
import json

root = "http://172.19.0.1:8000/"
payload = {
    "user_ID": "1",
    "interactions": [
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

headers = {"accept": "application/json"}
headers = {"Content-Type: application/json"}

response = re.post(root + "get_rec/", json=payload)
# response = re.post(root + "get_rec/", json=payload)
print(response.json())
