from elasticsearch import Elasticsearch
import csv
import json

client = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("zakaria", "ZaKaRiA1234"),
    verify_certs=False
    # api_key=(
    #    "get_polls",
    #    "U1NObEpvb0IxN2R6ZC1mekhyRno6VmEtc01RanZSXzJTVkdZLUFCSFBZQQ==",
    # ),
    # API key ID and secret
)
# resp = client.get(index="polls", id="eSNVJooB17dzd-fzTbAt", human=True)
resp = client.search(
    index="polls",
    query={"match_all": {}},
    size=210,
)

# print(type(resp["hits"]))

with open("./data/elas_polls.json", "w") as output:
    json.dump(resp["hits"]["hits"], output, indent=4)


# print("Got %d Hits:" % resp["hits"]["total"]["value"])
# for hit in resp["hits"]["hits"]:
#    print(f"{hit['_source']}")
