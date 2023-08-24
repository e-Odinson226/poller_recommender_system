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


# Specify the name of the Elasticsearch index you want to retrieve data from
index_name = "polls"

# Initialize variables
batch_size = 100  # Adjust this based on your needs
from_index = 0
all_instances = []

# Use a while loop to paginate through the results
while True:
    # Define a match-all query and set the size and from parameters
    query = {"query": {"match_all": {}}, "size": batch_size, "from": from_index}

    # Use the search method to retrieve a batch of instances from the index
    results = client.search(index=index_name, body=query)

    # Extract the instances from the search results
    instances = results["hits"]["hits"]

    # If there are no more instances, break the loop
    if not instances:
        break

    # Append the batch of instances to the list of all instances
    all_instances.extend(instances)

    # Increment the from_index to retrieve the next batch
    from_index += batch_size

# Print or process all_instances as needed
for instance in all_instances:
    print(instance["_source"])
print(all_instances)

with open("./data/elas_polls.json", "w") as output:
    for instance in all_instances:
        json.dump(instance["_source"], output, indent=4)

    # json.dump(resp["hits"]["hits"], output, indent=4)


# print("Got %d Hits:" % resp["hits"]["total"]["value"])
# for hit in resp["hits"]["hits"]:
#    print(f"{hit['_source']}")
