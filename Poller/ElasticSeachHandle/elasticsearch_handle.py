from elasticsearch import Elasticsearch
import json
import elastic_transport
from .elastic_exceptions import *


class ElasticsearchHandel:
    def __init__(self, elasticsearch_url, username, password, fingerprint):
        self.elasticsearch_url = elasticsearch_url
        self.username = username
        self.password = password
        self.fingerprint = fingerprint
        self.client = Elasticsearch(
            hosts=self.elasticsearch_url,
            basic_auth=(self.username, self.password),
            ssl_assert_fingerprint=self.fingerprint,
        )

    def get_index(self, index_name, batch_size=100):
        setattr(self, index_name, [])
        index_list = getattr(self, index_name)
        from_index = 0
        all_instances = []

        while True:
            # query = {"query": {"match_all": {}}, "size": batch_size, "from": from_index}
            results = self.client.search(
                index=index_name,
                query={"match_all": {}},
                size=batch_size,
                from_=from_index,
            )
            instances = results["hits"]["hits"]

            all_instances.extend(instances)
            from_index += batch_size
            if len(instances) < 100:
                break

        setattr(self, index_name, [instance["_source"] for instance in all_instances])
        return getattr(self, index_name)

    def get_interactions(self, index_name, user_id, batch_size=100):
        # setattr(self, index_name, [])
        # index_list = getattr(self, index_name)
        from_index = 0
        all_instances = []

        query = {
            "match_phrase": {"userId": user_id},
        }

        results = self.client.search(
            index=index_name,
            query=query,
            size=batch_size,
            from_=from_index,
            timeout="1s",
        )
        # instances = results["hits"]["hits"][0]
        hits = results["hits"].get("hits")

        if not hits:
            # raise ValueError("User doesn't have any interactions.")
            raise InteractionNotFound()

        return hits[0].get("_source")

    def get_user_network_polls(self, user_id, batch_size=100):
        # setattr(self, index_name, [])
        # index_list = getattr(self, index_name)
        from_index = 0
        all_instances = []

        query = {
            "query": {
                "bool": {
                    "filter": {"term": {"userPrivatePolls.keyword": user_id}},
                    "must": {
                        "nested": {
                            "path": "userFolllowingIds",
                            "query": {
                                "terms": {
                                    "userFolllowingIds.keyword": [
                                        "list",
                                        "of",
                                        "following",
                                        "user",
                                        "ids",
                                    ]
                                }
                            },
                        }
                    },
                }
            }
        }

        results = self.client.search(
            index="users",
            query=query,
            size=batch_size,
            from_=from_index,
            timeout="1s",
        )
        # instances = results["hits"]["hits"][0]
        hits = results["hits"].get("hits")

        if not hits:
            # raise ValueError("User doesn't have any interactions.")
            raise InteractionNotFound()

        return hits[0].get("_source")

    def get_trend_polls(self, polls, ret_list=True):
        # polls = getattr(self, "polls")
        # trend_polls = sorted(polls, key=lambda x: (-x["numberOfPollups"], -x["numberOfVotes"], -x["numberOfLike"]))
        trend_polls = sorted(
            polls,
            key=lambda x: (
                -x["numberOfVotes"],
                -x["numberOfLike"],
                # -x["numberOfPollUp"],
            ),
        )
        return trend_polls

        # recs = trend_polls["id"]

        # print("\n", filtered_trend_polls, "\n")
        # setattr(self, "trend_polls", trend_polls)

    def export_index_to_file(self, index, index_file_path):
        try:
            with open(index_file_path, "w") as output:
                # for instance in self.instances:
                #        json.dump(instance["_source"], output, indent=4)
                json.dump(index, output, indent=4)
        except Exception as exp:
            print("Export Error", exp)


def get_index(self, index_name, batch_size=100):
    setattr(self, index_name, [])
    index_list = getattr(self, index_name)
    from_index = 0
    all_instances = []

    while True:
        # query = {"query": {"match_all": {}}, "size": batch_size, "from": from_index}
        results = self.client.search(
            index=index_name,
            query={"match_all": {}},
            size=batch_size,
            from_=from_index,
        )
        instances = results["hits"]["hits"]

        all_instances.extend(instances)
        from_index += batch_size
        if len(instances) < 100:
            break

    setattr(self, index_name, [instance["_source"] for instance in all_instances])
    return getattr(self, index_name)


""" if __name__ == "__main__":
    elasticsearch_url = "https://159.203.183.251:9200"
    username = "pollett"
    password = "9r0&rJP@19GY"
    fingerprint = "CE:AA:F7:FF:04:C7:31:14:78:9C:62:D4:CE:98:F9:EF:56:DA:70:45:37:14:E3:F8:66:0A:25:ED:05:04:83:EC"

    elastic_handle = ElasticsearchHandel(
        elasticsearch_url, username, password, fingerprint
    )

    polls = elastic_handle.get_index("polls")
    elastic_handle.export_index_to_file(polls, "./data/elas_polls.json")

    interactions = elastic_handle.get_index("userpollinteractions")
    elastic_handle.export_index_to_file(interactions, "./data/elas_interactions.json")
    # print(polls) """
