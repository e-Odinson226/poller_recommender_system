from elasticsearch import Elasticsearch
import json

#   "ElasticSearch" : {
#       "Url": "https://159.203.183.251:9200",
#       "UserName": "pollett",
#       "Password": "9r0&rJP@19GY",
#       "FingerPrint": "CE:AA:F7:FF:04:C7:31:14:78:9C:62:D4:CE:98:F9:EF:56:DA:70:45:37:14:E3:F8:66:0A:25:ED:05:04:83:EC"
#     }


class ElasticsearchDataExporter:
    def __init__(self, elasticsearch_url, username, password, fingerprint):
        self.elasticsearch_url = elasticsearch_url
        self.username = username
        self.password = password
        self.fingerprint = fingerprint
        self.client = Elasticsearch(
            self.elasticsearch_url,
            basic_auth=(self.username, self.password),
            ssl_assert_fingerprint=self.fingerprint,
        )
        self.instances = []

    def export_index(self, index_name, batch_size=100):
        from_index = 0
        all_instances = []

        while True:
            query = {"query": {"match_all": {}}, "size": batch_size, "from": from_index}
            results = self.client.search(index=index_name, body=query)
            instances = results["hits"]["hits"]

            if not instances:
                break

            all_instances.extend(instances)
            from_index += batch_size
        # self.instances = all_instances
        self.instances = [instance["_source"] for instance in all_instances]
        # self.instances = json.dumps(self.instances)
        return self.instances

    def export_index_to_file(self, file_path="./data/elas_polls.json"):
        try:
            with open(file_path, "w") as output:
                # for instance in self.instances:
                #        json.dump(instance["_source"], output, indent=4)
                json.dump(self.instances, output, indent=4)
        except Exception as exp:
            print("Export Error", exp)


if __name__ == "__main__":
    elasticsearch_url = "https://159.203.183.251:9200"
    username = "pollett"
    password = "9r0&rJP@19GY"
    fingerprint = "CE:AA:F7:FF:04:C7:31:14:78:9C:62:D4:CE:98:F9:EF:56:DA:70:45:37:14:E3:F8:66:0A:25:ED:05:04:83:EC"

    exporter = ElasticsearchDataExporter(
        elasticsearch_url, username, password, fingerprint
    )
    # index_name = "userpollinteractions"
    polls = exporter.export_index("polls")
    exporter.export_index_to_file()
    # print(polls)
