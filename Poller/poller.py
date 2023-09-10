from flask import Flask, redirect, url_for, jsonify, request
from flask_restful import Api, Resource
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

from .RecommenderSystem.recommender_system import *
from .ElasticSeachHandle.elasticsearch_handle import *


app = Flask(__name__)
print(app)
api = Api(app)


class Rec(Resource):
    def __init__(self):
        pd.set_option("display.max_columns", None)

        elasticsearch_url = os.environ.get("POLLER_ELASTICSEARCH_URL")
        username = os.environ.get("POLLER_USERNAME")
        password = os.environ.get("POLLER_PASSWORD")
        fingerprint = os.environ.get("POLLER_FINGERPRINT")

        # elasticsearch_url = "https://159.203.183.251:9200"
        # username = "pollett"
        # password = "9r0&rJP@19GY"
        # fingerprint = "CE:AA:F7:FF:04:C7:31:14:78:9C:62:D4:CE:98:F9:EF:56:DA:70:45:37:14:E3:F8:66:0A:25:ED:05:04:83:ec"

        # self.polls = get_polls_list("/data/polls_synthetic.csv")
        self.elastic_handle = ElasticsearchHandel(
            elasticsearch_url, username, password, fingerprint
        )

    def post(self):
        try:
            args = request.get_json(force=True)
            user_id = args.get("userId")

            self.polls = self.elastic_handle.get_index("polls")
            self.polls = pd.DataFrame.from_records(self.polls)
            # self.polls = encode_topics(self.polls)

            self.polls_tf_idf_matrix = create_tf_idf_matrix(self.polls, "topics")

            self.cosine_similarity_matrix = calc_cosine_similarity_matrix(
                self.polls_tf_idf_matrix, self.polls_tf_idf_matrix
            )

            self.userInteractions = self.elastic_handle.get_interactions(
                "userpollinteractions", user_id
            )

            self.userInteractions = [
                interaction["pollId"]
                for interaction in self.userInteractions["userPollActions"][:20]
            ]
            self.recommended_list = gen_rec_from_list_of_polls(
                self.userInteractions,
                self.polls,
                self.cosine_similarity_matrix,
                10,
            )

            recommended_polls = self.polls[self.polls["id"].isin(self.recommended_list)]

            recommended_polls = recommended_polls[
                ["id", "ownerId", "question", "options", "topics"]
            ].to_dict(orient="records")

            # print(f"{recommended_polls['poll_ID', 'author_ID', 'title', 'topic']}")
            # print(f"{recommended_polls.columns.to_list()}")
            # print(
            #    f"{recommended_polls['poll_ID', 'author_ID', 'title', 'option', 'topic']}"
            # )

            result = {
                "user_ID": user_id,
                "recommended_polls": recommended_polls,
            }

            return result, 200
            return self.userInteractions, 200
        except InteractionNotFound as e:
            exception = {
                "Message": e.args[0],
                "Error": "Value Error",
                "Code": e.args[1],
            }
            return jsonify(exception)
        except TlsError as e:
            exception = {
                "Message": e.args,
                "Error": "TLS Error",
                "Code": 120,
            }
            return jsonify(exception)


api.add_resource(Rec, "/get_rec/")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
