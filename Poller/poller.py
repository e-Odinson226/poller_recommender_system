from flask import Flask, redirect, url_for, jsonify, request
from flask_restful import Api, Resource
import pandas as pd
import os

from .RecommenderSystem.recommender_system import *
from .ElasticSeachHandle.elasticsearch_handle import *

app = Flask(__name__)
print(f"--------------------\n1. [{app}] Started")
api = Api(app)


try:
    elasticsearch_url = os.environ.get("POLLER_ELASTICSEARCH_URL")
    username = os.environ.get("POLLER_USERNAME")
    password = os.environ.get("POLLER_PASSWORD")
    fingerprint = os.environ.get("POLLER_FINGERPRINT")
    elastic_handle = ElasticsearchHandel(
        elasticsearch_url, username, password, fingerprint
    )
    if elasticsearch_url and username and password and fingerprint:
        print("--------------------\n2. Environment variables were read correctly.")

except ValueError as e:
    from dotenv import load_dotenv

    load_dotenv()
    elasticsearch_url = os.getenv("POLLER_ELASTICSEARCH_URL")
    username = os.getenv("POLLER_USERNAME")
    password = os.getenv("POLLER_PASSWORD")
    fingerprint = os.getenv("POLLER_FINGERPRINT")
    try:
        elastic_handle = ElasticsearchHandel(
            elasticsearch_url, username, password, fingerprint
        )
        if elasticsearch_url and username and password and fingerprint:
            print(
                f"--------------------\n2. Elastic variables were read from [.env] file."
            )
    except TypeError:
        print("--------------------\n2. Failed to read environment variables.")
        print(e)
        exit()


class Rec(Resource):
    def __init__(self):
        pd.set_option("display.max_columns", None)
        self.polls = elastic_handle.get_index("polls")
        elastic_handle.get_trend_polls()

    def get(self):
        try:
            user_id = request.args.get("userId")

            self.polls_df = pd.DataFrame.from_records(self.polls)
            # self.polls = encode_topics(self.polls_df)

            self.polls_tf_idf_matrix = create_tf_idf_matrix(self.polls_df, "topics")

            self.cosine_similarity_matrix = calc_cosine_similarity_matrix(
                self.polls_tf_idf_matrix, self.polls_tf_idf_matrix
            )

            self.userInteractions = elastic_handle.get_interactions(
                "userpollinteractions", user_id
            )

            self.userInteractions = [
                interaction["pollId"]
                for interaction in self.userInteractions["userPollActions"][:20]
            ]
            self.recommended_list = gen_rec_from_list_of_polls(
                self.userInteractions,
                self.polls_df,
                self.cosine_similarity_matrix,
                10,
            )

            recommended_polls = self.polls_df[
                self.polls_df["id"].isin(self.recommended_list)
            ]

            recommended_polls = recommended_polls[
                ["id", "ownerId", "question", "options", "topics"]
            ].to_dict(orient="records")

            result = {
                "user_ID": user_id,
                "recommended_polls": recommended_polls,
            }

            return result, 200

        except InteractionNotFound as e:
            result = {
                "recommended_polls": elastic_handle.trend_polls,
            }
            return jsonify(result)

        except TlsError as e:
            exception = {
                "Message": e.args,
                "Error": "TLS Error",
                "Code": 120,
            }
            return jsonify(exception)
        except ConnectionTimeout as e:
            exception = {
                "Message": e.args,
                "Error": "Elastic connection timed out",
                "Code": 130,
            }
            return jsonify(exception)


api.add_resource(Rec, "/get_rec/")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
