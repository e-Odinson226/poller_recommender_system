from flask import Flask, redirect, url_for, jsonify, request
from flask_restful import Api, Resource
import json

import rec_sys as rs

app = Flask(__name__)
api = Api(app)


class Rec(Resource):
    def __init__(self) -> None:
        self.polls = rs.get_polls_list("/data/polls_synthetic.csv")
        self.liked_polls = []

    def post(self):
        args = request.get_json(force=True)
        user_id = args.get("user_ID")
        interactions = args.get("interactions")

        self.polls = rs.encode_topics(self.polls)
        polls_tf_idf_matrix = rs.create_tf_idf_matrix(self.polls, "topic")
        cosine_similarity_matrix = rs.calc_cosine_similarity_matrix(
            polls_tf_idf_matrix, polls_tf_idf_matrix
        )

        self.recommended_list = rs.gen_rec_from_list_of_polls(
            [dic["poll_ID"] for dic in interactions],
            self.polls,
            cosine_similarity_matrix,
            10,
        )

        recommended_polls = self.polls[
            self.polls["poll_ID"].isin(self.recommended_list)
        ]

        recommended_polls = recommended_polls[
            ["poll_ID", "author_ID", "title", "option", "topic"]
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


api.add_resource(Rec, "/get_rec/")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
