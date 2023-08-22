from flask import Flask, redirect, url_for, jsonify
from flask_restful import Api, Resource, reqparse
import json

from rec_sys import *

app = Flask(__name__)
api = Api(app)


class Rec(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        # Define the expected argument(s)

        self.parser.add_argument("data")

        self.parser.add_argument(
            "user_ID", type=str, required=True, help="User ID is required"
        )
        self.parser.add_argument(
            "interactions",
            type=list,
            required=True,
            action="append",
            help="Interactions list is required",
        )

        # self.parser.add_argument("user_ID", type=int, help="Description of param1")
        # self.parser.add_argument("interactions", type=list, action="append")

    def post(self):
        args = self.parser.parse_args()

        user_id = args.get("user_ID")
        interactions = args.get("interactions")

        print(f"user_ID: {user_id}")
        print("Interactions:")

        for interaction in interactions:
            print(interaction)

        result = {"user_id": user_id, "interactions": interactions}
        return result, 200

        # user_id = args["user_ID"]
        ## interactions = args["interactions"]
        # interactions = args.get("interactions")


api.add_resource(Rec, "/get_rec/")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
