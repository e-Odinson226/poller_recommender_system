from flask import Flask, redirect, url_for, jsonify
from flask_restful import Api, Resource, reqparse

from rec_sys import *

app = Flask(__name__)
api = Api(app)


class Rec(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        # Define the expected argument(s)
        self.parser.add_argument("user_ID", type=int, help="Description of param1")
        self.parser.add_argument("aciton", type=str, help="Description of param2")

    def post(self, user_id):
        args = self.parser.parse_args()
        args["polls"]
        print(args)
        # response = { args}
        return args


api.add_resource(Rec, "/get_rec/<int:user_id>")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
