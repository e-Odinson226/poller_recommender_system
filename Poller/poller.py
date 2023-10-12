from flask import Flask, redirect, url_for, jsonify, request
from flask_restful import Api, Resource
import pandas as pd
from dotenv import load_dotenv
import os
import redis
import pickle

from .RecommenderSystem.recommender_system import *
from .ElasticSeachHandle.elasticsearch_handle import *

app = Flask(__name__)
api = Api(app)
print(f"--------------------\n1. [{app}] Started")


def inject_dependencies(func):
    def decorated_function(*args, **kwargs):
        try:
            # Create or obtain your dependencies here
            elasticsearch_url = os.environ.get("POLLER_ELASTICSEARCH_URL")
            username = os.environ.get("POLLER_USERNAME")
            password = os.environ.get("POLLER_PASSWORD")
            fingerprint = os.environ.get("POLLER_FINGERPRINT")
            elastic_handle = ElasticsearchHandel(
                elasticsearch_url, username, password, fingerprint
            )
            if elasticsearch_url and username and password and fingerprint:
                print(
                    f"--------------------\n2. Environment variables were read correctly."
                )
                # print(f"\tELASTIC_USERNAME: {elasticsearch_url} ")
                # print(f"\tELASTIC_USERNAME: {username} ")
                # print(f"\tELASTIC_USERNAME: {password} ")
                # print(f"\tELASTIC_USERNAME: {fingerprint} ")
            kwargs["elastic_handle"] = elastic_handle

            # polls = elastic_handle.get_index("polls")
            # polls_df = pd.DataFrame.from_records(polls)
            # kwargs["polls_df"] = polls_df
            # kwargs["polls"] = polls

            trend_polls = elastic_handle.get_trend_polls(polls)
            kwargs["trend_polls"] = trend_polls

            r = redis.Redis(host="localhost", port=6379, db=0)
            kwargs["r"] = r

            # pd.set_option("display.max_columns", None)

            return func(*args, **kwargs)

        except ConnectionTimeout as e:
            print(type(e))

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
                    print(f"---\n2. Elastic variables were read from [.env] file.")
                    print(f"---\n{elasticsearch_url}")
                    print(f"---\n{username}")
                    print(f"---\n{password}")
                    print(f"---\n{fingerprint}")
            except TypeError:
                print("--------------------\n2. Failed to read environment variables.")
                print(e)
                exit()
        except IndexError as e:
            raise e
            exit()

    return decorated_function


class Rec(Resource):
    @inject_dependencies
    def get(self, elastic_handle, r, polls, trend_polls, polls_df):
        try:
            user_id = request.args.get("userId")

            retrieved_data = r.get(user_id)
            if retrieved_data:
                deserialized_dict = pickle.loads(retrieved_data)
                # print(deserialized_dict.get("user_matrix"))
            else:
                # TODO: generate matrix
                return jsonify({"Error": "User matrix not found"})

        # TODO: redis error
        except ConnectionTimeout as e:
            exception = {
                "Message": e.args,
                "Error": "Elastic connection timed out",
                "Code": 130,
            }
            return jsonify(exception)

        try:
            # Get the page number from the query parameters, default to page 1 if not provided
            page = int(request.args.get("page", 1))
            all = int(request.args.get("all", 0))
            items_per_page = int(request.args.get("page_size", 10))

            # Calculate the starting and ending indices for the current page
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page

            polls_tf_idf_matrix = deserialized_dict.get("user_matrix")
            # polls_tf_idf_matrix = pickle.loads(serialized_polls_tf_idf_matrix)

            self.cosine_similarity_matrix = calc_cosine_similarity_matrix(
                polls_tf_idf_matrix, polls_tf_idf_matrix
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
                polls_df,
                self.cosine_similarity_matrix,
                100,
            )

            recommended_polls = polls_df[polls_df["id"].isin(self.recommended_list)]

            # recommended_polls = recommended_polls[
            #    ["id", "ownerId", "question", "options", "topics"]
            # ].to_dict(orient="records")
            recommended_polls = recommended_polls["id"].tolist()
            total_recommended_polls_count = len(recommended_polls)
            if all == 1:
                try:
                    response = {
                        "list": "all",
                        "user_ID": user_id,
                        "total_count": total_recommended_polls_count,
                        "recommended_polls": recommended_polls,
                    }

                    return jsonify(response)
                except elastic_transport.ConnectionTimeout as e:
                    exception = {
                        "Message": e.args,
                        "Error": "Elastic connection timed out",
                        "Code": 130,
                    }
                    return jsonify(exception)

            # Slice the data to get the items for the current page
            paginated_data = recommended_polls[start_idx:end_idx]

            # Calculate the total number of pages
            total_pages = len(recommended_polls) // items_per_page + (
                len(recommended_polls) % items_per_page > 0
            )

            # Create a response dictionary with the paginated data and pagination information
            response = {
                "list": "recom",
                "user_ID": user_id,
                "page": page,
                "total_count": total_recommended_polls_count,
                "recommended_polls": paginated_data,
            }

            return jsonify(response)

        except InteractionNotFound as e:
            user_id = request.args.get("userId")

            # Get the page number from the query parameters, default to page 1 if not provided
            page = int(request.args.get("page", 1))

            items_per_page = int(request.args.get("page_size", 10))

            # Calculate the starting and ending indices for the current page
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page

            # Slice the data to get the items for the current page
            trend_polls = [poll["id"] for poll in trend_polls]
            paginated_data = trend_polls[start_idx:end_idx]

            # Calculate the total number of pages
            total_pages = len(trend_polls) // items_per_page + (
                len(trend_polls) % items_per_page > 0
            )

            # Create a response dictionary with the paginated data and pagination information
            response = {
                "list": "trend",
                "user_ID": user_id,
                "page": page,
                "total_count": len(trend_polls),
                "recommended_polls": paginated_data,
            }

            return jsonify(response)
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


class Gen(Resource):
    @inject_dependencies
    def get(self, elastic_handle, r, trend_polls):
        try:
            print(f"FLAG ------------------------------ Generating user's matrix ")
            user_id = request.args.get("userId")

            polls = elastic_handle.get_index("polls")

            # TODO: apply user's specific restrictions and filters
            # filters: dict
            # filters = elastic.get_filters(user_id)
            # polls = elastic.apply_filters(polls, filters)

            polls_df = pd.DataFrame.from_records(polls)
            polls_tf_idf_matrix = create_souped_tf_idf_matrix(polls_df)
            # serialized_polls_tf_idf_matrix = pickle.dumps(polls_tf_idf_matrix)

            user_matrix = {
                "user_id": user_id,
                "user_matrix": polls_tf_idf_matrix,
            }
            serialized_data = pickle.dumps(user_matrix)
            r.set(user_id, serialized_data)

            response = {
                "message": "record submited",
                "record id": user_id,
            }

            return jsonify(response)

        except InteractionNotFound as e:
            user_id = request.args.get("userId")

            # Get the page number from the query parameters, default to page 1 if not provided
            page = int(request.args.get("page", 1))

            items_per_page = int(request.args.get("page_size", 10))

            # Calculate the starting and ending indices for the current page
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page

            # Slice the data to get the items for the current page
            # print(len(trend_polls[0]))

            trend_polls = [poll["id"] for poll in trend_polls]

            paginated_data = trend_polls[start_idx:end_idx]

            # Calculate the total number of pages
            total_pages = len(trend_polls) // items_per_page + (
                len(trend_polls) % items_per_page > 0
            )

            # Create a response dictionary with the paginated data and pagination information
            response = {
                "list": "trend",
                "user_ID": user_id,
                "page": page,
                "total_count": len(trend_polls),
                "recommended_polls": paginated_data,
            }

            return jsonify(response)
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


api.add_resource(Gen, "/gen_mat/")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
