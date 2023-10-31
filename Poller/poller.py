from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import pandas as pd


import redis
import pickle

from .extensions import *
from .RecommenderSystem.recommender_system import *
from .ElasticSeachHandle.elasticsearch_handle import *

app = Flask(__name__)
api = Api(app)
print(f"--------------------\n** {app} Started **\n--------------------")
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)


try:
    redis_pool = create_redis_pool(host="localhost", port=6379, db=0)
    elastic_handle = create_elastic_connection(
        poller_elasticsearch_url="POLLER_ELASTICSEARCH_URL",
        poller_username="POLLER_USERNAME",
        poller_password="POLLER_PASSWORD",
        poller_fingerprint="POLLER_FINGERPRINT",
    )

    polls = elastic_handle.get_index("polls")
    polls_df = pd.DataFrame.from_records(polls)

except IndexError as e:
    raise e


class Rec(Resource):
    def get(self):
        # get user data from Redis
        user_id = request.args.get("userId")

        redis_connection = redis.Redis(connection_pool=redis_pool)

        if user_id is not None and redis_connection.exists(user_id):
            print(f"The data for {user_id} exists in Redis.")
            retrieved_data = redis_connection.get(user_id)
            deserialized_dict = pickle.loads(retrieved_data)

        else:
            # TODO query to database
            print(f"The was no entry for {user_id} in Redis.")

            # TODO: generate matrix
            return jsonify({"Error": "User matrix not found"})

        # TODO: redis error

        try:
            # Get the page number from the query parameters, default to page 1 if not provided
            page = int(request.args.get("page", 1))
            all = int(request.args.get("all", 0))
            items_per_page = int(request.args.get("page_size", 10))

            # Calculate the starting and ending indices for the current page
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page

            polls_tf_idf_matrix = deserialized_dict.get("polls_tf_idf_matrix")
            # filtered_polls_df = deserialized_dict.get("filtered_polls_df")
            filtered_polls_df = deserialized_dict.get("concatenated_df")

            # polls_tf_idf_matrix = pickle.loads(serialized_polls_tf_idf_matrix)

            cosine_similarity_matrix = calc_cosine_similarity_matrix(
                polls_tf_idf_matrix, polls_tf_idf_matrix
            )

            userInteractions = elastic_handle.get_interactions(
                "userpollinteractions", user_id
            )

            userInteractions = [
                interaction["pollId"]
                for interaction in userInteractions["userPollActions"][:20]
            ]

            recommended_polls_df = gen_rec_from_list_of_polls_df(
                interacted_polls=userInteractions,
                filtered_polls_df=filtered_polls_df,
                cosine_similarity_matrix=cosine_similarity_matrix,
                number_of_recommendations=100,
            )
            print(f"type: {type(recommended_polls_df)} len:{len(recommended_polls_df)}")
            # ordered_recommendations = order(recommended_polls_df)
            older_recommended_polls, newer_recommended_polls = split_df_by_lifetime(
                recommended_polls_df
            )

            trend_polls = deserialized_dict.get("filtered_trend_polls_list")
            trend_polls_df = list_to_df(trend_polls, filtered_polls_df)
            print(f"type: {type(trend_polls_df)} len:{len(trend_polls_df)}")

            # ordered_trend_polls_df = order(trend_polls_df)
            older_trend_polls, newer_trend_polls = split_df_by_lifetime(trend_polls_df)

            recommended_polls_list = pd.concat(
                [
                    newer_recommended_polls,
                    newer_trend_polls,
                    older_recommended_polls,
                    older_trend_polls,
                ],
                ignore_index=False,
            )

            # If you want to reset the index
            recommended_polls_list = recommended_polls_list.reset_index(drop=True)
            recommended_polls_list = recommended_polls_list["id"].tolist()
            recommended_polls_list = remove_duplicates(recommended_polls_list)

            print(f"-----------------------{type(filtered_polls_df)}")
            # recommended_polls = filtered_polls_df[
            #    filtered_polls_df["id"].isin(recommended_list)
            # ]
            # recommended_polls = recommended_polls["id"].tolist()

            total_recommended_polls_count = len(recommended_polls_list)

            if all == 1:
                try:
                    response = {
                        "list": "all",
                        "user_ID": user_id,
                        "total_count": total_recommended_polls_count,
                        "recommended_polls": recommended_polls_list,
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
            paginated_data = recommended_polls_list[start_idx:end_idx]

            # Calculate the total number of pages
            total_pages = len(recommended_polls_list) // items_per_page + (
                len(recommended_polls_list) % items_per_page > 0
            )

            # Create a response dictionary with the paginated data and pagination information
            response = {
                "list": "ordered_recom",
                "user_ID": user_id,
                "total_count": len(recommended_polls_list),
                "total_pages": total_pages,
                "page": page,
                "total_count": total_recommended_polls_count,
                "recommended_polls": paginated_data,
            }

            return jsonify(response)

        except InvalidParameterError as a:
            response = {
                "user_ID": user_id,
                "recommended_polls": [],
                "warning": "NO VALID POLL AVAILABLE",
            }

            return jsonify(response)

        except InteractionNotFound as e:
            # Slice the data to get the items for the current page
            # trend_polls = [poll["id"] for poll in trend_polls]
            trend_polls = deserialized_dict.get("filtered_trend_polls_list")

            page = int(request.args.get("page", 1))
            items_per_page = int(request.args.get("page_size", 10))

            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page

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
                "warning": "User has NO INTERACTION",
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
    def post(self):
        try:
            print(
                f"------------------------------\nGenerating user's matrix...\n------------------------------"
            )
            args = request.get_json(force=True)

            # user_id  = request.args.get("userId")
            user_id = args.get("userId")

            # constraint_parameters = request.args.get("constraint_parameters")
            constraint_parameters = args.get("constraint_parameters")

            polls = elastic_handle.get_index("polls")
            polls_df = pd.DataFrame.from_records(polls)
            filtered_polls_df = polls_df[
                polls_df.apply(filter_polls, args=(constraint_parameters,), axis=1)
            ]

            # filtered_polls_df = filtered_polls_df.reset_index(drop=True)
            allowed_polls_list = get_allowed_private_polls(
                # params={"userId": "bbe64b34-ba34-4fbd-a62f-e6c84c0423b4"}
                params={"userId": user_id}
            )
            allowed_private_polls = polls_df[polls_df["id"].isin(allowed_polls_list)]
            allowed_private_polls = allowed_private_polls[
                allowed_private_polls.apply(
                    filter_polls, args=(constraint_parameters,), axis=1
                )
            ]

            concatenated_df = pd.concat(
                [filtered_polls_df, allowed_private_polls], axis=0
            )
            # Reset the index if needed
            concatenated_df.reset_index(drop=True, inplace=True)

            # filtered_polls_csv = filtered_polls_df.to_csv("data.csv", index=False)
            # with open(
            #    "filtered_polls_csv.csv", "w", encoding="UTF8", newline=""
            # ) as file:
            #    writer = csv.writer(file)
            # filtered_polls_df.to_csv("filtered_polls_csv.csv")
            # filtered_polls_df.to_pickle("filtered_polls_pkl.pickle")

            polls_tf_idf_matrix = create_souped_tf_idf_matrix(concatenated_df)

            trend_polls = elastic_handle.get_trend_polls(polls)
            trend_polls_df = pd.DataFrame.from_records(trend_polls)
            filtered_trend_polls_df = trend_polls_df[
                trend_polls_df.apply(
                    filter_polls, args=(constraint_parameters,), axis=1
                )
            ]
            filtered_trend_polls_df = filtered_trend_polls_df.reset_index(drop=True)
            filtered_trend_polls_list = filtered_trend_polls_df["id"].tolist()

            print(f"filtered_trend_polls_list: {type(filtered_trend_polls_list)}")
            print(f"filtered_trend_polls_list: {len(filtered_trend_polls_list)}")

            # serialized_polls_tf_idf_matrix = pickle.dumps(polls_tf_idf_matrix)
            user_matrix = {
                "user_id": user_id,
                "polls_tf_idf_matrix": polls_tf_idf_matrix,
                # "filtered_polls_df": filtered_polls_df,
                "concatenated_df": concatenated_df[["id", "createdAt"]],
                "filtered_trend_polls_list": filtered_trend_polls_list,
            }
            serialized_data = pickle.dumps(user_matrix)
            redis_connection = redis.Redis(connection_pool=redis_pool)
            redis_connection.set(user_id, serialized_data)

            response = {
                "message": "record submited",
                "record id": user_id,
            }

            return jsonify(response)

        except ValueError as e:
            if concatenated_df.empty:
                raise ValueError("no valid poll for this user") from e

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
