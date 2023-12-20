from flask import Flask, jsonify, request
from flask_restful import Api, Resource


from .extensions import *
from .RecommenderSystem.recommender_system import *
from .ElasticSeachHandle.elasticsearch_handle import *

app = Flask(__name__)
api = Api(app)
print(f"--------------------\n** {app} Started **\n--------------------")
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)


try:
    # print(str(os.environ.get("private_polls_API")))
    private_polls_API = (
        "https://"
        + str(os.environ.get("private_polls_API"))
        + "/api/Recommend/Polls/GetPrivatePollThatUserCanSee"
    )
    # print(private_polls_API)

    if private_polls_API is None:
        raise ValueError("private_polls_API is None")

    redis_pool = create_redis_pool(host="localhost", port=6379, db=0)
    elastic_handle = create_elastic_connection(
        poller_elasticsearch_url="POLLER_ELASTICSEARCH_URL",
        poller_username="POLLER_USERNAME",
        poller_password="POLLER_PASSWORD",
        poller_fingerprint="POLLER_FINGERPRINT",
    )

    # polls = elastic_handle.get_index("polls")
    # polls_df = pd.DataFrame.from_records(polls)
    client = MongoClient("mongodb://localhost:27017")
    db = client["pollet"]
    collection = db["tfidf_matrix"]

except IndexError as e:
    raise e


class Rec(Resource):
    def get(self):
        try:
            print(
                f"------------------------------\nProcessing the 'Rec' GET request..."
            )
            # get user data from Redis
            user_id = request.args.get("userId")
            redis_connection = redis.Redis(connection_pool=redis_pool)
            user_entity = get_user_entity(
                user_id=user_id,
                mongo_collection=collection,
                redis_connection=redis_connection,
            )

            # If the entity exists, reset the expiration time (e.g., to 60 seconds)
            redis_connection.expire(user_id, 10)

            # Get the page number from the query parameters, default to page 1 if not provided
            page = int(request.args.get("page", 1))
            all = int(request.args.get("all", 0))
            items_per_page = int(request.args.get("page_size", 10))

            # Calculate the starting and ending indices for the current page
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page

            polls_tf_idf_matrix = user_entity.get("polls_tf_idf_matrix")
            filtered_polls_df = user_entity.get("concatenated_df")
            print(f"type[polls_tf_idf_matrix]:{type(polls_tf_idf_matrix)}")

            start = time.time()
            cosine_similarity_matrix = calc_cosine_similarity_matrix(
                polls_tf_idf_matrix, polls_tf_idf_matrix
            )
            calc_cosine_similarity_matrix_time = time.time() - start
            print(
                f"[calc_cosine_similarity_matrix]:{calc_cosine_similarity_matrix_time:.4f}"
            )

            start = time.time()
            userInteractions = elastic_handle.get_interactions(
                "userpollinteractions", user_id
            )
            elastic_handle_get_interactions_time = time.time() - start
            print(
                f"[elastic_handle_get_interactions_time]:{elastic_handle_get_interactions_time:.4f}"
            )

            userInteractions = [
                interaction["pollId"]
                for interaction in userInteractions["userPollActions"][:20]
            ]

            start = time.time()
            recommended_polls_df = gen_rec_from_list_of_polls_df(
                interacted_polls=userInteractions,
                filtered_polls_df=filtered_polls_df,
                cosine_similarity_matrix=cosine_similarity_matrix,
                number_of_recommendations=100,
            )
            gen_rec_from_list_of_polls_df_time = time.time() - start
            print(
                f"[gen_rec_from_list_of_polls_df_time]:{gen_rec_from_list_of_polls_df_time:.4f}"
            )

            trend_polls = user_entity.get("filtered_trend_polls_list")
            trend_polls_df = list_to_df(trend_polls, filtered_polls_df)

            live_polls_flag = int(request.args.get("live_polls", 0))

            start = time.time()
            recommended_polls_list = order_v5(
                recommended_polls_df=recommended_polls_df,
                trend_polls_df=trend_polls_df,
                live_polls_flag=live_polls_flag,
            )
            print(f"recommended_polls_list: -------------{recommended_polls_list}")

            order_time = time.time() - start
            print(f"[order_time]:{order_time:.4f}")

            # print(
            #    f"-------------- is distinct = {len(recommended_polls_list) == len(set(recommended_polls_list))}"
            # )
            # total_recommended_polls_count = len(recommended_polls_list)

            if all == 1:
                try:
                    response = {
                        "list": "all",
                        "user_ID": user_id,
                        # "total_count": total_recommended_polls_count,
                        "recommended_polls": recommended_polls_list,
                        "Code": 200,
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
                "source": "redis",
                "list": "ordered_recom",
                "user_ID": user_id,
                "total_count": len(recommended_polls_list),
                "total_pages": total_pages,
                "page": page,
                "total_count": total_recommended_polls_count,
                "recommended_polls": paginated_data,
                "Code": 200,
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
            page = int(request.args.get("page", 1))
            items_per_page = int(request.args.get("page_size", 10))
            all = int(request.args.get("all", 0))

            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page

            user_entity = get_user_entity(
                user_id=user_id,
                mongo_collection=collection,
                redis_connection=redis_connection,
            )

            trend_polls = user_entity.get("filtered_trend_polls_list")
            filtered_polls_df = user_entity.get("concatenated_df")

            trend_polls_df = list_to_df(trend_polls, filtered_polls_df)

            recommended_polls_list = order_v5(
                trend_polls_df,
            )
            print(
                f"-------------- is distinct = {len(recommended_polls_list) == len(set(recommended_polls_list))}"
            )
            if all == 1:
                try:
                    response = {
                        "list": "trend-all",
                        "user_ID": user_id,
                        "total_count": len(recommended_polls_list),
                        "recommended_polls": recommended_polls_list,
                        "warning": "User has NO INTERACTION",
                        "Code": 200,
                    }

                    return jsonify(response)
                except elastic_transport.ConnectionTimeout as e:
                    exception = {
                        "Message": e.args,
                        "Error": "Elastic connection timed out",
                        "Code": 130,
                    }
                    return jsonify(exception)
            paginated_data = recommended_polls_list[start_idx:end_idx]

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
                "Code": 200,
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
            print(f"------------------------------  Processing the post request...")
            start_post = time.time()
            args = request.get_json(force=True)
            user_id = args.get("userId")
            constraint_parameters = args.get("constraint_parameters")

            start = time.time()
            polls = elastic_handle.get_index("polls")
            polls_df = pd.DataFrame.from_records(polls)

            filtering_time = time.time() - start
            print(f"elastic_handle.get_index('polls') duration:{filtering_time:.4f}")

            filtered_polls_df = polls_df[
                polls_df.apply(filter_polls, args=(constraint_parameters,), axis=1)
            ]

            allowed_polls_list = get_allowed_private_polls(
                {"userId": user_id}, private_polls_API
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

            # print(f"------------------------------\nGenerating user's matrix...")
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
            # serialized_polls_tf_idf_matrix = pickle.dumps(polls_tf_idf_matrix)
            start = time.time()
            user_matrix = {
                "user_id": user_id,
                "polls_tf_idf_matrix": polls_tf_idf_matrix,
                # "filtered_polls_df": filtered_polls_df,
                "concatenated_df": concatenated_df[
                    ["id", "createdAt", "endedAt", "valid"]
                ],
                "filtered_trend_polls_list": filtered_trend_polls_list,
            }

            serialized_data = pickle.dumps(user_matrix)
            redis_connection = redis.Redis(connection_pool=redis_pool)
            dump_time = time.time() - start
            print(f"Dumping duration:{dump_time:.4f}")

            redis_connection.set(user_id, serialized_data)

            # Save each matrix to MongoDB
            start = time.time()
            insert_item_to_mongodb(
                polls_tf_idf_matrix=polls_tf_idf_matrix,
                collection=collection,
                user_id=user_id,
                polls_df=concatenated_df[["id", "createdAt", "endedAt", "valid"]],
                filtered_trend_polls_list=filtered_trend_polls_list,
            )
            save_to_mongo = time.time() - start
            print(f"Save to DB duration:{save_to_mongo:.4f}")

            response = {
                "message": "record submited",
                "record id": user_id,
            }
            end_post = time.time() - start_post
            print(f"Post process duration:{end_post:.4f}")
            print(
                f"------------------------------  The post request processing has been completed.\n"
            )
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
