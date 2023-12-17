from ast import Dict
from telnetlib import STATUS
from types import NoneType
from typing import Any
import redis
import os
from dotenv import load_dotenv
from sklearn.utils._param_validation import InvalidParameterError
import zlib
import base64
from pymongo import MongoClient
from io import BytesIO
from scipy.sparse import save_npz, load_npz
import pandas as pd
import time
import pickle


from .ElasticSeachHandle.elasticsearch_handle import *


def create_redis_pool(host, port, db):
    return redis.ConnectionPool(host=host, port=port, db=db)


def create_elastic_connection(
    poller_elasticsearch_url, poller_username, poller_password, poller_fingerprint
):
    try:
        elasticsearch_url = os.environ.get(poller_elasticsearch_url)
        username = os.environ.get(poller_username)
        password = os.environ.get(poller_password)
        fingerprint = os.environ.get(poller_fingerprint)
        elastic_handle = ElasticsearchHandel(
            elasticsearch_url, username, password, fingerprint
        )
        if elasticsearch_url and username and password and fingerprint:
            print(
                "[2. Environment variables were read correctly through (enivronment variables).]"
            )
            return elastic_handle
    except ConnectionTimeout as e:
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
                    "[2. Environment variables were read correctly through (getenv).]"
                )
            return elastic_handle
        except TypeError:
            print("[2. Failed to read environment variables.]")
            print(e)


def remove_duplicates(input_list):
    seen = set()
    result = []

    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def get_entity(redis_client, entity_key, extend_expiration=600):
    # Get the entity from Redis
    entity = redis_client.get(entity_key)

    # If the entity exists, reset the expiration time (e.g., to 60 seconds)

    if entity:
        redis_client.expire(entity_key, extend_expiration)
        print(f"The data for {entity_key} exists in Redis.")
        return entity


def find_duplicates(lst):
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return duplicates


def insert_item_to_mongodb(
    polls_tf_idf_matrix,
    collection,
    user_id,
    polls_df,
    filtered_trend_polls_list,
    timer=False,
):
    # Save the sparse matrix to a BytesIO buffer
    buffer = BytesIO()

    start_time = time.time()
    save_npz(buffer, polls_tf_idf_matrix)
    elapse_npz_time = time.time() - start_time

    # Reset the buffer position to the beginning
    buffer.seek(0)

    # Read the buffer content into binary data
    start_time = time.time()
    binary_data = buffer.read()
    binary_data_time = time.time() - start_time

    # Compress the binary data
    start_time = time.time()
    compressed_data = zlib.compress(binary_data)
    compressed_data_time = time.time() - start_time

    # Encode the compressed data as base64 for BSON storage
    start_time = time.time()
    encoded_data = base64.b64encode(compressed_data).decode("utf-8")
    encoded_data_time = time.time() - start_time

    polls_dict = polls_df.to_dict(orient="records")

    # Insert the encoded data into MongoDB
    start_time = time.time()

    # --------------

    filter_criteria = {"user_id": user_id}  # Replace with your actual filter criteria
    # Specify the replacement document

    replacement_document = {
        "user_id": user_id,
        "polls_tf_idf_matrix": encoded_data,
        "concatenated_df": polls_dict,
        "filtered_trend_polls_list": filtered_trend_polls_list,
    }
    # Replace the document
    collection.replace_one(filter_criteria, replacement_document)

    # collection.insert_one(
    #    {
    #        "user_id": user_id,
    #        "polls_tf_idf_matrix": encoded_data,
    #        "concatenated_df": polls_dict,
    #        "filtered_trend_polls_list": filtered_trend_polls_list,
    #    }
    # )

    # --------------
    insert_one_time = time.time() - start_time
    if timer:
        print(f"Function 'save_npz' took {elapse_npz_time:.4f} seconds.")
        print(f"Function 'buffer.read' took {binary_data_time:.4f} seconds.")
        print(f"Function 'zlib.compress' took {compressed_data_time:.4f} seconds.")
        print(f"Function 'base64.b64encode' took {encoded_data_time:.4f} seconds.")
        print(f"Function 'insert_one_time' took {insert_one_time:.4f} seconds.")


def read_matrix_from_mongodb(collection, user_id) -> dict[str, Any]:
    # Retrieve the document from MongoDB
    result = collection.find_one({"user_id": user_id})

    print(result)
    # Decode the base64 data
    encoded_data = result.get("polls_tf_idf_matrix", "")
    compressed_data = base64.b64decode(encoded_data)

    # Decompress the data
    binary_data = zlib.decompress(compressed_data)

    # Create a BytesIO object from the binary data
    buffer = BytesIO(binary_data)

    # Load the sparse matrix from the BytesIO buffer
    polls_tf_idf_matrix = load_npz(buffer)

    # Create a DataFrame from the concatenated_df
    concatenated_df = pd.DataFrame(result.get("concatenated_df"))

    # Get other values
    user_id = result.get("user_id", "")
    filtered_trend_polls_list = result.get("filtered_trend_polls_list", [])

    return {
        "user_id": user_id,
        "polls_tf_idf_matrix": polls_tf_idf_matrix,
        "concatenated_df": concatenated_df,
        "filtered_trend_polls_list": filtered_trend_polls_list,
    }


def save_matrix_to_mongodb_file(matrix, collection, user_id):
    # Save the sparse matrix to a compressed file
    save_npz("matrix.npz", matrix)

    # Read the compressed file into a binary stream
    with open("matrix.npz", "rb") as file:
        binary_data = file.read()

    # Compress the binary data
    compressed_data = zlib.compress(binary_data)

    # Encode the compressed data as base64 for BSON storage
    encoded_data = base64.b64encode(compressed_data).decode("utf-8")

    # Insert the encoded data into MongoDB

    collection.insert_one({"user_id": user_id, "sparse_matrix": encoded_data})


def get_user_entity(user_id, redis_connection, mongo_collection):
    # Get the entity from Redis
    # print(f"redis_connection.exists(user_id):{redis_connection.exists(user_id)}")
    if redis_connection.exists(user_id):
        print(f"Getting user entity from redis")
        serialized_user_entity = redis_connection.get(user_id)
        user_entity = pickle.loads(serialized_user_entity)
        redis_connection.expire(user_id, 5)
        return user_entity

    else:
        print(f"Getting user entity from mongo")
        return read_matrix_from_mongodb(mongo_collection, user_id)


#
## user_entity = read_matrix_from_mongodb(collection, user_id)
#
# if user_entity:
#    # Serialize the data using pickle
#    serialized_data = pickle.dumps(user_entity)
#
#    # Cache the data in Redis
#    redis_connection = redis.Redis(connection_pool=redis_pool)
#    redis_connection.set(user_id, serialized_data)
#
#    # user_entity = pickle.loads(serialized_user_entity)
#
#    # Get the page number from the query parameters, default to page 1 if not provided
#    page = int(request.args.get("page", 1))
#    all = int(request.args.get("all", 0))
#    items_per_page = int(request.args.get("page_size", 10))
#
#    # Calculate the starting and ending indices for the current page
#    start_idx = (page - 1) * items_per_page
#    end_idx = start_idx + items_per_page
#
#    polls_tf_idf_matrix = user_entity.get("polls_tf_idf_matrix")
#    filtered_polls_df = user_entity.get("concatenated_df")
#    print(f"type[polls_tf_idf_matrix]:{type(polls_tf_idf_matrix)}")
#
#    cosine_similarity_matrix = calc_cosine_similarity_matrix(
#        polls_tf_idf_matrix, polls_tf_idf_matrix
#    )
#
#    userInteractions = elastic_handle.get_interactions(
#        "userpollinteractions", user_id
#    )
#
#    userInteractions = [
#        interaction["pollId"]
#        for interaction in userInteractions["userPollActions"][:20]
#    ]
#
#    recommended_polls_df = gen_rec_from_list_of_polls_df(
#        interacted_polls=userInteractions,
#        filtered_polls_df=filtered_polls_df,
#        cosine_similarity_matrix=cosine_similarity_matrix,
#        number_of_recommendations=100,
#    )
#
#    trend_polls = user_entity.get("filtered_trend_polls_list")
#    trend_polls_df = list_to_df(trend_polls, filtered_polls_df)
#
#    live_polls_flag = int(request.args.get("live_polls", 0))
#    recommended_polls_list = order_v4(
#        recommended_polls_df=recommended_polls_df,
#        trend_polls_df=trend_polls_df,
#        live_polls_flag=live_polls_flag,
#    )
#
#    total_recommended_polls_count = len(recommended_polls_list)
#
#    if all == 1:
#        try:
#            response = {
#                "list": "all",
#                "user_ID": user_id,
#                "total_count": total_recommended_polls_count,
#                "recommended_polls": recommended_polls_list,
#                "Code": 200,
#            }
#
#            return jsonify(response)
#        except elastic_transport.ConnectionTimeout as e:
#            exception = {
#                "Message": e.args,
#                "Error": "Elastic connection timed out",
#                "Code": 130,
#            }
#            return jsonify(exception)
#
#    # Slice the data to get the items for the current page
#    paginated_data = recommended_polls_list[start_idx:end_idx]
#
#    # Calculate the total number of pages
#    total_pages = len(recommended_polls_list) // items_per_page + (
#        len(recommended_polls_list) % items_per_page > 0
#    )
#
#    # Create a response dictionary with the paginated data and pagination information
#    response = {
#        "source": "mongo",
#        "list": "ordered_recom",
#        "user_ID": user_id,
#        "total_count": len(recommended_polls_list),
#        "total_pages": total_pages,
#        "page": page,
#        "total_count": total_recommended_polls_count,
#        "recommended_polls": paginated_data,
#        "Code": 200,
#    }
#
#    return jsonify(response)
## No data found for this user
# else:
#    exception = {
#        "list": "error",
#        "user_ID": user_id,
#        "page": 0,
#        "total_count": 0,
#        "recommended_polls": [],
#        "warning": "No entry in database",
#        "Code": 110,
#    }
#    return jsonify(exception)
#
# return user_entity
#
