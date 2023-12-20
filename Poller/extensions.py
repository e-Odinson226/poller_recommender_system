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
):
    # Save the sparse matrix to a BytesIO buffer
    buffer = BytesIO()

    save_npz(buffer, polls_tf_idf_matrix)

    # Reset the buffer position to the beginning
    buffer.seek(0)

    # Read the buffer content into binary data

    binary_data = buffer.read()

    # Compress the binary data

    compressed_data = zlib.compress(binary_data)

    # Encode the compressed data as base64 for BSON storage

    encoded_data = base64.b64encode(compressed_data).decode("utf-8")

    polls_dict = polls_df.to_dict(orient="records")

    filter_criteria = {"user_id": user_id}  # Replace with your actual filter criteria
    # Specify the replacement document

    replacement_document = {
        "user_id": user_id,
        "polls_tf_idf_matrix": encoded_data,
        "concatenated_df": polls_dict,
        "filtered_trend_polls_list": filtered_trend_polls_list,
    }
    # Replace the document
    collection.replace_one(filter_criteria, replacement_document, upsert=True)

    # collection.insert_one(
    #    {
    #        "user_id": user_id,
    #        "polls_tf_idf_matrix": encoded_data,
    #        "concatenated_df": polls_dict,
    #        "filtered_trend_polls_list": filtered_trend_polls_list,
    #    }
    # )


def read_matrix_from_mongodb(collection, user_id) -> dict[str, Any]:
    # Retrieve the document from MongoDB
    result = collection.find_one({"user_id": user_id})

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


def get_user_entity(user_id, redis_connection, mongo_collection):
    # Get the entity from Redis
    # print(f"redis_connection.exists(user_id):{redis_connection.exists(user_id)}")
    if redis_connection.exists(user_id):
        print(f"Getting user entity from redis")
        serialized_user_entity = redis_connection.get(user_id)
        user_entity = pickle.loads(serialized_user_entity)
        redis_connection.expire(user_id, 5)
        return user_entity, "redis"

    else:
        print(f"Getting user entity from mongo")
        user_matrix = read_matrix_from_mongodb(mongo_collection, user_id)
        serialized_data = pickle.dumps(user_matrix)
        redis_connection.set(user_id, serialized_data, ex=10)
        return user_matrix, "mongo"
