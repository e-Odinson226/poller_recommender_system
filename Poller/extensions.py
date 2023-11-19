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


def check_key_exists(redis_client, key):
    if not redis_client.exists(key):
        raise KeyError(f"The key '{key}' does not exist in Redis.")


def find_duplicates(lst):
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return duplicates


# duplicates = find_duplicates(recommended_polls_list)

# for item in duplicates:
#    print(f"{item} is a duplicate.")
#
# print(len(list(duplicates)))


def save_matrix_to_mongodb(
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

    # Insert the encoded data into MongoDB
    collection.insert_one(
        {
            "user_id": user_id,
            "polls_tf_idf_matrix": encoded_data,
            "concatenated_df": polls_dict,
            "filtered_trend_polls_list": filtered_trend_polls_list,
        }
    )


def read_matrix_from_mongodb(collection, user_id):
    # Retrieve the document from MongoDB
    result = collection.find_one({"user_id": user_id})

    if result:
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
        concatenated_df = pd.DataFrame(result.get("concatenated_df", []))

        # Get other values
        user_id = result.get("user_id", "")
        filtered_trend_polls_list = result.get("filtered_trend_polls_list", [])

        return {
            "user_id": user_id,
            "polls_tf_idf_matrix": polls_tf_idf_matrix,
            "concatenated_df": concatenated_df,
            "filtered_trend_polls_list": filtered_trend_polls_list,
        }
    else:
        return None


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
