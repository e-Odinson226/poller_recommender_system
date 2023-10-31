import redis
import os
from dotenv import load_dotenv
from sklearn.utils._param_validation import InvalidParameterError


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
