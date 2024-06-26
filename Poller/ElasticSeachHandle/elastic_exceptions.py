from elastic_transport import TlsError, ConnectionTimeout, ConnectionError


class InteractionNotFound(Exception):
    args = ("User doesn't have any interactions.", 110)


class ElasticConnectionError(TlsError):
    args = ("Elastic connection didn't established.", 120)
