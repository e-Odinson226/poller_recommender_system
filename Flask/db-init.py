from elasticsearch import Elasticsearch, helpers
import configparser

from pathlib import Path

path = Path(__file__).resolve()


config = configparser.ConfigParser()
conf = config.read(Path(path, "/example.ini"))

print(conf)
