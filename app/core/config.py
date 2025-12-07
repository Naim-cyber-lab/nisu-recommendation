import os
from dotenv import load_dotenv

load_dotenv()

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME", "elastic")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "changeme")

USERS_INDEX = "nisu_users"
INDEX_WINKERS = "nisu_winkers"
INDEX_EVENTS = "nisu_events"
EVENTS_INDEX = "nisu_events"
