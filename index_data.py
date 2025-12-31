import pandas as pd
import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers

#File written with help of Chatgpt

CSV_PATH = "steam_games_cleaned.csv"
INDEX_NAME = "steam_bm25"
OS_URL = "http://localhost:9200"

CHUNK_SIZE_CHARS = 1000   # simple, safe baseline chunking
BULK_BATCH_SIZE = 500     # bulk indexing for speed


# ----------------------------
# Opensearch
# ----------------------------

host = 'localhost'
port = 9200

# Create the osearch with SSL/TLS enabled, but hostname verification disabled.
load_dotenv()
PASSWORD = os.getenv("OPENSEARCH_PASSWORD")
osearch = OpenSearch(
    hosts = [{'host': host, 'port': port}],
    http_compress = True, # enables gzip compression for request bodies
    http_auth = ('admin',PASSWORD),
    use_ssl = True,
    verify_certs = False,
    ssl_assert_hostname = False,
    ssl_show_warn = False,
)



def create_index():


        # Full index body goes under `body`
    index_body = {
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "appid": {"type": "keyword"},
                "name": {"type": "text", "index": False},
                "text": {"type": "text"}  # BM25 will use this field
            }
        }
    }

    # Create index
    if not osearch.indices.exists(index=INDEX_NAME):
        osearch.indices.create(index=INDEX_NAME, body=index_body)


# ----------------------------
# Chunking (baseline, model-agnostic)
# ----------------------------
def chunk_text(text, max_chars=CHUNK_SIZE_CHARS):
    for i in range(0, len(text), max_chars):
        yield text[i:i + max_chars]

# ----------------------------
# Indexing pipeline
# ----------------------------
def index_documents():
    actions = []
    df = pd.read_csv(CSV_PATH, low_memory=False)

    for _,row in df.iterrows():
        appid = row["appid"]
        name = row["name"]
        description = row["short_description"]

        for chunk_id, chunk in enumerate(chunk_text(description)):
            doc_id = f"{appid}_{chunk_id}"

            doc = {
                "doc_id": doc_id,
                "appid": appid,
                "name": name,
                "text": chunk
            }

            actions.append({
                "_index": INDEX_NAME,
                "_id": doc_id,
                "_source": doc
            })

            if len(actions) >= BULK_BATCH_SIZE:
                helpers.bulk(osearch, actions)
                actions = []

    if actions:
        helpers.bulk(osearch, actions)

# ----------------------------
# Simple BM25 query
# ----------------------------
def bm25_search(query, size=10):
    response = osearch.search(
        index=INDEX_NAME,
        size=size,
        body={
        "query": {
            "match": {
                "text": query
            }
        }
    }
    )

    results = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        results.append({
            "appid": src["appid"],
            "name": src["name"],
            "score": hit["_score"]
        })

    return results

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print("Creating index...")
    create_index()

    print("Indexing documents...")
    index_documents()

    print("Done indexing.")
