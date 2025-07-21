import os, io, pickle, boto3, faiss, numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "ecom-raw-events")
FAISS_INDEX_FILE = os.getenv("FAISS_INDEX_FILE", "faiss.index")
ITEMID_MAP_FILE = os.getenv("ITEMID_MAP_FILE", "itemid_map.pkl")





def load_faiss_index():
    logging.info("Loading FAISS index from S3: %s", FAISS_INDEX_FILE)
    s3 = boto3.client("s3", region_name=REGION)
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, FAISS_INDEX_FILE, buf)
    buf.seek(0)
    index = faiss.read_index(faiss.PyCallbackIOReader(buf.read))
    logging.info("FAISS index loaded successfully.")
    return index

def load_itemid_map():
    logging.info("Loading item ID map from S3: %s", ITEMID_MAP_FILE)
    s3 = boto3.client("s3", region_name=REGION)
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, ITEMID_MAP_FILE, buf)
    buf.seek(0)
    itemid_ids = pickle.load(buf)
    logging.info("Item ID map loaded successfully.")
    # convert all item IDs to whole numbers
    itemid_ids = [str(int(float(itemid))) for itemid in itemid_ids] 
    
    logging.info("Item IDs converted to string format.")
    
    return {
        "itemid_to_index": {itemid: idx for idx, itemid in enumerate(itemid_ids)},
        "index_to_itemid": {idx: itemid for idx, itemid in enumerate(itemid_ids)},
    }

def get_similar_items(itemid, index, itemid_to_index, index_to_itemid, k=5):
    logging.info("Querying similar items for itemid: %s", itemid)
    if itemid not in itemid_to_index:
        logging.error("Item ID %s not found in index.", itemid)
        raise ValueError("Item ID not found.")
    query_idx = itemid_to_index[itemid]
    query_vec = index.reconstruct(query_idx).reshape(1, -1)
    scores, indices = index.search(query_vec, k + 1)
    similar_items = [index_to_itemid[i] for i in indices[0] if index_to_itemid[i] != itemid][:k]
    logging.info("Found %d similar items for itemid: %s", len(similar_items), itemid)
    
    return similar_items

# Uncomment the following lines to test the function directly
# def query_faiss():
#     test_itemid = "49337"
#     try:
#         similar_items = get_similar_items(test_itemid, load_faiss_index(), load_itemid_map()["itemid_to_index"], load_itemid_map()["index_to_itemid"])
#         logging.info("Similar items for %s: %s", test_itemid, similar_items)
#     except ValueError as e:
#         logging.error("Error retrieving similar items: %s", e)

# if __name__ == "__main__":
#     query_faiss()