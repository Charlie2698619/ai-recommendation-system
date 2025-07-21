import os
import io
import pickle
import boto3
import numpy as np
import faiss
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "ecom-raw-events")
EMBEDDING_PREFIX = os.getenv("EMBEDDING_PREFIX", "embeddings.pkl")
FAISS_INDEX_FILE = os.getenv("FAISS_INDEX_FILE", "faiss.index")
ITEMID_MAP_FILE = os.getenv("ITEMID_MAP_FILE", "itemid_map.pkl")

s3 = boto3.client("s3", region_name=REGION)

def load_embeddings():
    logging.info("Loading embeddings from S3 bucket: %s, key: %s", S3_BUCKET, EMBEDDING_PREFIX)
    response = s3.get_object(Bucket=S3_BUCKET, Key=EMBEDDING_PREFIX)
    buffer = io.BytesIO(response['Body'].read())
    data = pickle.load(buffer)
    logging.info("Loaded %d embeddings.", len(data['itemid']))
    return data['itemid'], np.array(data['vectors'], dtype=np.float32)

def normalize_vectors(vectors):
    logging.info("Normalizing vectors.")
    faiss.normalize_L2(vectors)
    return vectors

def build_faiss_index(vectors):
    logging.info("Building FAISS index.")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    logging.info("FAISS index built with %d vectors.", vectors.shape[0])
    return index

def save_index_to_s3(index, itemid):
    logging.info("Saving FAISS index to S3: %s", FAISS_INDEX_FILE)
    index_bytes = faiss.serialize_index(index)
    index_buffer = io.BytesIO(index_bytes)
    s3.upload_fileobj(index_buffer, S3_BUCKET, FAISS_INDEX_FILE)
    logging.info("FAISS index saved to s3://%s/%s", S3_BUCKET, FAISS_INDEX_FILE)
    
    # save itemid map
    logging.info("Saving item ID map to S3: %s", ITEMID_MAP_FILE)
    itemid_map_buffer = io.BytesIO()
    pickle.dump(itemid, itemid_map_buffer)
    itemid_map_buffer.seek(0)
    s3.upload_fileobj(itemid_map_buffer, S3_BUCKET, ITEMID_MAP_FILE)
    logging.info("Item ID map saved to s3://%s/%s", S3_BUCKET, ITEMID_MAP_FILE)
     
def main():
    logging.info("Starting FAISS index training process.")
    itemid, vectors = load_embeddings()
    logging.info("Loaded %d itemid with vectors shape %s.", len(itemid), vectors.shape)

    vectors = normalize_vectors(vectors)
    logging.info("Vectors normalized.")

    index = build_faiss_index(vectors)
    logging.info("FAISS index built.")

    save_index_to_s3(index, itemid)
    logging.info("FAISS index and item ID map saved successfully.")
    logging.info("Training complete.") 

def train_faiss_index():
    """Main function to train FAISS index."""
    try:
        main()
    except Exception as e:
        logging.error("Error during FAISS index training: %s", e, exc_info=True)
        raise
    
if __name__ == "__main__":
    train_faiss_index()