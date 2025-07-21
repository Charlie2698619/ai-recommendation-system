import os
import io
import boto3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import pickle
import logging


REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "ecom-raw-events")
EMBEDDING_PREFIX = os.getenv("EMBEDDING_PREFIX", "embeddings")
ITEM_FEATURES_FILE = os.getenv("ITEM_FEATURES_FILE", "train/train_ready_batch_")

# hyperparameters
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", 100))
PCA_COMPONENTS = int(os.getenv("PCA_COMPONENTS", 64))


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
s3 = boto3.client('s3', region_name=REGION)


def list_parquet_files():
    logging.info("Listing parquet files in S3 bucket...")
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=ITEM_FEATURES_FILE)
    files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.parquet')]
    files.sort(key=lambda x: x.split('_')[-1])  # Sort by file suffix
    logging.info(f"Found {len(files)} parquet files.")
    return files

def load_parquet_from_s3(key):
    logging.info(f"Loading parquet file from S3: {key}")
    response = s3.get_object(Bucket=S3_BUCKET, Key=key)
    df = pd.read_parquet(io.BytesIO(response['Body'].read()))
    logging.info(f"Loaded {len(df)} rows from {key}.")
    return df


def preprocess_features(df):
    df = df.dropna(subset=['itemid']).copy()
    df['itemid'] = df['itemid'].astype(str)
    text_cols = [col for col in df.columns if df[col].dtype == 'object' and col not in ['itemid', 'user_id', 'event', 'event_timestamp']]
    df[text_cols] = df[text_cols].fillna('unknown').astype(str)
    df['combined_text'] = df[text_cols].agg(' '.join, axis=1)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_matrix = MinMaxScaler().fit_transform(df[numeric_cols]) if numeric_cols else None
    return df['itemid'].tolist(), df['combined_text'], numeric_matrix

def generate_embeddings(texts, numeric_matrix, tfidf_model=None):
    if tfidf_model is None:
        tfidf_model = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
        tfidf_matrix = tfidf_model.fit_transform(texts).toarray()
    else:
        tfidf_matrix = tfidf_model.transform(texts).toarray()

    if numeric_matrix is not None:
        full_matrix = np.hstack((tfidf_matrix, numeric_matrix))
    else:
        full_matrix = tfidf_matrix

    return full_matrix, tfidf_model

def reduce_dimensionality(matrix):
    if matrix.shape[1] > PCA_COMPONENTS:
        pca = PCA(n_components=PCA_COMPONENTS)
        return pca.fit_transform(matrix)
    return matrix

def save_embeddings_to_s3(itemids, vectors):
    logging.info("Saving final embeddings to S3...")
    buffer = io.BytesIO()
    pickle.dump({"itemid": itemids, "vectors": vectors}, buffer)
    buffer.seek(0)
    s3.upload_fileobj(buffer, S3_BUCKET, EMBEDDING_PREFIX)
    logging.info(f"Saved embeddings to s3://{S3_BUCKET}/{EMBEDDING_PREFIX}")

def main():
    logging.info("Starting item embedding generation...")
    all_vectors = []
    all_itemids = []

    parquet_files = list_parquet_files()
    tfidf_model = None  # Train on first batch

    # incrementally process each parquet file
    for idx, key in enumerate(parquet_files):
        logging.info(f"Processing file {idx+1}/{len(parquet_files)}: {key}")
        df = load_parquet_from_s3(key)
        itemids, texts, numeric_matrix = preprocess_features(df)

        if tfidf_model is None:
            vectors, tfidf_model = generate_embeddings(texts, numeric_matrix)
        else:
            vectors, _ = generate_embeddings(texts, numeric_matrix, tfidf_model)

        all_vectors.append(vectors)
        all_itemids.extend(itemids)

    # Stack and reduce once
    full_matrix = np.vstack(all_vectors)
    reduced_matrix = reduce_dimensionality(full_matrix)

    logging.info(f"Final embedding matrix shape: {reduced_matrix.shape}")
    save_embeddings_to_s3(all_itemids, reduced_matrix)
    logging.info("Item embeddings generation complete.")

def generate_item_embeddings():
    try:
        main()
    except Exception as e:
        logging.error(f"Error generating item embeddings: {e}")
        raise

if __name__ == "__main__":
    generate_item_embeddings()