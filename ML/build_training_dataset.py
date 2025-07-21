import pandas as pd
import boto3
import os
import io
import json
import logging
from decimal import Decimal
import time
from botocore.exceptions import EndpointConnectionError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("S3_BUCKET", "ecom-raw-events")
TABLE_NAME = os.getenv("DYNAMODB_TABLE", "user_interactions")
OUTPUT_PREFIX = os.getenv("TRAINING_PREFIX", "train")

logging.info(f"Using region: {REGION}")
logging.info(f"Table name: {TABLE_NAME}")

# AWS clients
s3 = boto3.client('s3', region_name=REGION)
dynamodb = boto3.resource('dynamodb', region_name=REGION)
table = dynamodb.Table(TABLE_NAME)

# Convert DynamoDB items to DataFrame
def convert_items_to_dataframe(items):
    logging.info("Converting DynamoDB items to DataFrame...")

    def convert_value(value):
        if isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, list):
            return [convert_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: convert_value(v) for k, v in value.items()}
        else:
            return value

    cleaned = [{k: convert_value(v) for k, v in item.items()} for item in items]
    df = pd.DataFrame(cleaned)

    # Fix transactionid column
    if "transactionid" in df.columns:
        df["transactionid"] = df["transactionid"].astype(str).fillna("")

    logging.info(f"Converted to DataFrame with shape {df.shape}.")
    return df


# Save DataFrame to S3 as Parquet
def save_to_parquet(df, batch_index):
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    filename = f"{OUTPUT_PREFIX}/train_ready_batch_{batch_index}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.parquet"
    s3.upload_fileobj(buffer, BUCKET_NAME, filename)
    logging.info(f"Saved batch {batch_index} with shape {df.shape} to s3://{BUCKET_NAME}/{filename}")

# DynamoDB scan with batching logic
def scan_dynamodb_and_save_batches(batch_size=50000, scan_limit=2000):
    logging.info("Starting DynamoDB scan in batches...")
    buffer = []
    count = 0
    batch_index = 0
    last_evaluated_key = None

    while True:
        try:
            if last_evaluated_key:
                response = table.scan(Limit=scan_limit, ExclusiveStartKey=last_evaluated_key)
            else:
                response = table.scan(Limit=scan_limit)
        except EndpointConnectionError as e:
            logging.warning(f"Endpoint connection error: {e}. Retrying in 2s...")
            time.sleep(2)
            continue

        items = response.get("Items", [])
        buffer.extend(items)
        count += len(items)

        logging.info(f"Scanned total {count} items so far (buffered: {len(buffer)})")

        # When we accumulate a full batch
        if len(buffer) >= batch_size:
            df = convert_items_to_dataframe(buffer[:batch_size])
            save_to_parquet(df, batch_index)
            buffer = buffer[batch_size:]
            batch_index += 1

        last_evaluated_key = response.get("LastEvaluatedKey")
        if not last_evaluated_key:
            break

    # Save the remaining records
    if buffer:
        df = convert_items_to_dataframe(buffer)
        save_to_parquet(df, batch_index)
        logging.info("Final partial batch saved.")

    logging.info(f"Total items scanned: {count}, total batches saved: {batch_index + 1}")

# Main function
def build_training_dataset():
    try:
        scan_dynamodb_and_save_batches()
        logging.info("Training dataset build completed successfully.")
    except Exception as e:
        logging.error(f"Error during training dataset build: {e}")
        raise

if __name__ == "__main__":
    build_training_dataset()
