import pandas as pd
import requests
import time
import logging
import os
from datetime import datetime
import math
import boto3
import json
import uuid

# Configuration
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "ecom-raw-events")
EXPORT_PREFIX = os.getenv("EXPORT_PREFIX", "batches")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 2000))

# AWS Clients
s3 = boto3.client("s3", region_name=AWS_REGION)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_item_features():
    """Loads and processes item properties into a feature DataFrame."""
    item_properties_part1 = pd.read_csv("retailrocket_data/item_properties_part1.csv")
    item_properties_part2 = pd.read_csv("retailrocket_data/item_properties_part2.csv")
    
    item_properties = pd.concat([item_properties_part1, item_properties_part2], ignore_index=True)
    item_properties = item_properties.sort_values(by="timestamp").drop_duplicates(subset=["itemid", "property"], keep="last")
    item_properties = item_properties.rename(columns={"timestamp": "item_timestamp"})
    
    item_features = item_properties.pivot(index="itemid", columns="property", values="value").reset_index()
    item_features.columns.name = None
    return item_features

def safe_timestamp(ts):
    try:
        ts = float(ts)
        if ts < 0 or ts > 4102444800:  # Year 2100-ish
            return "INVALID_TIMESTAMP"
        return datetime.fromtimestamp(ts).isoformat()
    except (ValueError, TypeError):
        return "INVALID_TIMESTAMP"


def to_event(row):
    """Converts a DataFrame row to a JSON event."""
    event = {
        "user_id": str(row["visitorid"]),
        "item_id": str(row["itemid"]),
        "event": row["event"],
        "property": row.get("property", None),
        "value": row.get("value", None),
        "event_timestamp": safe_timestamp(row["event_timestamp"]),
        "item_timestamp": safe_timestamp(row.get("item_timestamp", 0)) if "item_timestamp" in row else None
    }
    # Handle additional columns dynamically
    for col in row.index: 
        if col not in event:  
            val = row[col] 
            if isinstance(val, float) and not isinstance(val, str): 
                if math.isnan(val):
                    val = "nNaN"
                elif math.isinf(val):
                    val = "nInf" if val > 0 else "n-Inf"
            event[col] = val 
    return event

def save_batch_to_s3(batch_data):
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"batch_{timestamp}_{uuid.uuid4().hex}.json"
    file_path = f"/tmp/{filename}"

    with open(file_path, "w") as f:
        json.dump(batch_data, f, indent=2)

    s3_key = f"{EXPORT_PREFIX}/{filename}"
    s3.upload_file(file_path, S3_BUCKET, s3_key)
    logging.info(f"Uploaded batch of {len(batch_data)} events to s3://{S3_BUCKET}/{s3_key}")

def stream_events_to_s3(item_features):
    event_iterator = pd.read_csv("retailrocket_data/events.csv", chunksize=CHUNK_SIZE)
    for chunk in event_iterator:
        chunk = chunk.rename(columns={"timestamp": "event_timestamp"})
        chunk['event_timestamp'] = chunk["event_timestamp"] / 1000
        merged_chunk = chunk.merge(item_features, how="left", on="itemid")

        events = [to_event(row) for _, row in merged_chunk.iterrows()]
        save_batch_to_s3(events)
        logging.info(f"Processed and saved chunk of {len(events)} events.")
        
        
def main():
    logging.info("Starting batch event processing to S3")
    start_time = time.time()
    try:
        item_features = get_item_features()
        stream_events_to_s3(item_features)
        logging.info("All chunks processed and saved to S3.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    elapsed_time = time.time() - start_time
    logging.info(f"Completed in {elapsed_time:.2f} seconds")

def simulate_events():
    main()

if __name__ == "__main__":
    simulate_events()

    
    
    
    
        