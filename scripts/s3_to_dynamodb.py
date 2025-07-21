import os
import json
import logging
from datetime import datetime
import boto3
from decimal import Decimal
import uuid

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

REGION = os.getenv("AWS_REGION", "us-east-1")
TABLE_NAME = os.getenv("DYNAMODB_TABLE", "user_interactions")
S3_BUCKET = os.getenv("S3_BUCKET", "ecom-raw-events")
IMPORT_PREFIX = os.getenv("IMPORT_PREFIX", "batches")

# Init AWS clients
dynamodb = boto3.resource('dynamodb', region_name=REGION)
s3 = boto3.client('s3', region_name=REGION)
table = dynamodb.Table(TABLE_NAME)

def parse_json_number(value):
    if isinstance(value, float) or isinstance(value, int):
        return Decimal(str(value))
    elif isinstance(value, dict):
        return {k: parse_json_number(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [parse_json_number(v) for v in value]
    else:
        return value

def list_s3_batches():
    logging.info(f"Listing batches in S3 bucket '{S3_BUCKET}' with prefix '{IMPORT_PREFIX}'")
    objects = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=IMPORT_PREFIX)
    return [obj['Key'] for obj in objects.get('Contents', []) if obj['Key'].endswith('.json')]

def load_batch_from_s3(s3_key):
    logging.info(f"Loading batch from S3: {s3_key}")
    response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    data = response['Body'].read().decode('utf-8')
    return json.loads(data, parse_float=Decimal)

def write_to_dynamodb(events):
    logging.info(f"Writing {len(events)} events to DynamoDB table '{TABLE_NAME}'")
    with table.batch_writer() as batch:
        for item in events:
            try:
                item = {k: parse_json_number(v) for k, v in item.items()}

                # Ensure user_id exists
                if "user_id" not in item:
                    logging.warning(f"Missing user_id, skipping item: {item}")
                    continue

                item['user_id'] = str(item['user_id'])         # required for GSI
                item['event_id'] = str(uuid.uuid4())           # required as PK

                # Optionally ensure timestamp is string or datetime
                if "event_timestamp" in item:
                    item["event_timestamp"] = str(item["event_timestamp"])

                batch.put_item(Item=item)

            except Exception as e:
                logging.warning(f"Failed to write item: {item.get('user_id', 'UNKNOWN')}, error: {e}")
    logging.info(f"Successfully wrote {len(events)} events to DynamoDB.")

def s3_to_dynamodb():
    logging.info("Starting S3 to DynamoDB import")
    start_time = datetime.now()
    
    try:
        s3_batches = list_s3_batches()
        if not s3_batches:
            logging.warning("No batches found in S3.")
            return
        
        for s3_key in s3_batches:
            events = load_batch_from_s3(s3_key)
            write_to_dynamodb(events)
        
        duration = (datetime.now() - start_time).total_seconds()
        logging.info(f"Import completed successfully in {duration:.2f} seconds.")
        
    except Exception as e:
        logging.error(f"Error during import: {e}", exc_info=True)

if __name__ == "__main__":
    s3_to_dynamodb()
