#!/bin/bash

# Set your bucket name (must be globally unique)
BUCKET_NAME="ecom-raw-events"

echo "Creating S3 bucket: $BUCKET_NAME in us-east-1..."

aws s3api create-bucket \
  --bucket "$BUCKET_NAME" \
  --region us-east-1 \
  --endpoint-url https://s3.amazonaws.com \
  --output json

echo "✅ Bucket created: $BUCKET_NAME"
