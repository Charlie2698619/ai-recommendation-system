#!/bin/bash

TABLE_NAME="user_interactions"

echo "Creating DynamoDB table: $TABLE_NAME..."

aws dynamodb create-table \
  --table-name "$TABLE_NAME" \
  --attribute-definitions AttributeName=user_id,AttributeType=S \
  --key-schema AttributeName=user_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1 \
  --output json

echo "✅ DynamoDB table created: $TABLE_NAME"
