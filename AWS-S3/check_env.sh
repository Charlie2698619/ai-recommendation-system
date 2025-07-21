#!/bin/bash

echo "🔍 Checking AWS CLI version..."
aws --version

echo "🔍 Checking current AWS profile..."
aws configure list

echo "🔍 Listing S3 buckets..."
aws s3 ls

echo "🔍 Listing DynamoDB tables..."
aws dynamodb list-tables --region us-east-1
