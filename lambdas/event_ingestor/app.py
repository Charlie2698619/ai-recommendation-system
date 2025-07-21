import json
import os
import boto3
from datetime import datetime


dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['DYNAMODB_TABLE'])


def lambda_handler(event, context):
    try: 
        print("Received event:", event)
        if 'body' not in event or not event['body']:
            raise ValueError("Event body is missing or empty")
        
        body = json.loads(event['body'])
        print("Parsed body:", body)  
        
        user_id = body.get('user_id')
        item_id = body.get('item_id')
        event_type = body.get('event')
        property = body.get('property', None)
        value = body.get('value', None)
        event_timestamp = body.get('event_timestamp', datetime.utcnow().isoformat())
        item_timestamp = body.get('item_timestamp', None)
        
        item = {
            'user_id': user_id,
            'item_id': item_id,
            'event': event_type,
            'property': property,
            'value': value,
            'event_timestamp': event_timestamp,
            'item_timestamp': item_timestamp
        }
        
        # Store the event in DynamoDB
        table.put_item(Item=item) # upsert pattern if the item already exists they will be overwritten
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Event stored successfully'})
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
        