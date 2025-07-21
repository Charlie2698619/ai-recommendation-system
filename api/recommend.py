from fastapi import FastAPI, HTTPException
from typing import List
import logging
from ML import query_faiss 
import os
from pydantic import BaseModel
import boto3
import numpy as np
from boto3.dynamodb.conditions import Key


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
app = FastAPI(title="AI Recommendation System", version="1.0")

dynamodb = boto3.resource("dynamodb", region_name=os.getenv("AWS_DEFAULT_REGION"))
interaction_table = dynamodb.Table(os.getenv("DYNAMODB_TABLE"))

TOP_K = int(os.getenv("TOP_K", 5))
faiss_index = None
itemid_to_index = {}
index_to_itemid = {}

@app.on_event("startup")
def startup_event():
    global faiss_index, itemid_to_index, index_to_itemid 
    faiss_index = query_faiss.load_faiss_index()
    maps = query_faiss.load_itemid_map()
    itemid_to_index = maps["itemid_to_index"]
    index_to_itemid = maps["index_to_itemid"]
    logging.info("FAISS index and map loaded successfully.")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/recommend_user/{user_id}", response_model=List[str])
def recommend_for_user(user_id: str, k: int = TOP_K):
    try:
        # Get user interaction history (latest 100 interactions)
        response = interaction_table.query(
        IndexName="user_id-index",
        KeyConditionExpression=Key("user_id").eq(user_id),
        Limit=100,
        ScanIndexForward=False  # Recent first
        )
        items = response.get("Items", [])
        if not items:
            raise HTTPException(status_code=404, detail="No interactions found for this user")

        # Get valid itemids the user has interacted with
        item_ids = [str(item["itemid"]) for item in items if str(item["itemid"]) in itemid_to_index]
        if not item_ids:
            raise HTTPException(status_code=404, detail="No valid item embeddings for this user")

        # Get vectors of those items and average them
        user_vectors = [faiss_index.reconstruct(itemid_to_index[item]) for item in item_ids]
        user_vector = np.mean(user_vectors, axis=0).reshape(1, -1)

        # Query FAISS with user vector
        scores, indices = faiss_index.search(user_vector, k + len(item_ids))

        # Filter out previously seen items
        recommendations = [index_to_itemid[i] for i in indices[0] if index_to_itemid[i] not in item_ids]
        logging.info(f"Recommended for user {user_id}: {recommendations[:k]}")
        return recommendations[:k]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))