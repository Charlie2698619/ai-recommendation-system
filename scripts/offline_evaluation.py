import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
import pickle
import logging

# --- Configuration ---
EVENTS_FILE = 'retailrocket_data/events.csv'
ITEM_EMBEDDINGS_FILE = 'ML/item_embeddings.npy'
FAISS_INDEX_FILE = 'ML/faiss_index.bin'
ITEM_ID_MAP_FILE = 'ML/itemid_map.pkl'  # Path to the item ID map
K = 10
TRAIN_SPLIT_RATIO = 0.8

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['itemid'] = df['itemid'].astype(str) 
    df = df.sort_values('timestamp')
    return df


def split_data(df, split_ratio):
    """Splits the data into training and testing sets based on time."""
    split_point = int(len(df) * split_ratio)
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]
    print(f"Data split: {len(train_df)} training events, {len(test_df)} testing events.")
    return train_df, test_df

def get_user_history(df):
    """Aggregates item interactions for each user."""
    return df.groupby('visitorid')['itemid'].apply(list)

def load_model_and_maps(embeddings_file, index_file, map_file):
    """Loads model artifacts and creates itemid-to-index mappings."""
    print("Loading model artifacts and ID maps...")
    item_embeddings = np.load(embeddings_file)
    faiss_index = faiss.read_index(index_file)
    
    with open(map_file, 'rb') as f:
        itemid_list = pickle.load(f)
        
    # Create the mappings
    itemid_to_index = {item_id: i for i, item_id in enumerate(itemid_list)}
    index_to_itemid = {i: item_id for i, item_id in enumerate(itemid_list)}
    
    print("Model and mappings loaded.")
    return item_embeddings, faiss_index, itemid_to_index, index_to_itemid

def generate_recommendations(user_history, item_embeddings, faiss_index, itemid_to_index, index_to_itemid, k):
    """Generates top K recommendations for a single user."""
    # Convert user's itemid history to embedding indices
    history_indices = [itemid_to_index[item_id] for item_id in user_history if item_id in itemid_to_index]
    
    if not history_indices:
        logging.debug(f"User skipped: no matching itemids in embeddings: {user_history}")
        return []  

    user_item_embeddings = item_embeddings[history_indices]
    user_embedding = np.mean(user_item_embeddings, axis=0).reshape(1, -1)
    
    faiss.normalize_L2(user_embedding) # Normalize the user vector
    
    distances, indices = faiss_index.search(user_embedding, k + len(history_indices)) # Fetch more to filter seen items
    
    # Convert result indices back to itemids
    recommended_item_ids = [index_to_itemid[i] for i in indices[0]]
    
    # Filter out items the user has already seen
    seen_items = set(user_history)
    final_recommendations = [rec for rec in recommended_item_ids if rec not in seen_items]
    
    return final_recommendations[:k]

def evaluate(train_history, test_history, item_embeddings, faiss_index, itemid_to_index, index_to_itemid, k):
    """Runs the full evaluation loop."""
    print(f"Starting evaluation for Precision@{k} and Recall@{k}...")
    
    all_precisions = []
    all_recalls = []
    skipped_users = 0
    
    common_users = set(train_history.index) & set(test_history.index)
    print(f"Found {len(common_users)} users present in both training and testing sets.")

    for user_id in tqdm(common_users):
        user_train_history = train_history[user_id]
        
        recommendations = generate_recommendations(user_train_history, item_embeddings, faiss_index, itemid_to_index, index_to_itemid, k)
        
        if not recommendations:
            skipped_users += 1
            continue

        ground_truth = set(test_history[user_id])
        hits = len(set(recommendations) & ground_truth)
        
        precision = hits / k if k > 0 else 0
        recall = hits / len(ground_truth) if len(ground_truth) > 0 else 0
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        
    avg_precision = np.mean(all_precisions) if all_precisions else 0
    avg_recall = np.mean(all_recalls) if all_recalls else 0
    logging.info(f"Skipped {skipped_users} users with no recommendations.")
    return avg_precision, avg_recall, skipped_users

def main():
    """Main function to run the offline evaluation."""
    df = load_data(EVENTS_FILE)
    train_df, test_df = split_data(df, TRAIN_SPLIT_RATIO)
    
    train_user_history = get_user_history(train_df)
    test_user_history = get_user_history(test_df)
    
    item_embeddings, faiss_index, itemid_to_index, index_to_itemid = load_model_and_maps(
        ITEM_EMBEDDINGS_FILE, FAISS_INDEX_FILE, ITEM_ID_MAP_FILE
    )
    
    avg_precision, avg_recall, skipped_users = evaluate(
        train_user_history, test_user_history, item_embeddings, faiss_index, itemid_to_index, index_to_itemid, K
    )
    
    print("\n--- Offline Evaluation Results ---")
    print(f"Precision@{K}: {avg_precision:.4f}")
    print(f"Recall@{K}:    {avg_recall:.4f}")
    print(f"Skipped Users: {skipped_users}")
    print("----------------------------------")

def run_offline_evaluation():
    """Main function to run the offline evaluation."""
    try:
        main()
    except Exception as e:
        logging.error(f"Error during offline evaluation: {e}")
        raise
    

