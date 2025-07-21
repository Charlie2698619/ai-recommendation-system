
import pandas as pd
import numpy as np
import faiss
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import logging

# --- Configuration ---
DATA_DIR = 'retailrocket_data'
EVENTS_FILE = os.path.join(DATA_DIR, 'events.csv')
ITEM_PROPS_PART1_FILE = os.path.join(DATA_DIR, 'item_properties_part1.csv')
ITEM_PROPS_PART2_FILE = os.path.join(DATA_DIR, 'item_properties_part2.csv')
CATEGORY_TREE_FILE = os.path.join(DATA_DIR, 'category_tree.csv')

OUTPUT_DIR = 'ML'
ITEM_EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, 'item_embeddings.npy')
FAISS_INDEX_FILE = os.path.join(OUTPUT_DIR, 'faiss_index.bin')
ITEM_ID_MAP_FILE = os.path.join(OUTPUT_DIR, 'itemid_map.pkl')

# Hyperparameters from your original script
TFIDF_MAX_FEATURES = 100
PCA_COMPONENTS = 64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_combine_data():
    logging.info("Loading datasets...")
    events = pd.read_csv(EVENTS_FILE)
    props1 = pd.read_csv(ITEM_PROPS_PART1_FILE)
    props2 = pd.read_csv(ITEM_PROPS_PART2_FILE)
    
    # Combine both parts
    item_props = pd.concat([props1, props2])
    
    logging.info("Converting item properties to key:value format...")
    item_props['kv'] = item_props['property'] + ":" + item_props['value'].astype(str)
    
    logging.info("Aggregating properties by item...")
    item_features = item_props.groupby('itemid')['kv'].apply(lambda x: ' '.join(x)).reset_index()
    
    # Get all item IDs (seen in both events and properties)
    all_item_ids = pd.concat([
        events['itemid'],
        item_props['itemid']
    ]).unique()
    df = pd.DataFrame(all_item_ids, columns=['itemid'])
    
    logging.info("Merging properties with item list...")
    df = pd.merge(df, item_features, on='itemid', how='left')
    df['kv'] = df['kv'].fillna("unknown")
    
    logging.info(f"Created text-based features for {len(df)} unique items.")
    return df.rename(columns={"kv": "combined_text"})


def preprocess_features(df):
    df['itemid'] = df['itemid'].astype(str)
    return df, df['combined_text']


def generate_embeddings(text_series):
    """Generates embeddings from text features using TF-IDF and PCA."""
    logging.info("Generating embeddings...")
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    tfidf_matrix = tfidf.fit_transform(text_series).toarray()
    
    if tfidf_matrix.shape[1] < PCA_COMPONENTS:
        logging.warning(f"TF-IDF returned {tfidf_matrix.shape[1]} dims < PCA_COMPONENTS={PCA_COMPONENTS}. Skipping PCA.")
        reduced_matrix = tfidf_matrix
    else:
        pca = PCA(n_components=PCA_COMPONENTS)
        reduced_matrix = pca.fit_transform(tfidf_matrix)

        
    logging.info(f"Embedding matrix shape: {reduced_matrix.shape}")
    return reduced_matrix.astype(np.float32)

def build_and_save_faiss_index(vectors, item_ids):
    """Builds a FAISS index, normalizes vectors, and saves artifacts."""
    logging.info("Normalizing vectors...")
    faiss.normalize_L2(vectors)
    
    logging.info("Building FAISS index...")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    
    logging.info(f"Saving FAISS index to {FAISS_INDEX_FILE}")
    faiss.write_index(index, FAISS_INDEX_FILE)
    
    logging.info(f"Saving item ID map to {ITEM_ID_MAP_FILE}")
    with open(ITEM_ID_MAP_FILE, 'wb') as f:
        pickle.dump(item_ids, f)
        
    logging.info(f"Saving item embeddings to {ITEM_EMBEDDINGS_FILE}")
    np.save(ITEM_EMBEDDINGS_FILE, vectors)


def main():
    """Main function to generate all necessary evaluation files."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    logging.info("--- Starting Data Preparation for Offline Evaluation ---")
    
    # 1. Load and process data
    item_df = load_and_combine_data()
    
    # 2. Preprocess features
    item_df, text_series = preprocess_features(item_df)
    
    # 3. Generate embeddings
    embeddings = generate_embeddings(text_series)
    
    # 4. Build and save FAISS index and mappings
    item_ids_list = item_df['itemid'].tolist()
    build_and_save_faiss_index(embeddings, item_ids_list)
    
    logging.info("--- All evaluation files have been successfully generated. ---")
    logging.info(f"You can now run the offline evaluation script.")

def prepare_evaluation_data():
    """Main function to prepare evaluation data."""
    try:
        main()
    except Exception as e:
        logging.error(f"Error during evaluation data preparation: {e}")
        raise
