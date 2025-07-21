# 🤖 AI Recommendation System

This is a full-stack recommendation engine demo built using FAISS, FastAPI, DynamoDB, S3, and Streamlit.

## 🎯 Project Goals
- Recommend items to returning users (based on history)
- Recommend similar items for anonymous users

## 🧠 ML Pipeline
- TF-IDF + MinMaxScaler + PCA for embedding generation
- FAISS for Approximate Nearest Neighbor (ANN) search
- Streamlit UI to demo results

## 📊 Tech Stack
| Layer        | Technology         |
|--------------|--------------------|
| Backend      | FastAPI, boto3     |
| Frontend     | Streamlit          |
| Embedding    | TF-IDF + PCA       |
| Indexing     | FAISS              |
| Storage      | S3 + DynamoDB      |
| Container    | Docker + Compose   |

## 🔄 Workflow
1. Upload user events to S3 (data lake)
2. Store events in DynamoDB (data warehouse)
3. Build training dataset (Parquet)
4. Generate item embeddings
5. Train FAISS index and upload to S3
6. Launch API + Streamlit for recommendation

## 🧪 How to Run
### 1. Clone the repo
```bash
git clone https://github.com/yourusername/ai-recommendation-system.git
cd ai-recommendation-system
````

### 2. Set up environment

Create `.env` from template:

```bash
cp .env.example .env
```

### 3. Build and launch

```bash
docker-compose up --build
```

### 4. Access:

* FastAPI: [http://localhost:8080/docs](http://localhost:8080/docs)
* Streamlit: [http://localhost:8501](http://localhost:8501)

## 📈 Example Use Cases

* `GET /recommend_user/{user_id}`
* `GET /recommend/{item_id}`

## 📷 Screenshots

*Add screenshots of Streamlit app and FastAPI Swagger UI.*


