# 🤖 AI Recommendation System

This is a full-stack recommendation engine demo built using FAISS, FastAPI, DynamoDB, S3, and Streamlit.

## 🎯 Project Goals
- Deliver intelligent product recommendations to enhance user engagement and increase conversion.
- Handle both **cold-start (new user)** and **warm-start (known user)** scenarios.
- Showcase a **production-grade MLOps-ready pipeline** using AWS, Docker, and modern ML tools.

## 📊 Data Source
- This project uses open-source **RetailRocket** e-commerce datasets:

- [(https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)]

## 🧠 Machine Learning Strategy

| Use Case                     | Model Type                   | Inputs Used                                  |
|-----------------------------|------------------------------|----------------------------------------------|
| Recommend to returning user | Content-based + history avg  | User interaction history + item embeddings   |
| Recommend similar items     | Item-to-item content-based   | TF-IDF + numeric embeddings similarity       |

- **TF-IDF**: Vectorize all item text attributes.
- **MinMaxScaler**: Normalize numerical attributes.
- **PCA**: Reduce dimensions to improve FAISS performance.
- **FAISS**: Fast similarity search for embedding-based recommendations.


## 🔄 Workflow
1. Upload user events to S3 (data lake)
2. Store events in DynamoDB (data warehouse)
3. Build training dataset (Parquet)
4. Generate item embeddings
5. Train FAISS index and upload to S3
6. Launch API + Streamlit for recommendation


## 🧪 Accuracy Evaluation

This system uses offline evaluation for personalized recommendations.

📊 Evaluation Methodology

- Holdout last interaction per user (temporal split)
- Store training/test sets (earlier vs. recent interactions)
- Generate top-K recommendations for each user
- Calculate Precision@K and Recall@K metrics
- Filter valid users (present in both train/test)
- Automate evaluation pipeline for reproducibility

📈 Key Metrics

- Precision@K: Proportion of recommended items in top K that are relevant
- Recall@K: Proportion of relevant items captured in top K recommendations
- User Coverage: Percentage of users with valid recommendations



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

<img width="807" height="670" alt="image" src="https://github.com/user-attachments/assets/1289e798-8337-4b9f-b6b2-183c4f0e5057" />


