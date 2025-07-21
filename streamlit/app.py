import streamlit as st
import requests
import os


API_URL = os.getenv("STREAMLIT_API_URL", "http://ai-recommendation-system:8080")  # Update if deployed


st.set_page_config(page_title="AI Recommendation System Demo", page_icon="🤖", layout="centered")

st.title("🔍 AI Recommendation System Demo")

# Project Overview
with st.expander("📘 Project Overview & ML Strategy"):
    st.markdown("""
    ### 🎯 Objective
    This AI Recommendation System is designed to deliver personalized product suggestions for:
    - 🧑‍💼 **Returning users** based on their interaction history
    - 🆕 **New or anonymous visitors** based on item similarity

    ### 🧠 Machine Learning Approach
    - **Text + Numeric Embeddings**: Items are represented using a mix of text fields (via TF-IDF) and numeric attributes (via MinMaxScaler).
    - **Dimensionality Reduction**: PCA compresses high-dimensional vectors into compact embeddings.
    - **ANN Search**: FAISS is used to index and retrieve the most similar items efficiently.

    ### 🧩 Models in Action
    | Use Case | Model Target | Data Used |
    |----------|--------------|------------|
    | Recommend to returning user | Content-based + history | User interaction history + item embeddings |
    | Recommend similar items | Item-to-item content-based | Item embedding similarity |

    ### 🔄 Workflow
    1. User/item interaction logs are stored in **DynamoDB**
    2. Training batches are created and stored in **S3**
    3. Embeddings are generated and stored using **TF-IDF + PCA**
    4. FAISS index is built on embeddings
    5. Streamlit + FastAPI serve the results via recommendation APIs

    This app demonstrates a full-stack ML system that supports **real-time recommendation** from cloud-stored models.
    """)




# Recommendation Option
option = st.radio("Recommend by:", ["User ID", "Item ID"])

if option == "User ID":
    user_id = st.text_input("Enter a User ID:").strip()
    if st.button("Recommend", key="user_recommend") and user_id:
        with st.spinner("Getting recommendations..."):
            try:
                res = requests.get(f"{API_URL}/recommend_user/{user_id}", timeout=8)
                if res.status_code == 200:
                    recs = res.json() if isinstance(res.json(), list) else res.json().get("recommendations", [])
                    if recs:
                        st.success("🔁 Recommended Items:")
                        for i, item in enumerate(recs, 1):
                            st.write(f"{i}. **Item ID:** {item}")
                    else:
                        st.warning("No recommendations found.")
                else:
                    st.error(f"API error: {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f"API request failed: {e}")

elif option == "Item ID":
    item_id = st.text_input("Enter an Item ID:").strip()
    if st.button("Recommend", key="item_recommend") and item_id:
        with st.spinner("Finding similar items..."):
            try:
                res = requests.get(f"{API_URL}/recommend/{item_id}", timeout=8)
                if res.status_code == 200:
                    recs = res.json() if isinstance(res.json(), list) else res.json().get("similar_items", [])
                    if recs:
                        st.success("🔗 Similar Items:")
                        for i, item in enumerate(recs, 1):
                            st.write(f"{i}. **Item ID:** {item}")
                    else:
                        st.warning("No similar items found.")
                else:
                    st.error(f"API error: {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f"API request failed: {e}")

# Footer
st.markdown("---")
st.caption("Demo powered by your AI Recommendation System – built by [your project/team name].")