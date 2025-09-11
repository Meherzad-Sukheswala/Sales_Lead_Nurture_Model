import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from azure.cosmos import CosmosClient

# === Page Config ===
st.set_page_config(page_title="PTB Score Dashboard", layout="wide")
st.title("🧠 PTB Score Predictor with Azure Auto Sync")

# === Load Secrets from Streamlit Cloud ===
endpoint = st.secrets["COSMOS_ENDPOINT"]
key = st.secrets["COSMOS_KEY"]
database_name = st.secrets["DATABASE_NAME"]
container_name = st.secrets["INPUT_CONTAINER"]

# === Fetch Data from Cosmos DB ===
@st.cache_data
def fetch_data():
    try:
        client = CosmosClient(endpoint, credential=key)
        db = client.get_database_client(database_name)
        container = db.get_container_client(container_name)
        items = list(container.read_all_items())
        return pd.DataFrame(items)
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {e}")
        return pd.DataFrame()

# === Upload Data to Cosmos DB ===
def upload_to_cosmos(df):
    try:
        client = CosmosClient(endpoint, credential=key)
        db = client.get_database_client(database_name)
        container = db.get_container_client(container_name)
        for row in df.to_dict(orient="records"):
            container.upsert_item(row)
        st.success("✅ Data uploaded to Cosmos DB")
    except Exception as e:
        st.error(f"❌ Upload failed: {e}")

# === Load and Display Data ===
df = fetch_data()

if df.empty:
    st.warning("No data found in Cosmos DB.")
else:
    # === Tabs ===
    tab1, tab2 = st.tabs(["📋 Scored Results", "📊 Insights"])

    with tab1:
        st.subheader("✅ Scored Results")
        st.dataframe(df)

        st.download_button("⬇️ Download CSV", df.to_csv(index=False), "scored_results.csv")
        if st.button("🚀 Upload to Cosmos DB"):
            upload_to_cosmos(df)

    with tab2:
        st.subheader("📊 PTB Score Distribution")
        if "PTB_Score" in df.columns:
            fig1, ax1 = plt.subplots()
            df["PTB_Score"].value_counts().sort_index().plot(kind='bar', ax=ax1, color='orange')
            ax1.set_xlabel("PTB Score")
            ax1.set_ylabel("Number of Customers")
            st.pyplot(fig1)
        else:
            st.warning("⚠️ 'PTB_Score' column not found.")

        st.subheader("✅ Policy Purchase Outcomes")
        if "Policy Purchased" in df.columns:
            fig2, ax2 = plt.subplots()
            df["Policy Purchased"].value_counts().rename(index={0: "Not Purchased", 1: "Purchased"}).plot(kind='bar', ax=ax2, color='green')
            ax2.set_ylabel("Number of Customers")
            st.pyplot(fig2)
        else:
            st.warning("⚠️ 'Policy Purchased' column not found.")

        st.subheader("🏅 Customers by Lead Tier")
        if "Lead_Tier" in df.columns:
            fig3, ax3 = plt.subplots()
            df["Lead_Tier"].value_counts().plot(kind='bar', ax=ax3, color='purple')
            ax3.set_xlabel("Lead Tier")
            ax3.set_ylabel("Number of Customers")
            st.pyplot(fig3)
        else:
            st.warning("⚠️ 'Lead_Tier' column not found.")

df = fetch_data()

# ✅ Add this debug line immediately after:
st.write("🔍 Available columns in the dataset:", df.columns.tolist())

