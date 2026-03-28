import streamlit as st
import requests

# --- CONFIGURATION ---
# This points the UI to your FastAPI backend
API_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Agentic Equity Research", page_icon="🏦", layout="wide")

st.title("🏦 Agentic Equity Research Command Center")
st.markdown("Deploy multi-agent AI squads to analyze equities and search historical semantic vaults.")

tab1, tab2 = st.tabs(["🚀 Deploy New Research Squad", "🔍 Search Semantic Vault"])

with tab1:
    st.header("Initiate AI Equity Research")
    col1, col2 = st.columns(2)
    with col1:
        company_name = st.text_input("Company Name (e.g., Tesla, Nvidia, Apple)")
    with col2:
        ticker = st.text_input("Stock Ticker (e.g., TSLA, NVDA, AAPL)").upper()

    if st.button("Deploy AI Agents", type="primary"):
        if company_name and ticker:
            # The UI shows a spinner while the FastAPI backend runs CrewAI
            with st.spinner(f"Agents are actively researching {company_name}... This takes 30-60 seconds."):
                try:
                    payload = {"company_name": company_name, "ticker": ticker}
                    # Send a POST request to your FastAPI server
                    response = requests.post(f"{API_URL}/research", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success("Research Complete & Archived to Vault!")
                        
                        # Display the JSON data in a beautiful dashboard layout
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Ticker", data["ticker"])
                        
                        rec = data["recommendation"]
                        rec_color = "green" if rec == "BUY" else "red" if rec == "SELL" else "orange"
                        col_b.markdown(f"### Recommendation: <span style='color:{rec_color}'>{rec}</span>", unsafe_allow_html=True)
                        col_c.metric("Risk Level", data["risk_level"])
                        
                        st.subheader("CIO Executive Summary")
                        st.write(data["executive_summary"])
                    else:
                        st.error(f"API Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("🚨 Connection Error: Is your FastAPI server running on port 8000?")
        else:
            st.warning("Please enter both Company Name and Ticker.")

# --- TAB 2: VECTOR DB SEARCH ---
with tab2:
    st.header("Search Historical AI Reports (ChromaDB)")
    query = st.text_input("Ask a natural language question (e.g., 'Which companies have high risk?', 'AI chip manufacturers')")
    limit = st.slider("Number of results to retrieve", 1, 5, 2)
    
    if st.button("Search Vector Vault"):
        if query:
            with st.spinner("Searching semantic embeddings..."):
                try:
                    # Send a GET request to your FastAPI search endpoint
                    response = requests.get(f"{API_URL}/search_vault", params={"query": query, "limit": limit})
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", {})
                        
                        documents = results.get("documents", [[]])[0]
                        metadatas = results.get("metadatas", [[]])[0]
                        
                        if not documents:
                            st.info("No matching reports found in the vault.")
                        else:
                            for doc, meta in zip(documents, metadatas):
                                # Display historical reports in collapsible boxes
                                with st.expander(f"Vault Entry: {meta.get('ticker', 'Unknown')} | Rec: {meta.get('recommendation', 'N/A')}"):
                                    st.markdown(f"**Risk Level:** {meta.get('risk_level', 'N/A')}")
                                    st.write(doc)
                    else:
                        st.error(f"API Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("🚨 Connection Error: Is your FastAPI server running on port 8000?")