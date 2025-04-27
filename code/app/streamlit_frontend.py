import streamlit as st
import os
import json
from explainable_rag_inference import run_inference  

st.set_page_config(page_title="Explainable HybridRAG Oncology Assistant", layout="wide")
st.title("Explainable HybridRAG Oncology Research Assistant")

# Session state setup
if "query_submitted" not in st.session_state:
    st.session_state.query_submitted = False

with st.form(key="query_form"):
    query = st.text_input("Enter your medical research query:")
    submit_button = st.form_submit_button("Submit")
    new_query_button = st.form_submit_button("New Query")

# Handle form actions
if submit_button and query:
    with st.spinner('Thinking...'):
        answer, dois, node_ids, subgraph_path, viz_html_path = run_inference(query)
        
        st.session_state.answer = answer
        st.session_state.dois = dois
        st.session_state.node_ids = node_ids
        st.session_state.subgraph_path = subgraph_path
        st.session_state.viz_html_path = viz_html_path
        st.session_state.query_submitted = True

if new_query_button:
    st.session_state.clear()
    st.stop()

# Show outputs if a query was submitted
if st.session_state.get("query_submitted"):
    st.subheader("Answer:")
    st.write(st.session_state.answer)

    st.subheader("Sources:")
    st.markdown("- **DOIs Used:**")
    for doi in st.session_state.dois:
        st.markdown(f"  - [{doi}](https://doi.org/{doi})")

    st.markdown("- **Node IDs Used:**")
    for node_id in st.session_state.node_ids:
        st.markdown(f"  - `{node_id}`")

    st.subheader("Knowledge Graph Subgraph:")
    with open(st.session_state.viz_html_path, "r", encoding="utf-8") as f:
        graph_html = f.read()
    st.components.v1.html(graph_html, height=800, scrolling=True)

    st.subheader("Export Results:")
    export_data = {
        "query": query,
        "answer": st.session_state.answer,
        "sources": {
            "dois": st.session_state.dois,
            "node_ids": st.session_state.node_ids
        },
        "subgraph_file": st.session_state.subgraph_path
    }
    export_json = json.dumps(export_data, indent=2)

    st.download_button(
        label="Download Results as JSON",
        data=export_json,
        file_name="rag_query_results.json",
        mime="application/json"
    )
