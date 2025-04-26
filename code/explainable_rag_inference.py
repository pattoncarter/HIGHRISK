import os
import json
import numpy as np
import pandas as pd
import faiss
import datetime
from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv
from subgraph_viz import visualize_subgraph
# Load environment variables
load_dotenv()

# Config
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_DIR = os.getenv("VECTOR_DIR", "vector_store")

# Setup
client = OpenAI()

def load_vector_store(vector_dir):
    index = faiss.read_index(os.path.join(vector_dir, "openai_embeddings.faiss"))
    metadata = pd.read_csv(os.path.join(vector_dir, "openai_embeddings_metadata.csv"))
    return index, metadata

def load_node_store(vector_dir):
    node_index = faiss.read_index(os.path.join(vector_dir, "node_embeddings.faiss"))
    node_metadata = pd.read_csv(os.path.join(vector_dir, "node_embeddings_metadata.csv"))
    return node_index, node_metadata

def embed_query(query, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=query, model=model)
    embedding = response.data[0].embedding
    return np.array(embedding).astype('float32')

def retrieve_documents(query_embedding, index, metadata, top_k=5):
    D, I = index.search(np.expand_dims(query_embedding, axis=0), top_k)
    results = metadata.iloc[I[0]].to_dict(orient="records")
    return results

def retrieve_knowledge_graph(query_embedding, node_index, node_metadata, driver, top_k_nodes=5, top_k_paths=10):
    # Step 1: Semantic node search
    D, I = node_index.search(np.expand_dims(query_embedding, axis=0), top_k_nodes)
    matched_nodes = node_metadata.iloc[I[0]]

    matched_element_ids = matched_nodes['element_id'].tolist()

    # Step 2: Expand 1-2 hop paths from matched nodes
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n)
            WHERE elementId(n) IN $node_ids
            MATCH path = (n)-[*1..2]-(m)
            RETURN path
            LIMIT $top_k_paths
            """,
            {"node_ids": matched_element_ids, "top_k_paths": top_k_paths}
        )

        paths = []
        for record in result:
            for rel in record["path"].relationships:
                paths.append({
                    "anchor": rel.start_node["name"],
                    "anchor_id": rel.start_node.element_id,
                    "neighbor": rel.end_node["name"],
                    "neighbor_id": rel.end_node.element_id
                })

        return paths

def generate_answer(query, docs, graph_paths):
    doc_context = ""
    doi_list = []
    for doc in docs:
        doc_context += f"- {doc['title']} (DOI: {doc['doi']})\n"
        doi_list.append(doc['doi'])

    kg_context = ""
    node_ids = set()
    for path in graph_paths:
        kg_context += f"- {path['anchor']} --> {path['neighbor']}\n"
        node_ids.add(path['anchor_id'])
        node_ids.add(path['neighbor_id'])

    prompt = f"""
You are a biomedical assistant. Based only on the following structured evidence (documents and knowledge graph paths), answer the clinical research question provided.

Documents:
{doc_context}

Knowledge Graph Relationships:
{kg_context}

Query: {query}

Format your answer as a professional clinical summary. Include citations for the sources you used (both DOIs and Node IDs).
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer_text = response.choices[0].message.content

    return answer_text, doi_list, list(node_ids)

def build_subgraph(node_ids, driver, output_path="subgraph.json"):
    from neo4j.graph import Node, Relationship

    with driver.session() as session:
        result = session.run(
            """
            MATCH (n)
            WHERE elementId(n) IN $node_ids
            MATCH path = (n)-[*1..2]-(m)
            RETURN path
            """,
            {"node_ids": node_ids}
        )

        nodes = {}
        relationships = []
        seen_relationships = set()  # Track already added edges

        for record in result:
            path = record["path"]
            for element in path.nodes:
                nodes[element.element_id] = {
                    "element_id": element.element_id,
                    "labels": list(element.labels),
                    "properties": dict(element.items())
                }
            for rel in path.relationships:
                source = rel.start_node.element_id
                target = rel.end_node.element_id
                rel_type = rel.type

                edge_key = (source, target, rel_type)  # Uniquely define edge by source-target-type

                if edge_key not in seen_relationships:
                    relationships.append({
                        "element_id": rel.element_id,
                        "type": rel_type,
                        "start_node_element_id": source,
                        "end_node_element_id": target,
                        "properties": dict(rel.items())
                    })
                    seen_relationships.add(edge_key)  # Mark as seen

        subgraph = {
            "nodes": list(nodes.values()),
            "relationships": relationships
        }

        with open(output_path, "w") as f:
            json.dump(subgraph, f, indent=2)

    print(f"Subgraph saved to {output_path}")

def main():
    query = input("Enter your medical research query: ")

    index, metadata = load_vector_store(VECTOR_DIR)
    node_index, node_metadata = load_node_store(VECTOR_DIR)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    query_embedding = embed_query(query)

    docs = retrieve_documents(query_embedding, index, metadata)
    graph_paths = retrieve_knowledge_graph(query_embedding, node_index, node_metadata, driver)

    answer, dois, node_ids = generate_answer(query, docs, graph_paths)

    print("\n\n===== Generated Answer =====\n")
    print(answer)
    print("\n===== Sources =====")
    print(f"DOIs Used: {dois}")
    print(f"Node IDs Used: {node_ids}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    subgraph_filename = f"subgraph_{timestamp}.json"
    build_subgraph(node_ids, driver, output_path=subgraph_filename)
    visualize_subgraph(subgraph_path=subgraph_filename)
    driver.close()

if __name__ == "__main__":
    main()
