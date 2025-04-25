import os
import openai
import numpy as np
import pandas as pd
import faiss
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Config
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_DIR = os.getenv("VECTOR_DIR", "vector_store")

# Setup
openai.api_key = OPENAI_API_KEY

def load_vector_store(vector_dir):
    index = faiss.read_index(os.path.join(vector_dir, "openai_embeddings.faiss"))
    metadata = pd.read_csv(os.path.join(vector_dir, "metadata.csv"))
    return index, metadata

def embed_query(query, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=query, model=model)
    return np.array(response['data'][0]['embedding']).astype('float32')

def retrieve_documents(query_embedding, index, metadata, top_k=5):
    D, I = index.search(np.array([query_embedding]), top_k)
    results = metadata.iloc[I[0]].to_dict(orient="records")
    return results

def retrieve_knowledge_graph(query, driver, top_k=5):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE toLower(a.name) CONTAINS toLower($query) OR toLower(b.name) CONTAINS toLower($query)
            RETURN a.name AS head, type(r) AS relation, b.name AS tail
            LIMIT $top_k
            """,
            query=query,
            top_k=top_k
        )
        return [record.data() for record in result]

def generate_answer(query, docs, triples):
    context = "".join([f"- {doc['title']} (DOI: {doc['doi']})\n" for doc in docs])
    kg_context = "".join([f"- {triple['head']} {triple['relation']} {triple['tail']}\n" for triple in triples])

    prompt = f"""
You are a biomedical assistant. Use the following document summaries and structured knowledge triples to answer the user's query.

Documents:
{context}

Knowledge Graph Triples:
{kg_context}

Query: {query}

Answer in both a clinical professional format and a simplified patient-friendly format.
"""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response['choices'][0]['message']['content']

def main():
    query = input("Enter your medical research query: ")

    index, metadata = load_vector_store(VECTOR_DIR)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    query_embedding = embed_query(query)
    docs = retrieve_documents(query_embedding, index, metadata)
    triples = retrieve_knowledge_graph(query, driver)

    answer = generate_answer(query, docs, triples)
    print("\n\n===== Generated Answer =====\n")
    print(answer)

    driver.close()

if __name__ == "__main__":
    main()
