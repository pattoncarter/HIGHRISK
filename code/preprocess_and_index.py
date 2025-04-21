import os
import json
import faiss
import numpy as np
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
VECTOR_DIR = "vector_store"
os.makedirs(VECTOR_DIR, exist_ok=True)

client = OpenAI()

# Load articles
def load_articles(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# Semantic chunking
def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ".", ","]
    )
    chunked_docs = []
    metadata = []

    for doc in docs:
        chunks = splitter.split_text(doc['abstract'])
        chunked_docs.extend(chunks)
        metadata.extend([{
            "pubmed_id": doc["pubmed_id"],
            "doi": doc.get("doi", "")
        }] * len(chunks))


    return chunked_docs, metadata

# Generate embeddings with OpenAI API
def generate_embeddings(chunks, model="text-embedding-3-small"):
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(input=chunk, model=model)
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings).astype('float32')

# Indexing using FAISS
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(VECTOR_DIR, "openai_embeddings.faiss"))
    return index

# Main execution
if __name__ == "__main__":
    articles = load_articles(os.path.join(DATA_DIR, "articles.json"))

    # Chunk documents semantically
    chunks, metadata = chunk_documents(articles)

    # Generate embeddings using OpenAI API
    embeddings = generate_embeddings(chunks)

    # Save metadata and chunks
    pd.DataFrame({
        "chunk_text": chunks,
        "pubmed_id": [m["pubmed_id"] for m in metadata],
        "doi": [m["doi"] for m in metadata]
    }).to_csv(os.path.join(DATA_DIR, "chunked_docs.csv"), index=False)


    # Create and save FAISS index
    create_faiss_index(embeddings)
    print("Indexing complete. FAISS index saved.")
