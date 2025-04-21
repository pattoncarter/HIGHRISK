import faiss
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Paths
VECTOR_DIR = "vector_store"
DATA_DIR = "data"

client = OpenAI()

# Load FAISS Index
faiss_index = faiss.read_index(os.path.join(VECTOR_DIR, "openai_embeddings.faiss"))

# Load Chunk Data
chunk_df = pd.read_csv(os.path.join(DATA_DIR, "chunked_docs.csv"))

# Generate embeddings directly from OpenAI
def embed_query(query, model="text-embedding-3-small"):
    response = client.embeddings.create(input=query, model=model)
    return np.array(response.data[0].embedding).astype('float32').reshape(1, -1)

# RAG query function
def query_rag(user_query, top_k=3):
    # 1. Embed the user's query
    query_vector = embed_query(user_query)

    # 2. Retrieve top-K similar chunks
    distances, indices = faiss_index.search(query_vector, top_k)
    retrieved_chunks = chunk_df.iloc[indices[0]]["chunk_text"].tolist()
    retrieved_dois = chunk_df.iloc[indices[0]]["doi"].tolist()

    # 3. Construct the RAG prompt
    prompt = construct_prompt(user_query, retrieved_chunks)

    # 4. Generate LLM response
    response = generate_completion(prompt)

    return response, retrieved_chunks, retrieved_dois

# Helper: Prompt construction
def construct_prompt(query, context_chunks):
    context_text = "\n---\n".join(context_chunks)
    prompt = f"""
    You are an AI assistant summarizing the latest medical research for a healthcare professional.
    Use the context provided to answer the query clearly and concisely.

    Context:
    {context_text}

    Query:
    {query}

    Answer:
    """
    return prompt.strip()

# Helper: LLM completion call
def generate_completion(prompt, model="gpt-4-turbo", max_tokens=300):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful medical research assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return completion.choices[0].message.content.strip()

# Example execution
if __name__ == "__main__":
    query = "What are recent findings on beta-blockers in heart failure management?"
    summary, context_used, context_dois = query_rag(query)

    print("🩺 Generated Summary:")
    print(summary)
    print("\n🔖 Context used:")
    for idx, (chunk, doi) in enumerate(zip(context_used, context_dois), 1):
        print(f"Chunk {idx}: {chunk[:200]}...")  # preview first 200 chars
        print(f"DOI: {doi}")
