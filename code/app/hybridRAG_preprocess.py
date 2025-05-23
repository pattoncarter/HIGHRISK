import json
from neo4j import GraphDatabase
import openai
import re
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import faiss
# Load environment variables
load_dotenv()

# Configuration (set these in .env file)
NEO4J_URI = os.getenv("NEO4J_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ARTICLES_FILE = os.getenv("ARTICLES_FILE", "data/oncology_articles_100.json")
BACKUP_FILE = os.getenv("BACKUP_FILE", "data/triples_backup.jsonl")
VECTOR_DIR = os.getenv("VECTOR_DIR", "vector_store")

#print(NEO4J_URI, NEO4J_PASSWORD, NEO4J_USER)

# Setup
openai.api_key = OPENAI_API_KEY
os.makedirs(os.path.dirname(BACKUP_FILE), exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)
client = OpenAI()

def load_articles(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# def extract_triples(abstract, title=""):
#     system_prompt = (
#         "You are a biomedical information extractor specialized in creating structured knowledge graphs from scientific abstracts. "
#         "Your goal is to extract factual knowledge and represent it as triples in the following structured format: [ENTITY1, RELATIONSHIP, ENTITY2]. "
#         "\n\nExtraction Rules:\n"
#         "- Only extract factual, explicit relationships stated or strongly implied in the abstract.\n"
#         "- Ignore uncertain or speculative language (e.g., 'might', 'could', 'possibly').\n"
#         "- Ignore background information without direct entity relationships.\n"
#         "- Extract multiple relationships separately if needed.\n"
#         "- No summarization, paraphrasing, or extra explanation.\n"
#         "\nOutput Format:\n"
#         "- One triple per line.\n"
#         "- Format each triple exactly like: [ENTITY1, RELATIONSHIP, ENTITY2]\n"
#         "- Sentence Case for entities, lowercase for relationships.\n"
#         "\nExample:\n"
#         "Given Abstract: 'Nivolumab has shown efficacy in treating melanoma.'\n"
#         "Output:\n"
#         "[Nivolumab, treats, melanoma]\n"
#     )
#     user_prompt = f"Title: {title}\nAbstract: {abstract}\n\nTriples:"

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4.1-mini",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             temperature=0.2,
#         )
#         triples_text = response.choices[0].message.content
#         triples = re.findall(r'\[([^\[\]]+?)\]', triples_text)
#         triples = [tuple(map(str.strip, triple.split(','))) for triple in triples]
#         return triples
#     except Exception as e:
#         print(f"Error processing abstract: {e}")
#         return []

def extract_triples(abstract, title=""):
    system_prompt = (
        "You are a biomedical information extractor specialized in creating structured knowledge graphs from scientific abstracts."
        "Your goal is to extract factual knowledge and represent it as triples in the following structured format: [ENTITY1, RELATIONSHIP, ENTITY2]. "
        "\n\nExtraction Rules:\n"
        "- Only extract factual, explicit relationships stated or strongly implied in the abstract.\n"
        "- Ignore uncertain or speculative language (e.g., 'might', 'could', 'possibly').\n"
        "- Ignore background information without direct entity relationships.\n"
        "- Extract multiple relationships separately if needed.\n"
        "- No summarization, paraphrasing, or extra explanation.\n"
        "\nOutput Format:\n"
        "- One triple per line.\n"
        "- Format each triple exactly like: [ENTITY1, RELATIONSHIP, ENTITY2]\n"
        "- Sentence Case for entities, lowercase for relationships.\n"
        "\nExample:\n"
        "Given Abstract: 'Nivolumab has shown efficacy in treating melanoma.'\n"
        "Output:\n"
        "[Nivolumab, treats, melanoma]\n"
     )
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract all factual biomedical triplets from this abstract:\n\nTitle: {title}\nAbstract: {abstract}"}
            ],
            functions=[
                {
                    "name": "extract_triplets",
                    "description": "Extract factual biomedical triplets from the input abstract.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "triplets": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "entity1": {"type": "string"},
                                        "relationship": {"type": "string"},
                                        "entity2": {"type": "string"}
                                    },
                                    "required": ["entity1", "relationship", "entity2"]
                                }
                            }
                        },
                        "required": ["triplets"]
                    }
                }
            ],
            function_call={"name": "extract_triplets"},
            temperature=0.2,
        )

        # Parse the structured JSON response
        arguments = response.choices[0].message.function_call.arguments
        arguments_json = json.loads(arguments)
        triples = arguments_json.get("triplets", [])

        return [(triplet["entity1"], triplet["relationship"], triplet["entity2"]) for triplet in triples]

    except Exception as e:
        print(f"Error processing abstract: {e}")
        return []


def save_triples_to_backup(triples, filepath):
    with open(filepath, "a") as f:
        for triple in triples:
            json.dump({"head": triple[0], "relation": triple[1], "tail": triple[2]}, f)
            f.write("\n")

def create_knowledge_graph(triples, driver):
    with driver.session() as session:
        for triple in triples:
            if len(triple) == 3:  # Only proceed if exactly 3 elements
                head, relation, tail = triple
                session.run(
                    "MERGE (e1:Entity {name: $head}) "
                    "MERGE (e2:Entity {name: $tail}) "
                    "MERGE (e1)-[:RELATION {type: $relation}]->(e2)",
                    head=head, tail=tail, relation=relation
                )
            else:
                print(f"Skipping malformed triple: {triple}")

def embed_texts(texts, model="text-embedding-ada-002"):
    embeddings = []
    for text in tqdm(texts, desc="Embedding texts", dynamic_ncols=True):
        response = client.embeddings.create(input=[text], model=model)
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings).astype('float32')

def save_embeddings_and_metadata(embeddings, metadata, vector_dir, filename_prefix="openai_embeddings"):
    pd.DataFrame(metadata).to_csv(os.path.join(vector_dir, f"{filename_prefix}_metadata.csv"), index=False)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(vector_dir, f"{filename_prefix}.faiss"))

def embed_node_names(driver, vector_dir):
    with driver.session() as session:
        result = session.run("MATCH (n:Entity) RETURN elementId(n) AS id, n.name AS name")

        node_embeddings = []
        node_metadata = []

        for record in tqdm(result, desc="Embedding KG nodes", dynamic_ncols=True):
            node_name = record["name"]
            element_id = record["id"]

            try:
                embedding = client.embeddings.create(
                    input=node_name,
                    model="text-embedding-ada-002"
                ).data[0].embedding

                node_embeddings.append(embedding)
                node_metadata.append({"element_id": element_id, "name": node_name})

            except Exception as e:
                print(f"Error embedding node '{node_name}': {e}")

        node_embeddings = np.array(node_embeddings).astype('float32')

        # Save node FAISS index and metadata
        save_embeddings_and_metadata(node_embeddings, node_metadata, vector_dir, filename_prefix="node_embeddings")

def main():
    articles = load_articles(ARTICLES_FILE)
    driver = GraphDatabase.driver(NEO4J_URI)

    texts = []
    metadata = []

    for article in tqdm(articles, desc="Processing articles", dynamic_ncols=True):
        if article.get('abstract'):
            full_text = f"Title: {article.get('title', '')}\nAbstract: {article['abstract']}"
            texts.append(full_text)
            metadata.append({
                "pubmed_id": article.get("pubmed_id"),
                "doi": article.get("doi", ""),
                "title": article.get("title", "")
            })

            triples = extract_triples(article['abstract'], article.get('title', ""))
            if triples:
                save_triples_to_backup(triples, BACKUP_FILE)
                create_knowledge_graph(triples, driver)

    embeddings = embed_texts(texts)
    save_embeddings_and_metadata(embeddings, metadata, VECTOR_DIR)

    embed_node_names(driver, VECTOR_DIR)

    driver.close()

    print("Processing complete. Triples stored in Neo4j, FAISS vector stores saved for both abstracts and KG nodes.")

if __name__ == "__main__":
    main()