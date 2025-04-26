
---

# Explainable HybridRAG Oncology Research Assistant

This project is a fully containerized, production-grade Explainable LightRAG + Knowledge Graph Retrieval system for biomedical research, with a focus on oncology.

It combines:
- Semantic document retrieval (using FAISS)
- Semantic Knowledge Graph (KG) retrieval (using Neo4j + FAISS)
- Explainable clinical summaries (using OpenAI GPT)
- Interactive knowledge graph visualization (using PyVis and Streamlit)

Designed for researchers who need grounded, explainable answers in oncology, with full source traceability.

---

## Main Components

- **Frontend**: Streamlit Web App
- **KG Storage**: Neo4j Database (Dockerized)
- **Vector Storage**: FAISS indices (local, persisted)
- **Embedding Models**: OpenAI Ada-002
- **Summarization Model**: OpenAI GPT-4.1-mini
- **Document Source**: PubMed abstracts (fetched automatically)

---

## Getting Started

Follow these steps to launch the system:

### 1. Clone the Repository

```bash
git clone https://your-repo-url.git
cd your-project-folder
```

### 2. Create a `.env` File

At the project root, create a `.env` file with the following contents:

```
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
VECTOR_DIR=vector_store
ARTICLES_FILE=data/oncology_articles_100.json
```

Replace `your_openai_api_key` with your actual OpenAI API key.

### 3. Build and Launch the System

Use the provided Makefile to spin everything up:

```bash
make up
```

This command will:
- Start Neo4j inside Docker.
- Build your application container.
- Automatically fetch PubMed oncology articles.
- Extract triplets and build the Knowledge Graph.
- Embed abstracts and node names.
- Launch the Streamlit frontend at [http://localhost:8501](http://localhost:8501).

---

## Project Directory Structure

```
/app/
  fetch_pubmed.py
  hybridRAG_preprocess.py
  explainable_rag_inference.py
  streamlit_frontend.py
  restore_neo4j_from_file.py      # Recovery script
  Dockerfile
  docker-compose.yml
  Makefile
  requirements.txt
  .env
/vector_store/                    # FAISS embeddings
/data/                             # Downloaded abstracts
/subgraphs/                        # Subgraph visualizations
```

---

## Available Makefile Commands

| Command       | Description                                           |
|---------------|-------------------------------------------------------|
| `make up`     | Build and launch the system (Neo4j, App, Frontend)     |
| `make down`   | Shut down and remove containers                       |
| `make logs`   | View live container logs for debugging                |
| `make rebuild`| Force rebuild of all images from scratch              |
| `make recover`| Restore the Knowledge Graph into Neo4j from backup    |

---

## Recovery Instructions (if Neo4j is Lost)

If the Neo4j database becomes corrupted or is reset:

1. Ensure Neo4j is running:

```bash
make up
```

2. In a separate terminal, restore the graph:

```bash
make recover
```

This will:
- Read triples from `data/triples_backup.jsonl`
- Rebuild the biomedical Knowledge Graph automatically in Neo4j

There is no need to re-fetch articles or re-embed abstracts.

---

## Example Workflow for a New Researcher

1. Pull the repository.
2. Add your `.env` file.
3. Run:

```bash
make up
```

4. Open your browser and navigate to [http://localhost:8501](http://localhost:8501).
5. Enter a clinical research query (for example, "How does Metformin affect breast cancer?").
6. Review the generated answer, sources, and interact with the subgraph visualization.
7. Export your results as JSON if needed.

---

## Notes

- This system is fully local and requires no external cloud infrastructure.
- If desired, it can be easily deployed to AWS, GCP, or Azure using Docker Compose or container orchestration services.
- PubMed data fetching may be subject to OpenAI API and PubMed API usage limits when scaled up significantly.

---

Would you also like me to give you a version that uses even more formal academic report style formatting (for example if you want to submit this as a project deliverable)?  
I can easily polish this even further if needed.