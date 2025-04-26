# 🧬 Explainable HybridRAG Oncology Research Assistant

This project is a fully containerized, production-grade **Explainable LightRAG + Knowledge Graph Retrieval** system for biomedical research, with a focus on **oncology**.

It combines:
- **Semantic document retrieval** (using FAISS)
- **Semantic Knowledge Graph (KG) retrieval** (using Neo4j + FAISS)
- **Explainable clinical summaries** (using OpenAI GPT)
- **Interactive knowledge graph visualization** (using PyVis + Streamlit)

Designed for researchers who need **grounded, explainable answers** in oncology, with full **source traceability**.

---

## 📦 Main Components

- **Frontend**: Streamlit Web App
- **KG Storage**: Neo4j Database (Dockerized)
- **Vector Storage**: FAISS indices (local, persisted)
- **Embedding Models**: OpenAI Ada-002
- **Summarization Model**: OpenAI GPT-4.1-mini
- **Document Source**: PubMed abstracts (fetched automatically)

---

## 🚀 Getting Started

Follow these steps to launch the entire system:

### 1. Clone the repository

```bash
git clone https://your-repo-url.git
cd your-project-folder
```
2. Create a .env File

At the project root, create a .env file containing:

OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
VECTOR_DIR=vector_store
ARTICLES_FILE=data/oncology_articles_100.json

✅ Replace your_openai_api_key with your actual OpenAI API key.
3. Build and Launch the System

Use make to spin everything up easily:

make up

✅ This will:

    Start Neo4j inside Docker.

    Build your app container.

    Automatically fetch PubMed oncology articles.

    Extract triplets and build the KG.

    Embed abstracts + nodes.

    Launch the Streamlit frontend on http://localhost:8501.

🛠 Project Directory Structure

/app/
  fetch_pubmed.py
  hybridRAG_preprocess.py
  explainable_rag_inference.py
  streamlit_frontend.py
  restore_neo4j_from_file.py  # Recovery script
  Dockerfile
  docker-compose.yml
  Makefile
  requirements.txt
  .env
  /vector_store/              # FAISS embeddings
  /data/                      # Downloaded abstracts
  /subgraphs/                 # Subgraph visualizations

🧪 Available Makefile Commands
Command	Description
make up	Build and launch the system (Neo4j + App + Frontend)
make down	Shut down and clean up containers
make logs	View live container logs (debugging)
make rebuild	Force rebuild all images from scratch
make recover	Restore the Knowledge Graph into Neo4j from triples_backup.jsonl
🔥 Recovery Instructions (if Neo4j is Lost)

If the Neo4j database becomes corrupted or is reset:

    Ensure Neo4j is running:

make up

    In a separate terminal, run:

make recover

✅ This will:

    Read from data/triples_backup.jsonl

    Restore all triples into Neo4j

    Rebuild the biomedical Knowledge Graph automatically

No need to re-fetch articles or re-embed abstracts!
⚡ Example Workflow for a New Researcher

    Pull the repo

    Add your .env

    Run:

make up

    Open a browser to http://localhost:8501

    Enter clinical queries (e.g., "How does Metformin affect breast cancer?")

    Review generated answers, sources, and interact with subgraph visualizations.

    Export your results as JSON if needed.

📋 Notes

    This system is fully local — no cloud infrastructure required.

    If needed, it can easily be deployed to AWS/GCP/Azure via Docker Compose or ECS.

    PubMed data fetching is limited by OpenAI API and PubMed API quotas if running at large scale.

🚀 Future Improvements (Ideas)

    Auto-backup updated Neo4j database at intervals

    GraphQL API access to the Knowledge Graph

    Deploy multi-user version with authentication

    Live subgraph clustering and advanced visualization

🙏 Acknowledgments

    Thanks to the PubMed database for public biomedical abstracts.

    Thanks to OpenAI for GPT models.

    Thanks to Neo4j and FAISS teams for incredible open source tools.