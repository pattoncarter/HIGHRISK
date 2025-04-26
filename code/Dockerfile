# Use a lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy your entire project
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Create necessary folders if missing
RUN mkdir -p vector_store data subgraphs

# Default Command:
# Fetch new documents -> Preprocess into Neo4j -> Launch Streamlit frontend
CMD ["bash", "-c", "python fetch_pubmed.py && python hybridRAG_preprocess.py && streamlit run streamlit_frontend.py --server.port=8501 --server.address=0.0.0.0"]
