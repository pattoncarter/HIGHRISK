# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc libpq-dev

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code
COPY . .

# Set environment variables (optional, otherwise use --env-file)
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "rag_inference.py"]
