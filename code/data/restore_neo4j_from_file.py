import json
import sys
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI)

def restore_triples(triple_file, driver):
    with driver.session() as session:
        with open(triple_file, "r") as f:
            for line in f:
                try:
                    triple = json.loads(line.strip())
                    head = triple['head']
                    relation = triple['relation']
                    tail = triple['tail']
                    
                    session.run(
                        "MERGE (e1:Entity {name: $head}) "
                        "MERGE (e2:Entity {name: $tail}) "
                        "MERGE (e1)-[:RELATION {type: $relation}]->(e2)",
                        head=head, tail=tail, relation=relation
                    )
                except Exception as e:
                    print(f"Skipping bad line: {line}")
                    print(e)

    print("✅ Graph restoration complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python restore_triples_cli.py <triple_store_file.jsonl>")
        sys.exit(1)

    triple_store_file = sys.argv[1]

    restore_triples(triple_store_file, driver)
    driver.close()
