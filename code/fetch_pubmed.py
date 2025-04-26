from Bio import Entrez
import json
from tqdm import tqdm
import os
from datetime import datetime

# Set your email to comply with NCBI guidelines
Entrez.email = "your_real_email@example.com"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_pubmed_articles(domain_term, max_articles=100, output_file="articles.json"):
    """
    Fetches articles from PubMed for a given MeSH domain term.
    
    Args:
        domain_term (str): MeSH query term, e.g., 'Oncology[MeSH Major Topic]'
        max_articles (int): Maximum number of articles to fetch
        output_file (str): Name of the output JSON file
    """
    handle = Entrez.esearch(db="pubmed", term=domain_term, retmax=max_articles)
    record = Entrez.read(handle)
    handle.close()
    ids = record["IdList"]

    articles = []

    for id_chunk in tqdm([ids[i:i+10] for i in range(0, len(ids), 10)]):
        fetch_handle = Entrez.efetch(db="pubmed", id=",".join(id_chunk), rettype="medline", retmode="xml")
        fetched_records = Entrez.read(fetch_handle)
        fetch_handle.close()

        for article_data in fetched_records['PubmedArticle']:
            pubmed_id = article_data['MedlineCitation']['PMID']
            abstract_text = ' '.join(article_data['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', [""]))
            title = article_data['MedlineCitation']['Article'].get('ArticleTitle', "")

            doi = ""
            article_ids = article_data['PubmedData']['ArticleIdList']
            for aid in article_ids:
                if aid.attributes.get('IdType') == 'doi':
                    doi = str(aid)
                    break  # Take first DOI if multiple exist

            articles.append({
                "pubmed_id": pubmed_id,
                "title": title,
                "abstract": abstract_text,
                "doi": doi,
                "domain": domain_term,
                "timestamp": datetime.now().isoformat()
            })

    save_path = os.path.join(DATA_DIR, output_file)
    with open(save_path, "w") as f:
        json.dump(articles, f, indent=2)

    print(f"Saved {len(articles)} articles to {save_path}")

if __name__ == "__main__":
    # Example for MVP
    fetch_pubmed_articles("Oncology[MeSH Major Topic]", max_articles=100, output_file="oncology_articles_100.json")

