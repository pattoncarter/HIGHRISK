from Bio import Entrez
import json
from tqdm import tqdm
import os
from datetime import datetime

Entrez.email = "your_email@example.com"

def fetch_pubmed_articles(search_term, max_articles=100):
    handle = Entrez.esearch(db="pubmed", term=search_term, retmax=max_articles)
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
            
            doi = ""
            article_ids = article_data['PubmedData']['ArticleIdList']
            for aid in article_ids:
                if aid.attributes.get('IdType') == 'doi':
                    doi = str(aid)
                    break  # Take first DOI if multiple exist

            articles.append({
                "pubmed_id": pubmed_id,
                "abstract": abstract_text,
                "doi": doi,
                "timestamp": datetime.now().isoformat()
            })

    # Save
    os.makedirs("data", exist_ok=True)
    with open("data/articles.json", "w") as f:
        json.dump(articles, f, indent=2)

if __name__ == "__main__":
    fetch_pubmed_articles("Cardiology[MeSH Major Topic]", 100)
