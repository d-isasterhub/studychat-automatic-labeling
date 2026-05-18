#!/usr/bin/python3
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="../chroma_db")
collection = client.get_collection("assignments")

def retrieve(query, k=5, assignment=None):
    '''
    Retrieve k nearest entries to query from collection (embedding-based).
    
    :param query: Query to be compared
    :param k: Number of nearest entries to extract from the collection
    :param assignment: (Optional) Limit results to assignment
    '''
    emb = model.encode([query])[0]
    
    query_args = {
        "query_embeddings": [emb.tolist()],
        "n_results": k
    }
    
    if assignment:
        query_args["where"] = {"assignment": assignment}
        
    results = collection.query(**query_args)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    
    return docs, metas