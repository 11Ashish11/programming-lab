"""
Semantic search implementation. It defines several
helper functions to set up the model, calculate embeddings
and calcualate the cosine similarity. A search function 
implements executing a vector query against a set of 
example documents.
"""

from typing import List,Optional
import sys
from pathlib import Path

import torch
import numpy as np
from collections import defaultdict
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_loader import setup_embedding_model

# This holds one of the search results
class SearchResult(BaseModel):
    text:Optional[str] = None
    embeddings:Optional[List[float]] = None
    similarity:float = 0
    rank:int = 0
    

def get_embedding(text, tokenizer, model):
    """Get embedding vector for a piece of text"""
    # Split input text into tokens (e.g., words)
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    # Rum embedding model
    with torch.no_grad():
        outputs = model(**inputs)
    # Get model output (from last model layer)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    # Normalize embeddings from 0 to 1
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings[0].numpy()


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def execute_search(query:str, documents:List[str]) -> List[SearchResult]:
    """
    Runs a vector search. More specifically, it compares the vector
    embeddings of the query to the vector embeddings of each 
    document. It returns a sorted list of results with the best result
    in the first position.
    """
    
    # 1 - Prepare search result list
    #     You can use these objects to store the information you need to 
    #     perform the search and return a list of results. For instance,
    #     you can store the embeddings and the cosine similarity in each
    #     result object.
    results:List[SearchResult] = [SearchResult(text=d) for d in documents]

    # 2 - Set up the model
    tokenizer, model = setup_embedding_model()

    # 3 - Calculate query and document embeddings
    query_embedding = get_embedding(query, tokenizer, model)
    for res in results:
        res.embeddings = get_embedding(res.text, tokenizer, model).tolist()

    # 4 - Calculate similarities between the query and each document
    for res in results:
        res.similarity = float(cosine_similarity(query_embedding, res.embeddings))

    # 5 - Sort search results by similarity and assign its rank
    results.sort(key=lambda r:r.similarity, reverse=True)
    for idx,res in enumerate(results):
        res.rank = idx+1
    
    # Return results    
    return results

def semantic_search(documents:List[str]) -> None:
    """
    Main function for executing the semantic
    search. It handles user input and prints out
    the results.
    """
    
    # Ask user for input query until they write 'quit'
    while True:
        # Ask user for a query
        query = input("\nEnter search query (or 'quit'): ").strip()
        if query.lower() == 'quit':
            break
        if not query:
            continue
        
        # Perform the actual search
        results:List[SearchResult] = execute_search(query,documents)
        
        # Print the results
        print("\nSearch results:")
        for result in results[:3]:
            print(f"\n{result.rank}. [{result.similarity:.3f}]")
            print(f'   {result.text}')
    

if __name__ == "__main__":
    
    # These are the documents we will search
    documents = [
        "Yoga and meditation help reduce stress and anxiety",
        "The Great Wall of China is visible from space",
        "Climate change is affecting global weather patterns",
        "Python is a popular programming language used for web development and data science",
        "Machine learning models can predict outcomes based on historical data",
        "The Eiffel Tower is a famous landmark in Paris, France",
        "Regular exercise improves cardiovascular health and mental wellbeing",
        "Deep learning uses neural networks with multiple layers",
        "Italian cuisine is known for pasta, pizza, and risotto",
        "JavaScript is essential for front-end web development",
    ]
    
    semantic_search(documents)