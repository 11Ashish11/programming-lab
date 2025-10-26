"""
Embedding Models Explorer
An interactive, visual way to understand how embedding models work
and what they can do
"""

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from collections import defaultdict

def setup_embedding_model():
    """Load a small, fast embedding model"""
    print("Loading embedding model... (this may take a moment)")
    
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    print("Model loaded!\n")
    return tokenizer, model


def get_embedding(text, tokenizer, model):
    """Get embedding vector for a piece of text"""
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    
    # Normalize
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    
    return embeddings[0].numpy()


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def similarity_demo(text1:str, text2:str):
    """Demonstrate how to use embeddings"""
    tokenizer, model = setup_embedding_model()
    
    print("Computing embeddings and similarities...\n")
    
    # Calculate vector for each text separately
    emb1 = get_embedding(text1, tokenizer, model)
    print(f"\nEmbedding vector for `{text1}`: {str(emb1[:4])[:45].replace("\n","")+"..."}")
    emb2 = get_embedding(text2, tokenizer, model)
    print(f"\nEmbedding vector for `{text2}`: {str(emb2[:4])[:45].replace("\n","")+"..."}")
    
    # Now calculate the similarity between the two vectors
    similarity = cosine_similarity(emb1, emb2)
    print(f"\nSimilarity is {similarity:.2f}\n\n")

if __name__ == "__main__":
    text1 = "I love melted cheese and pepperoni"
    text2 = "Pizza is my favorite food"
    similarity_demo(text1, text2)