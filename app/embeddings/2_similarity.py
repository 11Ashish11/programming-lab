"""
Embedding Models Explorer
An interactive, visual way to understand how embedding models work
and what they can do
"""

from typing import List, Tuple
import sys
from pathlib import Path

import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_loader import setup_embedding_model


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


def visualize_similarity(similarity):
    """Create a visual bar for similarity score"""
    # Similarity ranges from -1 to 1, but we'll mostly see 0 to 1
    normalized = max(0, similarity)  # Clamp to 0-1
    bar_length = int(normalized * 30)
    bar = '█' * bar_length + '░' * (30 - bar_length)
    
    # Emoji based on similarity
    if similarity > 0.8:
        emoji = "🔥"
        level = "Very Similar"
    elif similarity > 0.6:
        emoji = "✅"
        level = "Similar"
    elif similarity > 0.4:
        emoji = "🤔"
        level = "Somewhat Similar"
    elif similarity > 0.2:
        emoji = "😐"
        level = "Slightly Similar"
    else:
        emoji = "❌"
        level = "Not Similar"
    
    return bar, emoji, level

def similarity_demo(comparisons:List[Tuple[str,str]]):
    """Interactive similarity demonstration"""
    print("\n" + "=" * 70)
    print("SIMILARITY EXPLORER")
    print("=" * 70)
    print("Let's compare how similar different sentences are!\n")
    
    tokenizer, model = setup_embedding_model()
    
    print("Computing embeddings and similarities...\n")
    
    for text1, text2 in comparisons:
        emb1 = get_embedding(text1, tokenizer, model)
        emb2 = get_embedding(text2, tokenizer, model)
        
        similarity = cosine_similarity(emb1, emb2)
        bar, emoji, level = visualize_similarity(similarity)
        
        print(f'Text 1: "{text1}"')
        print(f'Text 2: "{text2}"')
        print(f'Similarity: [{bar}] {similarity:.3f} {emoji} {level}')
        print()
        
    print("=" * 70)
    print("Key Insight: The model learned that similar MEANINGS have similar")
    print("embeddings, even if the words are different!")
    print("=" * 70)

if __name__ == "__main__":
    
    """
    Compares multiple instances of sentence pairs to
    illustrate the difference in similarity.
    """
    
    # Demo comparisons
    comparisons:List[Tuple[str,str]] = [
        ("I love pizza", "Pizza is my favorite food"),
        ("I love pizza", "I enjoy eating pizza"),
        ("I love pizza", "I hate broccoli"),
        ("I love pizza", "The sky is blue"),
        ("The cat sleeps", "The dog rests"),
        ("Paris is beautiful", "Paris is gorgeous"),
        ("He runs fast", "She sprints quickly"),
        ("Python is a programming language", "Python is a type of snake"),
    ]
    
    
    similarity_demo(comparisons)