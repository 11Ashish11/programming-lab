"""
Embedding Models Explorer
An interactive, visual way to understand how embedding models work
and what they can do
"""

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


def what_are_embeddings():
    """Visual explanation of what embeddings are"""
    print("=" * 70)
    print("WHAT ARE EMBEDDINGS?")
    print("=" * 70)
    print("""
Embeddings are like GPS coordinates for words and sentences!

Just like every location on Earth has coordinates (latitude, longitude),
every piece of text gets converted into numbers (a vector).

Imagine a map where:
  • Similar meanings are CLOSE together
  • Different meanings are FAR apart

Example (simplified to 2D, real embeddings have 384+ dimensions):

    "dog" → [0.8, 0.3]
    "puppy" → [0.75, 0.35]  ← Close to "dog"!
    "cat" → [0.7, 0.25]     ← Also nearby (both pets)
    "car" → [-0.2, 0.9]     ← Far away (different concept)

Let's see this in action with real embeddings...
""")
    input("\nPress Enter to continue...")


def similarity_demo():
    """Interactive similarity demonstration"""
    print("\n" + "=" * 70)
    print("SIMILARITY EXPLORER")
    print("=" * 70)
    print("Let's compare how similar different sentences are!\n")
    
    tokenizer, model = setup_embedding_model()
    
    # Demo comparisons
    comparisons = [
        ("I love pizza", "Pizza is my favorite food", "Very similar meaning"),
        ("I love pizza", "I enjoy eating pizza", "Very similar meaning"),
        ("I love pizza", "I hate broccoli", "Both about food preferences"),
        ("I love pizza", "The sky is blue", "Completely unrelated"),
        ("The cat sleeps", "The dog rests", "Similar structure and meaning"),
        ("Paris is beautiful", "Paris is gorgeous", "Same meaning, different word"),
        ("He runs fast", "She sprints quickly", "Same concept, different words"),
        ("Python is a programming language", "Python is a type of snake", "Same word, different meaning!"),
    ]
    
    print("Computing embeddings and similarities...\n")
    
    for text1, text2, explanation in comparisons:
        emb1 = get_embedding(text1, tokenizer, model)
        emb2 = get_embedding(text2, tokenizer, model)
        
        similarity = cosine_similarity(emb1, emb2)
        bar, emoji, level = visualize_similarity(similarity)
        
        print(f'Text 1: "{text1}"')
        print(f'Text 2: "{text2}"')
        print(f'Similarity: [{bar}] {similarity:.3f} {emoji} {level}')
        print(f'Why? {explanation}')
        print()
        
    print("=" * 70)
    print("Key Insight: The model learned that similar MEANINGS have similar")
    print("embeddings, even if the words are different!")
    print("=" * 70)


def find_the_odd_one_out():
    """Game: find which sentence doesn't belong"""
    print("\n" + "=" * 70)
    print("GAME: FIND THE ODD ONE OUT")
    print("=" * 70)
    print("One of these sentences doesn't belong with the others.")
    print("Can you find it? (The model will too!)\n")
    
    tokenizer, model = setup_embedding_model()
    
    puzzles = [
        {
            'sentences': [
                "I'm learning to code in Python",
                "She's studying JavaScript programming",
                "He's making a sandwich",
                "They're taking a software engineering course"
            ],
            'odd_index': 2,
            'theme': 'Programming/Learning'
        },
        {
            'sentences': [
                "The dog barked loudly",
                "The cat meowed softly",
                "The bird chirped happily",
                "The building collapsed suddenly"
            ],
            'odd_index': 3,
            'theme': 'Animal sounds'
        },
        {
            'sentences': [
                "I feel happy today",
                "She seems joyful",
                "He appears content",
                "The table is wooden"
            ],
            'odd_index': 3,
            'theme': 'Emotions/feelings'
        },
    ]
    
    score = 0
    
    for i, puzzle in enumerate(puzzles, 1):
        print(f"\n--- Puzzle {i} ---")
        for j, sent in enumerate(puzzle['sentences']):
            print(f"{j+1}. {sent}")
        
        # User guesses
        while True:
            try:
                guess = int(input("\nWhich one is the odd one out? (1-4): "))
                if 1 <= guess <= 4:
                    break
                print("Please enter 1, 2, 3, or 4")
            except ValueError:
                print("Please enter a number")
        
        # Calculate embeddings
        embeddings = [get_embedding(sent, tokenizer, model) for sent in puzzle['sentences']]
        
        # For each sentence, calculate average similarity to all others
        avg_similarities = []
        for j, emb in enumerate(embeddings):
            similarities = [cosine_similarity(emb, other_emb) 
                          for k, other_emb in enumerate(embeddings) if k != j]
            avg_similarities.append(np.mean(similarities))
        
        # The odd one out has the lowest average similarity
        model_answer = np.argmin(avg_similarities) + 1
        
        print(f"\nYour answer: #{guess}")
        print(f"Model's answer: #{model_answer}")
        
        if guess == model_answer:
            print("✓ CORRECT! You found the odd one out!")
            score += 1
        else:
            print(f"✗ The odd one out was #{model_answer}")
        
        print(f"\nExplanation: The theme was '{puzzle['theme']}'")
        print("Average similarity to other sentences:")
        for j, sim in enumerate(avg_similarities, 1):
            indicator = "👈 Odd one out!" if j == model_answer else ""
            print(f"  Sentence {j}: {sim:.3f} {indicator}")
    
    print("\n" + "=" * 70)
    print(f"Final Score: {score}/{len(puzzles)}")
    if score == len(puzzles):
        print("🎉 Perfect! You understand semantic similarity!")
    print("=" * 70)


def semantic_search_demo():
    """Demonstrate semantic search - the killer app of embeddings"""
    print("\n" + "=" * 70)
    print("SEMANTIC SEARCH: Find Meaning, Not Just Words")
    print("=" * 70)
    print("""
Traditional search (like Ctrl+F) finds exact word matches.
Semantic search finds MEANING matches!

Example:
  Query: "How do I learn ML?"
  Traditional: Only finds exact phrase "How do I learn ML"
  Semantic: Also finds "Machine learning tutorials", 
            "Getting started with AI", etc.

Let's try it!
""")
    
    tokenizer, model = setup_embedding_model()
    
    # A mini "database" of documents
    documents = [
        "Python is a popular programming language used for web development and data science",
        "Machine learning models can predict outcomes based on historical data",
        "The Eiffel Tower is a famous landmark in Paris, France",
        "Regular exercise improves cardiovascular health and mental wellbeing",
        "Deep learning uses neural networks with multiple layers",
        "Italian cuisine is known for pasta, pizza, and risotto",
        "Climate change is affecting global weather patterns",
        "JavaScript is essential for front-end web development",
        "Yoga and meditation help reduce stress and anxiety",
        "The Great Wall of China is visible from space",
    ]
    
    print("Indexing documents (computing embeddings)...\n")
    doc_embeddings = [get_embedding(doc, tokenizer, model) for doc in documents]
    print(f"Indexed {len(documents)} documents!\n")
    
    # Demo searches
    demo_queries = [
        ("AI and neural networks", "Should find ML/DL docs"),
        ("healthy living tips", "Should find exercise/yoga docs"),
        ("coding languages", "Should find Python/JavaScript docs"),
        ("famous places to visit", "Should find landmarks"),
    ]
    
    print("Let's run some searches:\n")
    
    for query, expectation in demo_queries:
        print(f'Query: "{query}"')
        print(f'Expected: {expectation}\n')
        
        # Get query embedding
        query_emb = get_embedding(query, tokenizer, model)
        
        # Calculate similarities
        similarities = [cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embeddings]
        
        # Get top 3 results
        top_indices = np.argsort(similarities)[::-1][:3]
        
        print("Top 3 results:")
        for rank, idx in enumerate(top_indices, 1):
            bar, emoji, level = visualize_similarity(similarities[idx])
            print(f"{rank}. [{bar}] {similarities[idx]:.3f} {emoji}")
            print(f'   "{documents[idx]}"')
        print()
    
    # Interactive search
    print("=" * 70)
    print("Now you try! Search the document database:")
    print("=" * 70)
    
    while True:
        query = input("\nEnter search query (or 'quit'): ").strip()
        if query.lower() == 'quit':
            break
        if not query:
            continue
        
        query_emb = get_embedding(query, tokenizer, model)
        similarities = [cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embeddings]
        top_indices = np.argsort(similarities)[::-1][:3]
        
        print("\nSearch results:")
        for rank, idx in enumerate(top_indices, 1):
            bar, emoji, level = visualize_similarity(similarities[idx])
            print(f"\n{rank}. Score: {similarities[idx]:.3f} {emoji} {level}")
            print(f'   {documents[idx]}')


def clustering_demo():
    """Show how embeddings can automatically group similar texts"""
    print("\n" + "=" * 70)
    print("AUTOMATIC CLUSTERING: Group Similar Texts")
    print("=" * 70)
    print("""
Embeddings can automatically group similar texts together!
This is useful for:
  • Organizing customer feedback
  • Finding duplicate questions
  • Grouping news articles by topic
  • Content recommendation

Let's see it in action...
""")
    
    tokenizer, model = setup_embedding_model()
    
    # Mix of different topics
    texts = [
        # Sports (cluster 1)
        "The basketball game was intense last night",
        "She scored the winning goal in overtime",
        "The tennis match lasted five hours",
        
        # Food (cluster 2)
        "This pasta recipe is delicious",
        "I love trying new restaurants",
        "The chocolate cake was amazing",
        
        # Technology (cluster 3)
        "The new smartphone has great features",
        "Cloud computing is revolutionizing business",
        "The software update fixed many bugs",
        
        # Weather (cluster 4)
        "It's going to rain tomorrow",
        "The sunny weather is perfect for hiking",
        "The snowstorm closed many roads",
    ]
    
    print("Computing embeddings for all texts...\n")
    embeddings = [get_embedding(text, tokenizer, model) for text in texts]
    
    # Simple clustering: for each text, find its closest neighbors
    print("Finding clusters (groups of similar texts):\n")
    
    for i, text in enumerate(texts):
        similarities = []
        for j, other_emb in enumerate(embeddings):
            if i != j:
                sim = cosine_similarity(embeddings[i], other_emb)
                similarities.append((j, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f'{i+1}. "{text}"')
        print(f'   Most similar to:')
        for j, sim in similarities[:2]:
            print(f'     • "{texts[j]}" (similarity: {sim:.3f})')
        print()
    
    print("=" * 70)
    print("Notice how sports texts cluster with sports,")
    print("food with food, tech with tech, weather with weather!")
    print("The model learned these groupings from the MEANING, not keywords!")
    print("=" * 70)


def analogy_explorer():
    """Explore word analogies using embeddings"""
    print("\n" + "=" * 70)
    print("ANALOGY EXPLORER: Vector Arithmetic")
    print("=" * 70)
    print("""
Famous example: king - man + woman = queen

Embeddings support arithmetic! You can do:
  • Word relationships: Paris - France + Italy = Rome
  • Concept translation: Doctor - Hospital + School = Teacher
  • Attribute transfer: Cat - Meow + Dog = Bark

Let's try some analogies...
""")
    
    tokenizer, model = setup_embedding_model()
    
    # Prepare a vocabulary of words/phrases
    vocabulary = [
        "king", "queen", "man", "woman", "prince", "princess",
        "doctor", "nurse", "hospital", "clinic",
        "teacher", "student", "school", "university",
        "Paris", "France", "Rome", "Italy", "London", "England",
        "hot", "cold", "big", "small", "happy", "sad",
        "cat", "dog", "meow", "bark",
    ]
    
    print("Building vocabulary embeddings...\n")
    vocab_embeddings = {word: get_embedding(word, tokenizer, model) for word in vocabulary}
    
    analogies = [
        ("king", "man", "woman", "Expected: queen"),
        ("Paris", "France", "Italy", "Expected: Rome"),
        ("doctor", "hospital", "school", "Expected: teacher"),
        ("hot", "cold", "small", "Expected: big"),
    ]
    
    for word1, word2, word3, expected in analogies:
        print(f"Analogy: {word1} - {word2} + {word3} = ?")
        print(f"{expected}")
        
        # Vector arithmetic
        result_vector = vocab_embeddings[word1] - vocab_embeddings[word2] + vocab_embeddings[word3]
        
        # Find closest word in vocabulary
        similarities = []
        for word, emb in vocab_embeddings.items():
            if word not in [word1, word2, word3]:  # Exclude input words
                sim = cosine_similarity(result_vector, emb)
                similarities.append((word, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 3 results:")
        for word, sim in similarities[:3]:
            print(f"  {word}: {sim:.3f}")
        print()
    
    print("=" * 70)
    print("The model learned relationships between concepts!")
    print("=" * 70)


def duplication_detector():
    """Find duplicate or near-duplicate texts"""
    print("\n" + "=" * 70)
    print("DUPLICATE DETECTOR: Finding Copies and Paraphrases")
    print("=" * 70)
    print("""
Use case: You have a database of customer questions.
Many are duplicates or paraphrases of each other.
Embeddings can find these automatically!
""")
    
    tokenizer, model = setup_embedding_model()
    
    questions = [
        "How do I reset my password?",
        "I forgot my password, how can I recover it?",
        "What's the process for password reset?",
        "How much does shipping cost?",
        "What are the delivery charges?",
        "When will my order arrive?",
        "How long does shipping take?",
        "Can I return my purchase?",
        "What is your refund policy?",
        "How do I track my package?",
    ]
    
    print("Analyzing questions for duplicates...\n")
    
    embeddings = [get_embedding(q, tokenizer, model) for q in questions]
    
    # Find pairs with high similarity
    threshold = 0.7  # 70% similarity
    duplicates = []
    
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > threshold:
                duplicates.append((i, j, sim))
    
    duplicates.sort(key=lambda x: x[2], reverse=True)
    
    print("Potential duplicate pairs found:\n")
    for i, j, sim in duplicates:
        bar, emoji, level = visualize_similarity(sim)
        print(f"Similarity: [{bar}] {sim:.3f} {emoji}")
        print(f'  1. "{questions[i]}"')
        print(f'  2. "{questions[j]}"')
        print()
    
    print("=" * 70)
    print("These could be merged into single FAQ entries!")
    print("=" * 70)


def embedding_visualization():
    """Show what happens to embeddings with small text changes"""
    print("\n" + "=" * 70)
    print("EMBEDDING SENSITIVITY: How Changes Affect Embeddings")
    print("=" * 70)
    print("Let's see how small changes to text affect the embedding:\n")
    
    tokenizer, model = setup_embedding_model()
    
    base_text = "I love programming in Python"
    
    variations = [
        (base_text, "Original"),
        ("I love programming in Python", "Exact duplicate"),
        ("I love coding in Python", "Similar word (coding vs programming)"),
        ("I love programming in Java", "Different programming language"),
        ("I hate programming in Python", "Opposite sentiment"),
        ("Python programming is something I love", "Reordered"),
        ("The weather is nice today", "Completely different"),
    ]
    
    base_emb = get_embedding(base_text, tokenizer, model)
    
    print(f'Base text: "{base_text}"\n')
    print("Comparing variations:\n")
    
    for text, description in variations[1:]:
        emb = get_embedding(text, tokenizer, model)
        sim = cosine_similarity(base_emb, emb)
        bar, emoji, level = visualize_similarity(sim)
        
        print(f'{description}:')
        print(f'  "{text}"')
        print(f'  Similarity: [{bar}] {sim:.3f} {emoji}')
        print()
    
    print("=" * 70)
    print("Key insights:")
    print("• Exact duplicates → ~1.0 similarity")
    print("• Synonyms → High similarity (~0.8-0.9)")
    print("• Word order changes → Still high similarity")
    print("• Sentiment changes → Moderate similarity (~0.5-0.7)")
    print("• Different topics → Low similarity (~0.0-0.3)")
    print("=" * 70)


def main_menu():
    """Main menu for all embedding activities"""
    print("=" * 70)
    print("EMBEDDING MODELS EXPLORER")
    print("=" * 70)
    print("\nChoose an activity:")
    print("1. What are Embeddings? (start here!)")
    print("2. Similarity Explorer (see how similarity works)")
    print("3. Find the Odd One Out Game (test your intuition)")
    print("4. Semantic Search Demo (the killer application)")
    print("5. Automatic Clustering (group similar texts)")
    print("6. Analogy Explorer (vector arithmetic)")
    print("7. Duplicate Detector (find paraphrases)")
    print("8. Embedding Sensitivity (how changes affect embeddings)")
    print("9. Run all demonstrations (teaching mode)")
    
    choice = input("\nEnter choice (1-9): ").strip()
    
    if choice == '1':
        what_are_embeddings()
        similarity_demo()
    elif choice == '2':
        similarity_demo()
    elif choice == '3':
        find_the_odd_one_out()
    elif choice == '4':
        semantic_search_demo()
    elif choice == '5':
        clustering_demo()
    elif choice == '6':
        analogy_explorer()
    elif choice == '7':
        duplication_detector()
    elif choice == '8':
        embedding_visualization()
    elif choice == '9':
        print("\n🎓 Running full demonstration...\n")
        what_are_embeddings()
        similarity_demo()
        input("\nPress Enter to continue...")
        find_the_odd_one_out()
        input("\nPress Enter to continue...")
        semantic_search_demo()
        input("\nPress Enter to continue...")
        clustering_demo()
        input("\nPress Enter to continue...")
        embedding_visualization()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main_menu()