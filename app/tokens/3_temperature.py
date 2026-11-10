"""
GPT2 Token-by-Token Story Generation Demo
This script demonstrates how GPT2 generates text one token at a time,
making it visible and understandable for students.
"""

import sys
from pathlib import Path

import torch
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_loader import setup_model


def generate_comparison(prompt, temperatures=[0.3, 0.7, 1.2]):
    """
    Generate the same story with different temperatures to show the effect
    """
    
    model, tokenizer = setup_model()
    
    
    print("\n\n" + "=" * 70)
    print("TEMPERATURE COMPARISON")
    print("=" * 70)
    print(f"Prompt: \"{prompt}\"\n")
    
    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        story = prompt
        
        with torch.no_grad():
            for _ in range(30):
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :] / temp
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                story += tokenizer.decode(next_token[0])
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                if tokenizer.decode(next_token[0]).strip().endswith('.'):
                    break
        
        print(f"\"{story}\"")


if __name__ == "__main__":
    
    prompt = "The future of technology will be"
    generate_comparison( prompt,  temperatures=[0.3, 0.7, 1.2]  )