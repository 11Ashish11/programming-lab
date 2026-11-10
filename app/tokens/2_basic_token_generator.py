"""
GPT2 Token-by-Token Story Generation Demo
This script demonstrates how GPT2 generates text one token at a time.
"""

import sys
from pathlib import Path

import torch
import time
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_loader import setup_model

def complete_sentence(prompt:str, *, max_tokens:int=40, temperature:float=0.8):
    """
    Generates up to 40 tokens following the starting
    sentence.
    """
    model, tokenizer = setup_model()
    
    print(f"\nStarting prompt: \"{prompt}\"")
    print("\nGenerating story token by token:\n")
    print(prompt, end='', flush=True)
    
    #First, the input is converted into vector or tensor
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        for i in range(max_tokens):
            
            # Call the token generation model to generate just the next token
            outputs = model(input_ids)
            
            # Extract the output of the model, which are logits
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # We normalize the logits to get the probablities
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # We convert the vector back into text and print the new token
            new_text = tokenizer.decode(next_token[0])
            print(new_text, end='', flush=True)
            
            # Now we add the new token to the input so that we can
            # generate the token after that in the next run.
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # This is just for show. Delaying the process so it shows that 
            # tokens are generated.
            time.sleep(random.uniform(0.3, 0.7))
            
            # We stop at the end of a sentence.
            if new_text.strip().endswith('.'):
                break
    
    print("\n\nGeneration complete!")


if __name__ == "__main__":
    
    """
    First, run this module. But then modify the beginning to
    see what it does.
    """
    prompt = input("What is the beginning of your sentence? ")
    complete_sentence(prompt, temperature=0.8, max_tokens=40)