"""
GPT2 Token-by-Token Story Generation Demo
This script demonstrates how GPT2 generates text one token at a time.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_loader import setup_model


def generate_token(prompt:str, *, temperature:float=0.8, model=None, tokenizer=None) -> str:
    """
    Generates the next token following the starting sentence.
    """
    
    if model is None or tokenizer is None:
        model, tokenizer = setup_model()
    
    #First, the input is converted into vector or tensor
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        # Call the token generation model to generate just the next token
        outputs = model(input_ids)
        
        # Extract the output of the model, which are logits
        next_token_logits = outputs.logits[:, -1, :] / temperature
        
        # We normalize the logits to get the probablities
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        # We convert the vector back into text and print the new token
        new_text = tokenizer.decode(next_token[0])
        
        return new_text
    
    
def complete_sentence(prompt:str, *, max_tokens:int=40):
    """
    Generate a sentence using token generation.
    """
    
    model, tokenizer = setup_model()
    for _ in range(max_tokens):
        prompt += generate_token(prompt, model=model, tokenizer=tokenizer)

    # Print the complete output
    print("\n\n" + "=" * 70 + "\n" +prompt + "\n" + ("=" * 70) + "\n\n")
    

if __name__ == "__main__":
    
    """
    First, run this module. But then modify the beginning to
    see what it does.
    """
    user_input = input("What is the beginning of your sentence? ")
    complete_sentence(user_input)