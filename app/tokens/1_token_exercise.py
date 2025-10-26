"""
GPT2 Token-by-Token Story Generation Demo
This script demonstrates how GPT2 generates text one token at a time.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def setup_model():
    """Load GPT2 model and tokenizer"""
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    return model, tokenizer

def generate_token(prompt:str, *, temperature:float=0.8) -> str:
    """
    Generates up to 40 tokens following the starting
    sentence.
    """
    
    # Load the model
    model, tokenizer = setup_model()
    
    #First, the input is converted into vector or tensor
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
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
    
    # TODO Use looping to continue the sentence max_tokens times
    
    # Print the complete output
    print("\n\n" + "=" * 70 + "\n" +prompt + "\n" + ("=" * 70) + "\n\n")
    

if __name__ == "__main__":
    
    """
    First, run this module. But then modify the beginning to
    see what it does.
    """
    user_input = input("What is the beginning of your sentence? ")
    complete_sentence(user_input)