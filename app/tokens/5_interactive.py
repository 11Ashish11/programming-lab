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


def interactive_token_choice(prompt, model, tokenizer, max_tokens=50, temperature=0.8, num_choices=5):
    """
    Interactive mode where user chooses the next token from top candidates
    
    Args:
        prompt: Starting text
        model: GPT2 model
        tokenizer: GPT2 tokenizer
        max_tokens: Maximum number of tokens to generate
        temperature: Controls randomness
        num_choices: Number of top choices to show
    """
    print("=" * 70)
    print("INTERACTIVE TOKEN CHOICE MODE")
    print("=" * 70)
    print("You will choose which token to add next from the top candidates!")
    print("This lets you 'steer' the story in different directions.\n")
    
    print(f'Starting prompt: "{prompt}"')
    print("=" * 70)
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    generated_text = prompt
    
    print(f"\n{generated_text}", end='', flush=True)
    
    with torch.no_grad():
        for i in range(max_tokens):
            # Get model predictions for next token
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Get top candidates
            top_probs, top_indices = torch.topk(probs[0], num_choices)
            
            # Display choices
            print("\n\n" + "-" * 50)
            print(f"Token #{i+1} - Choose next token:")
            print("-" * 50)
            
            choices = []
            for idx, (token_id, prob) in enumerate(zip(top_indices, top_probs), 1):
                token_text = tokenizer.decode(token_id)
                choices.append((token_id, token_text, prob.item()))
                print(f"  {idx}. '{token_text}' ({prob.item():.1%} probability)")
            
            # Let user choose
            while True:
                try:
                    choice = input(f"\nYour choice (1-{num_choices}, or 'a' for auto, 'q' to quit): ").strip().lower()
                    
                    if choice == 'q':
                        print("\nFinal story:")
                        print("=" * 70)
                        print(f'"{generated_text}"')
                        print("=" * 70)
                        return generated_text
                    
                    if choice == 'a':
                        # Auto-select the most likely token
                        selected_idx = 0
                        print("(Auto-selected most likely token)")
                        break
                    
                    choice_num = int(choice)
                    if 1 <= choice_num <= num_choices:
                        selected_idx = choice_num - 1
                        break
                    else:
                        print(f"Please enter a number between 1 and {num_choices}")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            # Get the selected token
            selected_token_id, selected_text, selected_prob = choices[selected_idx]
            
            # Update the story
            generated_text += selected_text
            print(f"\n✓ Added: '{selected_text}'")
            print(f"Story so far: \"{generated_text}\"")
            
            # Update input_ids
            next_token = torch.tensor([[selected_token_id]])
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Check for natural ending
            if selected_text.strip().endswith('.'):
                print("\n(Natural sentence ending detected)")
                cont = input("Continue? (y/n): ").strip().lower()
                if cont != 'y':
                    break
    
    print("\n" + "=" * 70)
    print("FINAL STORY:")
    print("=" * 70)
    print(f'"{generated_text}"')
    print("=" * 70)
    
    return generated_text


if __name__ == "__main__":
    model, tokenizer = setup_model()
    prompt = input("\nEnter your starting prompt: ").strip()
    if prompt:
        try:
            temp = float(input("Temperature? (0.1-2.0, default 0.8): ") or "0.8")
            tokens = int(input("Max tokens? (10-50, default 30): ") or "30")
        except ValueError:
            print("Using defaults (temp=0.8, tokens=30)")
            temp = 0.8
            tokens = 30
        interactive_token_choice(prompt, model, tokenizer, max_tokens=tokens, temperature=temp)
    else:
        print("No prompt provided!")