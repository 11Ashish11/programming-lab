"""
GPT2 Token-by-Token Story Generation Demo
This script demonstrates how GPT2 generates text one token at a time,
making it visible and understandable for students.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time

def setup_model():
    """Load GPT2 model and tokenizer"""
    print("Loading GPT2 model... (this may take a moment)")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()  # Set to evaluation mode
    print("Model loaded!\n")
    return model, tokenizer


def generate_story_stepwise(prompt, model, tokenizer, max_tokens=50, temperature=0.8):
    """
    Generate a story one token at a time, showing the process
    
    Args:
        prompt: Starting text
        model: GPT2 model
        tokenizer: GPT2 tokenizer
        max_tokens: Maximum number of tokens to generate
        temperature: Controls randomness (higher = more random)
    """
    print("=" * 70)
    print("STARTING PROMPT:")
    print("=" * 70)
    print(f'"{prompt}"\n')
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    print("=" * 70)
    print("GENERATING TOKENS ONE BY ONE:")
    print("=" * 70)
    print(f"Starting with {len(input_ids[0])} tokens from the prompt\n")
    
    # Track the full generated text
    generated_text = prompt
    
    with torch.no_grad():
        for i in range(max_tokens):
            # Get model predictions for next token
            outputs = model(input_ids)
            
            # Get logits for the last token
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Get probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Decode the new token
            new_text = tokenizer.decode(next_token[0])
            
            # Get top 5 alternatives for educational purposes
            top_probs, top_indices = torch.topk(probs[0], 5)
            alternatives = [
                (tokenizer.decode(idx), prob.item()) 
                for idx, prob in zip(top_indices, top_probs)
            ]
            
            # Display the token generation
            print(f"Token #{i+1}:")
            print(f"  Generated: '{new_text}'")
            print(f"  Top alternatives:")
            for alt_text, alt_prob in alternatives[:3]:
                print(f"    '{alt_text}' ({alt_prob:.1%})")
            print()
            
            # Update the generated text
            generated_text += new_text
            
            # Append to input_ids for next iteration
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Small delay for readability
            time.sleep(0.3)
            
            # Stop if we generate an end token or period (optional)
            if new_text.strip().endswith('.'):
                print("(Encountered period - natural stopping point)")
                break
    
    print("=" * 70)
    print("FINAL GENERATED STORY:")
    print("=" * 70)
    print(f'"{generated_text}"')
    print("=" * 70)


def generate_comparison(prompt, model, tokenizer, temperatures=[0.3, 0.7, 1.2]):
    """
    Generate the same story with different temperatures to show the effect
    """
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


def interactive_token_generation():
    """
    Interactive mode where students can enter prompts and see generation
    """
    model, tokenizer = setup_model()
    
    print("\n" + "=" * 70)
    print("INTERACTIVE TOKEN GENERATION MODE")
    print("=" * 70)
    print("Enter prompts to see how GPT2 generates text token-by-token")
    print("Type 'compare' to see temperature comparison")
    print("Type 'choose' to manually choose each token")
    print("Type 'quit' to exit\n")
    
    while True:
        prompt = input("Enter your story prompt: ").strip()
        
        if prompt.lower() == 'quit':
            print("Goodbye!")
            break
        
        if prompt.lower() == 'compare':
            compare_prompt = input("Enter prompt for comparison: ").strip()
            generate_comparison(compare_prompt, model, tokenizer)
            continue
        
        if prompt.lower() == 'choose':
            choose_prompt = input("Enter prompt for interactive choice mode: ").strip()
            try:
                temp = float(input("Temperature? (0.1-2.0, default 0.8): ") or "0.8")
                tokens = int(input("Max tokens? (10-50, default 30): ") or "30")
            except ValueError:
                print("Using defaults (temp=0.8, tokens=30)")
                temp = 0.8
                tokens = 30
            print()
            interactive_token_choice(choose_prompt, model, tokenizer, max_tokens=tokens, temperature=temp)
            continue
        
        if not prompt:
            print("Please enter a prompt!\n")
            continue
        
        try:
            tokens = int(input("How many tokens to generate? (10-50): ") or "30")
            temp = float(input("Temperature? (0.1-2.0, default 0.8): ") or "0.8")
        except ValueError:
            print("Invalid input, using defaults (30 tokens, temp=0.8)")
            tokens = 30
            temp = 0.8
        
        print()
        generate_story_stepwise(prompt, model, tokenizer, tokens, temp)
        print("\n")


def demonstrate_concepts():
    """
    Pre-programmed demonstrations for classroom use
    """
    model, tokenizer = setup_model()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION 1: Basic Token Generation")
    print("=" * 70)
    generate_story_stepwise(
        "Once upon a time, there was a brave knight who",
        model, tokenizer, max_tokens=25, temperature=0.7
    )
    
    input("\nPress Enter to continue to next demonstration...")
    
    print("\n\n" + "=" * 70)
    print("DEMONSTRATION 2: Temperature Effects")
    print("=" * 70)
    generate_comparison(
        "The future of technology will be",
        model, tokenizer,
        temperatures=[0.3, 0.7, 1.2]
    )
    
    input("\nPress Enter to continue to interactive mode...")
    interactive_token_generation()


def simple_visualization():
    """
    A simpler version that just shows tokens being added without probabilities
    """
    model, tokenizer = setup_model()
    
    prompt = "In a galaxy far away"
    print(f"\nStarting prompt: \"{prompt}\"")
    print("\nGenerating story token by token:\n")
    print(prompt, end='', flush=True)
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        for i in range(40):
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :] / 0.8
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            new_text = tokenizer.decode(next_token[0])
            print(new_text, end='', flush=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            time.sleep(0.2)
            
            if new_text.strip().endswith('.'):
                break
    
    print("\n\nGeneration complete!")


if __name__ == "__main__":
    print("=" * 70)
    print("GPT2 TOKEN-BY-TOKEN STORY GENERATION DEMO")
    print("=" * 70)
    print("\nChoose a mode:")
    print("1. Full demonstration with explanations (learn about token generation)")
    print("2. Interactive mode (for experimentation)")
    print("3. Simple visualization (just watch tokens appear)")
    print("4. Interactive token choice (YOU pick each token!)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        demonstrate_concepts()
    elif choice == "2":
        interactive_token_generation()
    elif choice == "3":
        simple_visualization()
    elif choice == "4":
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
    else:
        print("Invalid choice. Running full demonstration...")
        demonstrate_concepts() 