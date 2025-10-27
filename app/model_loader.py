"""
This helps to pre-load the models we will use for
this programming lab. That way, you don't have
to wait for them to load when you are going through
the exercise.
"""


from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from transformers import AutoTokenizer, AutoModel



# Load GPT2 model and tokenizer
print("Loading GPT2 model... (this may take a moment)")
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()
print("Model loaded!\n")

# Load embedding model
print("Loading embedding model... (this may take a moment)")
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
print("Model loaded!\n")