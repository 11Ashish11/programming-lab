"""
Shared model loading for this lab.

Run this file once to download GPT-2 and the embedding model into the
local Hugging Face cache:

    python app/model_loader.py

Other scripts import setup_model() / setup_embedding_model() from here
so loading behavior stays in one place.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModel

GPT2_NAME = "gpt2"
EMBEDDING_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _from_pretrained(loader, name):
    """Prefer the local cache so we do not hit Hugging Face on every load."""
    try:
        return loader.from_pretrained(name, local_files_only=True)
    except OSError:
        return loader.from_pretrained(name)


def setup_model(*, verbose: bool = True):
    """Load GPT2 model and tokenizer."""
    if verbose:
        print("Loading GPT2 model... (this may take a moment)")
    model = _from_pretrained(GPT2LMHeadModel, GPT2_NAME)
    tokenizer = _from_pretrained(GPT2Tokenizer, GPT2_NAME)
    model.eval()
    if verbose:
        print("Model loaded!\n")
    return model, tokenizer


def setup_embedding_model(*, verbose: bool = True):
    """Load a small, fast embedding model."""
    if verbose:
        print("Loading embedding model... (this may take a moment)")
    tokenizer = _from_pretrained(AutoTokenizer, EMBEDDING_NAME)
    model = _from_pretrained(AutoModel, EMBEDDING_NAME)
    model.eval()
    if verbose:
        print("Model loaded!\n")
    return tokenizer, model


if __name__ == "__main__":
    setup_model()
    setup_embedding_model()
