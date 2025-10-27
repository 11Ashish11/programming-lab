# 🧠 Introduction: Working with Transformer Models

This programming lab introduces the fundamentals of **transformer models**, which are the foundational architecture behind modern **Large Language Models (LLMs)** such as _ChatGPT_, _Copilot_, and _Gemini_.

Transformers have revolutionized how computers process and understand language. They are used not only for text generation and translation but also for summarization, classification, question answering, and many other applications across language, vision, and audio.

---
## ⚙️ Running Models on Your Laptop

Training or running full LLMs (like ChatGPT) requires **GPUs** — specialized processors designed for parallel computation.

GPUs differ from the **CPU** (central processing unit) in your laptop because they can perform many small mathematical operations simultaneously, which is ideal for deep learning.

Since most students do not have access to high-end GPUs, this lab uses **smaller transformer models** that can run directly on a standard laptop CPU. These models are fast enough to provide hands-on experience while still demonstrating the key principles behind larger systems.

---
## 🧪 What You’ll Do in This Lab

In this lab, you will:

1. Learn how to **load and use pre-trained transformer models** from the [Hugging Face Transformers](https://huggingface.co/transformers) library.
2. Experiment with **token prediction** and **semantic similarity**.
3. Visualize and interpret model outputs to understand how predictions are made.
4. See how the same underlying technology used in ChatGPT can be applied to small, focused tasks on your own computer.

---
## 🏁 Getting started

Before you can dive into experimenting with transformer models, clone the lab's repository and do the usual to **set up**:

```bash
git clone git@github.com:enpm611/programming-lab.git
cd programming-lab
python3 -m venv env
source env/bin/activate # .\env\Scripts\activate for Windows
pip install -r requirements.txt
```

Next, open the project in VS Code. You will want to step through what the example code does

To speed things up when you are working on the exercises, it will be helpful to **download the models** ahead of time. For that, simply run the "Download Models" run configuration in VS Code or run `python app/model_loader.py`. That will take a little while to run.

Now you are all set!

---

# 1️⃣ Part 1 - Token prediction

At their core, transformer models are **token predictors**.  

A _token_ is a small unit of text — often a word, subword, or character — depending on how the text is tokenized.

When a model like ChatGPT generates text, it doesn’t produce entire sentences at once. Instead, it repeatedly predicts **the next most likely token** given the sequence of tokens it has already seen.

For example, suppose we give the model the following input:

```
It's cloudy. I think I need an
```

The model computes the probability of all possible next tokens (like “umbrella”, “idea”, “explanation”, etc.) and chooses the one with the highest likelihood:

```
It's cloudy. I think I need an umbrella
```

Then it repeats the process — predicting the next token after that — until it completes a coherent sentence or paragraph.

This simple iterative prediction mechanism is the basis of how ChatGPT generates everything from essays to code.

---
## ⌨️ Programming exercises - Token Prediction

Let's jump into coding. The following exercises will guide you through the topic of token prediction, starting very simple and becoming increasingly more sophisticated.

---
### One token, many tokens

Let's start with the basics. Open `app/tokens/1_token_exercise.py`, which contains several helper functions. 

:dart:`Exercise #1` Implement the function `complete_sentence()` that extends the `prompt` (the start of a sentence) with a `max_tokens` number of tokens, essentially producing a new sentence. The most important one for you is `generate_token()`, which given the beginning of a sentence, will produce the next token. Remember, a token can be a word, a punctuation marker or whitespace.

Adjust the number of `max_tokens` to see where the sentence goes with more tokens.

---
### Let's see 

In this exercise, we will run a basic token prediction mechanism and try to understand what it does. Open the script `app/tokens/2_basic_token_generator.py` and scroll to the bottom of the page where you'll find the entry point for that module. Run the model and watch on the console what it does.

:dart:`Exercise #2` **Change the `prompt`** to a different sentence and see what it generates. Does the model complete the sentence the way you expected?

---
### Let's turn up the heat

One of the most import parameters when working with transformer models is the **temperature**. The temperature controls the amount of randomness used in predicting the next token. The higher the temperature, the more random the output appears. This model has a minimum temperature of `0.2` and a maximum temperature of `2.0`. Change the temperature value that is part of the function call on the last line of the script and see how the generated sentences differ.

:dart:`Exercise #3` What happens with a **high temperature**? At what temperature do the predictions still make sense?

:dart:`Exercise #4` Increasing randomness also causes the model to generate different results for the same input. Start with a low temperature and increase it. Run the same sentence three times at each temperature. Do you see the **variance in output**?

:dart:`Exercise #5` Open the script `app/tokens/3_temperature.py`. It will simulate the variance of outputs with different inputs. How does the temperature the token generation when you **change up the prompt**?

---
### Keep it short

You will also find another parameter in the `complete_sentence()` function. The `max_tokens` parameter simply tells the models how many models maximum should be generated. This is important because the more tokens it generates, the more expensive the operation. Therefore, in a production setting, choosing the right `max_tokens` value is critical to balance cost with user experience.

:dart:`Exercise #6` Reduce the `max_tokens` to something **very small** and see what the model does.

:dart:`Exercise #7` In line `60`, we artificially stop generating more tokens when we see a period marking the end of a sentence. Remove or comment out those two lines and see how the output differs.

---
### ... probably?

Whenever the next token is generated, it is the token that the model predicted with the highest probability. But there are many other words that may or may not have been close candidates. The script `app/tokens/4_probabilities.py` shows three candidates and their probabilities for every generation step.

❓`Exercise #8` Run the script as is and follow the instructions. Do you see how the probabilities for some token candidates are pretty close in some cases? Do you also see that in other cases, there is a clear winner with a much higher probability than other candidates?

---
### Story telling

Let's take the previous example a bit further. So far, the model always chose the next token with the highest probability. 

:dart:`Exercise #9` The script `app/tokens/5_interactive.py` lets you tell your own story. After you enter the prompt and other parameters, it will show 5 candidate tokens with their probabilities. Enter a number from `1` to `5` to select the candidate you would like to continue the sentence with. Is the model able to tell a coherent story?

Note that when you change what token is selected, it completely change the trajectory of any further tokens. The model always relies on the last token (while also taking the rest of the sentence into account).

---

## 2️⃣ Part 2 - From Text to Vectors: How Transformers “Think”

Transformer models don’t actually operate directly on words or text tokens — they operate in **vector space**.

When we give a model some text, the first thing it does is convert each token (for example, “cloudy”, “umbrella”, “idea”) into a list of numbers, called an **embedding vector**. Each vector represents the meaning of that token in a high-dimensional space — often hundreds or even thousands of dimensions.

You can think of this vector space like a **map**:
- Tokens that are similar in meaning (like _“cat”_ and _“kitten”_) are positioned **close together** in that space.
- Tokens that are unrelated (like _“carburetor”_ and _“philosophy”_) are **far apart**.
    
This is somewhat similar to how GPS coordinates work. On a map, two locations that are physically close to each other have similar latitude and longitude values.  

In the same way, two words that have similar meanings will have **embedding vectors** with similar numerical coordinates. The distance (or angle) between two embedding vectors tells the model **how semantically related** those words are.

---
### 💡 Why This Matters

Understanding that transformers work in **vector space** explains why they can handle more than just text generation.  
The same mathematical structure lets them:

- Find **similar documents or sentences** (semantic search),
- Detect **duplicate issues** or related items (clustering),
- Analyze **sentiment** (positive vs. negative meaning), and
- Power **recommendation systems** or **code completion tools**.

Everything ultimately comes back to the geometry of these embedding vectors — the way meaning is represented and compared in high-dimensional space.


---
## ⌨️ Programming exercises - Embeddings

Now, let's see embeddings in action. Follow the instructions below and work on the highlighted exercises.

---
### Semantic similarities

Open the file `app/embeddings/1_basic_embeddings.py`. The bottom of the file specifies two sentences. The module calculates the embedding vectors and then compares those vectors. In essence, that script calculates how similar the sentences are in meaning.

:dart:`Exercise #10` **Change the input texts** and see how the similarity changes. 

:dart:`Exercise #11` One of the key strengths embeddings is that it allows finding matches between words that "look" different (syntactically) but are similar in meaning (semantically). Try the phrases `dog` and `cat`. What do you see? 

:dart:`Exercise #12` Stronger embedding model How much reasoning is the model able to do? Does it know that a `Chihuahua` is a kind of dog?

:dart:`Exercise #13` We have looked at nominal input so far. What about instances where there are **misspellings** in a sentence or other type of noise (e.g., Wikipedia style references)? How does that impact the quality of results.

---
### More similarities

Check out `app/embeddings/2_similarity.py`. It expands on the above example by comparing multiple sentence pairs. First, run the module and see how different types of sentence pairs generate different embeddings.

:dart:`Exercise #14` Change up the sentences to explore their similarity. What do you find surprising?

---
### Similarity applied to search

One of the primary applications of similarity calculations is found in vector search. Unlike traditional searches, it does not compare words but maps documents in vector space. It then checks what vector matches the query vector best. 

:dart:`Exercise #15` In this exercise, you will write your own code. You will implement a vector search function. Open `app/embeddings/3_search.py`. You will see that it has the scaffolding for semantic search. You just need to implement the missing pieces in the `execute_search()` function. As you implement semantic search, you want to take advantage of the following functions:
- `setup_embedding_model()` to set up your model
- `get_embedding()` to generate the embedding vectors
- `cosine_similarity()` to calculate the similarity between two embeddings in vector space

Simply implement the parts in `execute_search()` that are marked as `TODO` following the hints provided in the source code comments. When you run the module initially, you will see output like this:

```
Enter search query (or 'quit'): embeddings

Search results:

1. [0.000]
   Yoga and meditation help reduce stress and anxiety

2. [0.000]
   The Great Wall of China is visible from space

3. [0.000]
   Climate change is affecting global weather patterns
```

That is because no similarities have been calculated. When you are done, the result should be more sensible and looks something like this:

```
Enter search query (or 'quit'): embeddings

Search results:

1. [0.331]
   Deep learning uses neural networks with multiple layers

2. [0.144]
   Machine learning models can predict outcomes based on historical data

3. [0.071]
   The Eiffel Tower is a famous landmark in Paris, France
```

If you see this, you just implemented vector search!

---
## ⌨️ Have more time on your hands?

You can try the following modules:

- `app/tokens/6_token_generation_demonstration.py`: demonstrates various capabilities related to token predictions. Just follow the prompt when running the module
- `app/embeddings/4_embeddings_demonstration.py`: similar demonstration module for embeddings. Try playing some of the embeddings games.
