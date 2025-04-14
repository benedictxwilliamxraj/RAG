# Local RAG System using Ollama, LlamaIndex, and ChromaDB

This project builds a local Retrieval-Augmented Generation (RAG) system. It uses `llama-index` and `chromadb` with Ollama as the backend LLM. 
You can query your own documents using a local model with no internet or API key required.

## Features

- Create a persistent vector store from local PDFs
- Query documents using natural language
- Works fully offline with Ollama
- Fast re-use with separate create/run scripts

Make sure Ollama is installed and running:
- Under bash
- `ollama run llama3`


# How to Use
## Create the RAG (run once)
Place your PDFs inside the data/ folder.
  - `python ragcreate.py`
## Run the query script:
  - `python ragrun.py`
- It loads the saved index and lets you ask questions. You can change the question inside 

# You can ask things like
Then ask things like:


“What was the profit after tax in FY2024?”

“How much coal was produced last year?”
