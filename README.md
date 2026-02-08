# Document Q&A AI Agent (Gemini)

A lightweight document Q&A agent that ingests PDFs, extracts structured content, and answers user queries with citations. It includes a Streamlit UI and Arxiv lookup tool.

## Features
- Multi-PDF ingestion with text, tables, and optional vision-based extraction for figures/equations.
- Retrieval-augmented Q&A with citations and context window controls.
- Direct lookup, summarization, and metric extraction queries.
- Optional Arxiv API tool that can be triggered by the agent.

## Architecture (high level)
1. **Ingestion**: Parse PDFs, collect per-page content and tables, and (optionally) call Gemini Vision for figures/equations.
2. **Chunking + Embeddings**: Chunk text and create embeddings using the Gemini embedding model.
3. **Retrieval**: Cosine similarity search to build a compact context window.
4. **Answering**: Gemini text model answers using only the retrieved context and returns citations.

## Setup
1. Create a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables:
   - `GEMINI_API_KEY` (required)
   - Optional overrides:
     - `GEMINI_MODEL_TEXT` (default: `gemini-1.5-flash`)
     - `GEMINI_MODEL_VISION` (default: `gemini-1.5-flash`)
     - `GEMINI_MODEL_EMBED` (default: `models/text-embedding-004`)
     - `USE_VISION` (default: `false`)

## Run (Streamlit)
```bash
streamlit run src/app.py
```

## Example Queries
- "What is the conclusion of Paper X?"
- "Summarize the methodology of Paper C."
- "What are the accuracy and F1-score reported in Paper D?"
- "Find a paper about contrastive learning for time series on arxiv."

## Design Notes (my own choices)
- I kept the ingestion and retrieval pipeline modular to allow future swap-in of a vector DB.
- I added a simple tool-routing step so the agent can decide when Arxiv is relevant.
- I cap the context window by characters and include citations to reduce hallucinations.


## Security and Enterprise Considerations
- Secrets are injected via environment variables only.
- No PDF content is logged outside of local files.
- The agent is instructed to answer only from retrieved context and to admit gaps.

## Repo Contents
- `src/app.py`: Streamlit UI
- `src/ingestion`: PDF parsing and vision enhancement
- `src/retrieval`: Embeddings and vector search
- `src/agent`: Tool routing and Q&A prompts
- `src/tools`: Arxiv API helper
