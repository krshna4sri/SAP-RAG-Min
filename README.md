# SAP RAG Minimal

*A lightweight Retrieval-Augmented Generation (RAG) tool for SAP
documents and business processes*

![Python](https://img.shields.io/badge/python-3.10%2B-blue)\
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)\
![License](https://img.shields.io/badge/license-MIT-lightgrey)

------------------------------------------------------------------------

## 🚀 Overview

**SAP RAG Minimal** is a simple yet production-minded tool that allows
you to query your **SAP documentation and business process models**
using natural language. It combines dense embeddings with keyword
retrieval to provide accurate answers --- always with **citations** back
to the original documents.

### ✨ Features

-   📄 **Multi-format ingestion**: PDF, DOCX, PPTX, TXT/MD, and BPMN XML
    (Signavio/ARIS/SolMan exports).\
-   🔍 **Hybrid retrieval**: Embeddings (FAISS) + BM25 keyword search.\
-   📑 **Process-aware parsing**: SAP T-codes, module synonyms, BPMN
    process flows.\
-   🤖 **Answer generation**:
    -   OpenAI (if `OPENAI_API_KEY` set)\
    -   Ollama local models (`OLLAMA_MODEL`)\
    -   Fallback extractive QA (works offline)\
-   ⚡ **Interfaces**: FastAPI endpoints + CLI.\
-   📝 **Source citing**: Each answer includes doc path, page/slide, and
    preview snippet.

------------------------------------------------------------------------

## 📦 Installation

``` bash
git clone https://github.com/your-org/sap-rag-minimal.git
cd sap-rag-minimal

# Install dependencies
pip install -U fastapi uvicorn[standard] pydantic
pip install -U sentence-transformers faiss-cpu numpy rank-bm25
pip install -U pypdf python-docx python-pptx lxml
pip install -U transformers torch --extra-index-url https://download.pytorch.org/whl/cpu

# Optional (for OpenAI / Ollama support)
pip install -U openai httpx
```

------------------------------------------------------------------------

## ⚙️ Configuration

Environment variables (optional):

  ---------------------------------------------------------------------------
  Variable            Default                                  Description
  ------------------- ---------------------------------------- --------------
  `INDEX_DIR`         `.rag_index`                             Where the
                                                               index is
                                                               stored

  `EMBEDDING_MODEL`   `BAAI/bge-small-en-v1.5`                 Sentence
                                                               transformer
                                                               for embeddings

  `CROSS_ENCODER`     `cross-encoder/ms-marco-MiniLM-L-6-v2`   Optional
                                                               reranker

  `OPENAI_API_KEY`    *(unset)*                                Enables OpenAI
                                                               answers

  `OPENAI_MODEL`      `gpt-4o-mini`                            OpenAI model
                                                               to use

  `OLLAMA_MODEL`      *(unset)*                                Local Ollama
                                                               model name
                                                               (e.g.,
                                                               `llama3.1`)

  `OLLAMA_BASE_URL`   `http://localhost:11434`                 Ollama server
                                                               URL
  ---------------------------------------------------------------------------

------------------------------------------------------------------------

## 🚦 Usage

### Start API

``` bash
uvicorn sap_rag_minimal:app --reload --port 8000
```

### Ingest Documents

``` bash
curl -X POST "http://localhost:8000/ingest?root=/absolute/path/to/your/docs"
```

### Ask Questions

``` bash
curl "http://localhost:8000/query?q=How do we post an incoming invoice in SAP S/4HANA?"
```

------------------------------------------------------------------------

## 💻 CLI Mode

``` bash
# Ingest
python sap_rag_minimal.py ingest /absolute/path/to/your/docs

# Ask a question
python sap_rag_minimal.py ask "What is the T-code for vendor invoice posting?"
```

------------------------------------------------------------------------

## 🔮 Roadmap

-   [ ] Add connectors for SharePoint, SAP SolMan, and process mining
    tools.\
-   [ ] Support multilingual SAP documents (DE, ES, FR).\
-   [ ] Replace FAISS with production-grade vector DBs (pgvector,
    Qdrant, Milvus).\
-   [ ] Add evaluation harness (precision@k, MRR, answer quality).

------------------------------------------------------------------------

## 🤝 Contributing

Pull requests and issues are welcome! Please fork the repo and open a PR
with improvements.

------------------------------------------------------------------------

## 📜 License

MIT License © 2025
