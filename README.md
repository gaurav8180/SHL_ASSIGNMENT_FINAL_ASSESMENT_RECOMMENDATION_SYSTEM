# SHL Assessment Recommendation System

A comprehensive AI-powered recommendation system that matches job descriptions to relevant SHL assessments using advanced RAG (Retrieval-Augmented Generation) techniques, vector search, and LLM-based reranking.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

# SHL Assessment Recommendation System


# Live Deployment

**API Endpoint:** https://shl-backend-xvq9.onrender.com/docs#/default/recommend_recommend_post

**Frontend URL:** https://shl-frontend-wv3d.onrender.com/

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technical Stack](#technical-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The SHL Assessment Recommendation System is designed to help organizations quickly identify the most relevant SHL assessments for specific job roles. By analyzing job descriptions using natural language processing and semantic search, the system provides 5-10 targeted assessment recommendations complete with detailed metadata.

### Key Capabilities

- **Intelligent Query Understanding**: Extracts key information from job descriptions including role, skills, preferences, and required test types
- **Hybrid Search**: Combines dense vector search (semantic) and sparse retrieval (BM25) for optimal recall
- **LLM-Powered Reranking**: Uses Google's Gemini 2.5 Flash for context-aware reranking of results
- **Query Expansion**: Automatically enriches queries with synonyms and related terms
- **Adaptive Chunking**: Smart document segmentation for improved retrieval accuracy

## ✨ Features

- 🔍 **Multi-View Document Embedding**: Creates multiple embeddings per assessment (full text, title, description, signals)
- 🧠 **LLM-Based Query Analysis**: Extracts structured information from free-form job descriptions
- 🔄 **Reciprocal Rank Fusion (RRF)**: Intelligently merges results from multiple retrieval strategies
- 📊 **Comprehensive Metadata**: Includes duration, test types, remote testing support, and IRT/adaptive capabilities
- 🌐 **Web Interface**: User-friendly Streamlit UI for easy interaction
- 🚀 **REST API**: FastAPI backend for programmatic access
- 📈 **Evaluation Framework**: Built-in metrics and testing utilities

## 🏗️ Architecture

### System Flow

```
Job Description Input
        ↓
Query Analysis (LLM) → Extract: role, skills, preferences, duration, test_types
        ↓
Query Expansion (LLM) → Add synonyms & related terms
        ↓
Parallel Retrieval:
    ├─ Dense Search (Qdrant Vector DB + MMR)
    └─ Sparse Search (BM25)
        ↓
Hybrid Fusion (RRF)
        ↓
LLM Reranking (Top 50 → Top 10)
        ↓
Final Recommendations
```

### Technical Architecture

```
┌─────────────────────────────────────────────────┐
│              Frontend (Streamlit)                │
│  - Job Description Input                         │
│  - URL Parsing                                   │
│  - Results Display                               │
└─────────────────┬───────────────────────────────┘
                  │ HTTP
┌─────────────────┴───────────────────────────────┐
│           Backend (FastAPI)                      │
│  - /recommend endpoint                           │
│  - Request validation                            │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────┐
│         Recommendation Pipeline (main.py)        │
│  ┌─────────────────────────────────────────┐   │
│  │ 1. Query Analysis                        │   │
│  │    - Extract structured info (LLM)      │   │
│  │    - Generate expansion terms (LLM)     │   │
│  └──────────────┬──────────────────────────┘   │
│  ┌──────────────┴──────────────────────────┐   │
│  │ 2. Hybrid Retrieval                      │   │
│  │    - Dense: Qdrant (MMR, k=40)          │   │
│  │    - Sparse: BM25 (k=80)                │   │
│  └──────────────┬──────────────────────────┘   │
│  ┌──────────────┴──────────────────────────┐   │
│  │ 3. Fusion & Reranking                    │   │
│  │    - RRF merge (top 60)                  │   │
│  │    - LLM rerank (top 10)                │   │
│  └──────────────┬──────────────────────────┘   │
└─────────────────┴───────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────┐
│              Data Layer                          │
│  ┌─────────────────────────────────────────┐   │
│  │ Qdrant Vector Database                   │   │
│  │  - Multi-view embeddings                 │   │
│  │  - HNSW index (m=64, ef=512)            │   │
│  │  - Cosine similarity                     │   │
│  └─────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────┐   │
│  │ Google Gemini 2.5 Flash                 │   │
│  │  - Query analysis                        │   │
│  │  - Query expansion                       │   │
│  │  - Result reranking                      │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## 🛠️ Technical Stack

### Backend
- **FastAPI**: High-performance async web framework
- **LangChain**: LLM orchestration and RAG pipeline
- **LangGraph**: State machine for complex workflows
- **Google Generative AI**: Gemini 2.5 Flash for LLM operations
- **Qdrant**: Vector database for semantic search
- **Sentence Transformers**: Text embeddings (text-embedding-004)

### Frontend
- **Streamlit**: Interactive web interface
- **BeautifulSoup4**: Web scraping for URL-based job descriptions
- **Pandas**: Data manipulation and presentation

### Data Processing
- **RecursiveCharacterTextSplitter**: Adaptive document chunking
- **BM25**: Sparse retrieval (keyword-based search)
- **MMR (Maximal Marginal Relevance)**: Diversity in dense search results

### DevOps
- **Uvicorn**: ASGI server
- **python-dotenv**: Environment configuration
- **Rich**: Enhanced console output

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- Anaconda/Miniconda (recommended)
- Google API Key (for Gemini)
- Qdrant Cloud account (or local Qdrant instance)

### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/shl-recommendation-system.git
cd shl-recommendation-system
```

2. **Create and activate virtual environment**
```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using conda
conda create -n shl-recommender python=3.10
conda activate shl-recommender
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your credentials
```

Required environment variables:
- `GOOGLE_API_KEY`: Your Google Generative AI API key
- `QDRANT_URL`: Your Qdrant instance URL
- `QDRANT_API_KEY`: Your Qdrant API key

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
QDRANT_URL=your_qdrant_url.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
```

### Qdrant Configuration

The system automatically:
- Creates the collection if it doesn't exist
- Configures HNSW indexing (m=64, ef_construct=512)
- Uses cosine similarity for vector comparison
- Validates vector dimensions (768 for text-embedding-004)

### Model Configuration

Default models (configurable in `main.py`):
- **LLM**: `gemini-2.5-flash` (temperature=0)
- **Embeddings**: `models/text-embedding-004`
- **Vector dimension**: 768

## 🚀 Usage

### Running the Backend API

```bash
# Development mode
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Running the Streamlit UI

```bash
streamlit run ui/app.py
```

The UI will be available at `http://localhost:8501`

### Using the API Directly

```python
import requests

response = requests.post(
    "http://localhost:8000/recommend",
    json={
        "job_description": "Looking for a Java developer with Spring Boot experience"
    }
)

recommendations = response.json()["recommendations"]
for rec in recommendations:
    print(f"{rec['name']}: {rec['url']}")
```

### Command Line Testing

```bash
python main.py
```

This runs a test query defined in the `if __name__ == "__main__"` block.

## 📚 API Documentation

### POST /recommend

Recommends SHL assessments based on a job description.

**Request Body:**
```json
{
  "job_description": "string"
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/solutions/products/...",
      "remote_testing_support": "Yes",
      "adaptive_irt_support": "No",
      "duration": "18 minutes",
      "test_types": ["Knowledge"]
    }
  ]
}
```

**Status Codes:**
- `200`: Success
- `400`: Invalid request
- `500`: Server error

### Interactive API Documentation

FastAPI provides automatic API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📊 Evaluation

### Running Evaluations

```bash
python eval/train_and_evaluate.py
```

This script:
1. Loads labeled training data from `Gen_AI Dataset (1).xlsx`
2. Runs recommendations for each query
3. Calculates Recall@10 metrics
4. Generates detailed evaluation reports
5. Saves results to `evaluation_results.json`

### Metrics

- **Recall@K**: Percentage of relevant assessments found in top K recommendations
- **Mean Recall@10**: Average recall across all test queries
- **Per-Query Analysis**: Individual performance breakdown

### Sample Output

```
╔══════════════════════════════════════╗
║      Evaluation Results              ║
╠══════════════════════════════════════╣
║ Mean Recall@10    │ 0.4111          ║
║ Number of Queries │ 10              ║
╚══════════════════════════════════════╝
```

## 📁 Project Structure

```
shl-recommendation-system/
│
├── main.py                      # Core recommendation pipeline
├── app.py                       # FastAPI application (root)
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (not in repo)
├── .env.example                # Template for environment variables
│
├── backend/
│   └── app.py                  # FastAPI application (alternate)
│
├── ui/
│   ├── app.py                  # Streamlit web interface
│   └── requirements.txt        # UI-specific dependencies
│
├── eval/
│   └── train_and_evaluate.py  # Evaluation framework
│
├── shl_assessments.json        # Assessment catalog (1000+ assessments)
├── Gen_AI Dataset (1).xlsx     # Labeled training data
├── evaluation_results.json     # Evaluation metrics output
│
├── .streamlit/
│   └── config.toml            # Streamlit theme configuration
│
├── .venv/                     # Virtual environment (not in repo)
│
└── README.md                  # This file
```

## 🔬 Technical Deep Dive

### Multi-View Document Embedding Strategy

Each SHL assessment is represented by 4 different embedding views:

1. **Full Document**: Complete assessment information
2. **Title Only**: High-signal assessment name
3. **Short Description**: First 300 characters
4. **Signals**: Condensed metadata (test types, duration, flags)

This approach improves recall by matching different aspects of queries to different document representations.

### Adaptive Chunking

Documents are split using two strategies:
- **Short chunks**: 320 chars, 80 overlap (for texts < 800 chars)
- **Long chunks**: 800 chars, 200 overlap (for longer texts)

This ensures optimal chunk size for both brief and detailed content.

### Hybrid Retrieval with RRF

The system combines:
- **Dense retrieval** (Qdrant): Semantic similarity using embeddings
- **Sparse retrieval** (BM25): Keyword matching

Results are merged using Reciprocal Rank Fusion:
```
RRF_score = 1 / (k + rank)
```

where `k=300` is a constant and `rank` is the position in each retriever's results.

### LLM-Based Query Enhancement

**Query Analysis**: Extracts structured information:
```json
{
  "role": "Java Developer",
  "skills": ["Java", "Spring Boot", "SQL"],
  "preferences": ["coding assessments", "adaptive"],
  "duration": "40 minutes",
  "test_types": ["Knowledge", "Coding"]
}
```

**Query Expansion**: Generates 10-20 related terms to improve recall without explicitly mentioning them in the query.

### Performance Optimizations

- **gRPC connection** to Qdrant for faster communication
- **MMR diversity** in dense search to avoid redundant results
- **Cached embeddings** for repeated queries (via Qdrant)
- **Async operations** throughout the pipeline
- **Rate limiting** protection with delays between evaluation queries

