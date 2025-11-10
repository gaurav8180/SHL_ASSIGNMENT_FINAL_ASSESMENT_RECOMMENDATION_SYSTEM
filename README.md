Perfect ✅ — here’s your **complete updated `README.md` file**, now including your **live API endpoint** and **frontend URL** section near the top (right after the badges, before the Table of Contents).

You can copy this version directly into your repo (or I can generate it as a downloadable file if you prefer).

---

```markdown
# SHL Assessment Recommendation System

A comprehensive AI-powered recommendation system that matches job descriptions to relevant SHL assessments using advanced RAG (Retrieval-Augmented Generation) techniques, vector search, and LLM-based reranking.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 🌐 Live Deployment

- **API Endpoint:** [https://shl-backend-xvq9.onrender.com/docs#/default/recommend_recommend_post](https://shl-backend-xvq9.onrender.com/docs#/default/recommend_recommend_post)  
- **Frontend URL:** [https://shl-frontend-wv3d.onrender.com/](https://shl-frontend-wv3d.onrender.com/)

---

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

The SHL Assessment Recommendation System is designed to help organizations quickly identify the most relevant SHL assessments for specific job roles. By analyzing job descriptions using natural language processing and semantic search, the system provides 5–10 targeted assessment recommendations complete with detailed metadata.

### Key Capabilities

- **Intelligent Query Understanding:** Extracts key information from job descriptions including role, skills, preferences, and required test types  
- **Hybrid Search:** Combines dense vector search (semantic) and sparse retrieval (BM25) for optimal recall  
- **LLM-Powered Reranking:** Uses Google’s Gemini 2.5 Flash for context-aware reranking of results  
- **Query Expansion:** Automatically enriches queries with synonyms and related terms  
- **Adaptive Chunking:** Smart document segmentation for improved retrieval accuracy  

## ✨ Features

- 🔍 **Multi-View Document Embedding:** Creates multiple embeddings per assessment (full text, title, description, signals)  
- 🧠 **LLM-Based Query Analysis:** Extracts structured information from free-form job descriptions  
- 🔄 **Reciprocal Rank Fusion (RRF):** Intelligently merges results from multiple retrieval strategies  
- 📊 **Comprehensive Metadata:** Includes duration, test types, remote testing support, and IRT/adaptive capabilities  
- 🌐 **Web Interface:** User-friendly Streamlit UI for easy interaction  
- 🚀 **REST API:** FastAPI backend for programmatic access  
- 📈 **Evaluation Framework:** Built-in metrics and testing utilities  

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

````

## 🛠️ Technical Stack

### Backend
- **FastAPI:** High-performance async web framework  
- **LangChain:** LLM orchestration and RAG pipeline  
- **LangGraph:** State machine for complex workflows  
- **Google Generative AI:** Gemini 2.5 Flash for LLM operations  
- **Qdrant:** Vector database for semantic search  
- **Sentence Transformers:** Text embeddings (`text-embedding-004`)  

### Frontend
- **Streamlit:** Interactive web interface  
- **BeautifulSoup4:** Web scraping for URL-based job descriptions  
- **Pandas:** Data manipulation and presentation  

### Data Processing
- **RecursiveCharacterTextSplitter:** Adaptive document chunking  
- **BM25:** Sparse retrieval (keyword-based search)  
- **MMR (Maximal Marginal Relevance):** Diversity in dense search results  

### DevOps
- **Uvicorn:** ASGI server  
- **python-dotenv:** Environment configuration  
- **Rich:** Enhanced console output  

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
````

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

* `GOOGLE_API_KEY`: Your Google Generative AI API key
* `QDRANT_URL`: Your Qdrant instance URL
* `QDRANT_API_KEY`: Your Qdrant API key

## ⚙️ Configuration

### Environment Variables

```env
GOOGLE_API_KEY=your_google_api_key_here
QDRANT_URL=your_qdrant_url.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
```

### Qdrant Configuration

* Auto-creates collection if not present
* Configures HNSW indexing (m=64, ef_construct=512)
* Uses cosine similarity
* Validates vector dimension (768 for `text-embedding-004`)

### Model Configuration

* **LLM:** `gemini-2.5-flash` (temperature=0)
* **Embeddings:** `models/text-embedding-004`
* **Vector dimension:** 768

## 🚀 Usage

### Run Backend API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Run Streamlit UI

```bash
streamlit run ui/app.py
```

Frontend will be available at `http://localhost:8501`

### Test via API

```python
import requests

response = requests.post(
    "http://localhost:8000/recommend",
    json={"job_description": "Looking for a Java developer with Spring Boot experience"}
)

for rec in response.json()["recommendations"]:
    print(f"{rec['name']}: {rec['url']}")
```

## 📚 API Documentation

### POST /recommend

**Request:**

```json
{ "job_description": "string" }
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

**Interactive Docs:**

* Swagger UI: [https://shl-backend-xvq9.onrender.com/docs](https://shl-backend-xvq9.onrender.com/docs)
* ReDoc: [https://shl-backend-xvq9.onrender.com/redoc](https://shl-backend-xvq9.onrender.com/redoc)

## 📊 Evaluation

Run:

```bash
python eval/train_and_evaluate.py
```

Outputs:

* Recall@10 = 0.4111
* Evaluation logs saved to `evaluation_results.json`

## 📁 Project Structure

```
shl-recommendation-system/
├── main.py
├── app.py
├── backend/
│   └── app.py
├── ui/
│   ├── app.py
│   └── requirements.txt
├── eval/train_and_evaluate.py
├── shl_assessments.json
├── Gen_AI Dataset (1).xlsx
├── evaluation_results.json
├── .streamlit/config.toml
└── README.md
```

## 🔬 Technical Deep Dive

### Multi-View Embeddings

* Full text
* Title
* Short description
* Metadata signals
  → Improves recall by matching different query aspects.

### Hybrid Retrieval

Combines **dense (semantic)** + **sparse (keyword)** using **RRF**:

```
RRF_score = 1 / (k + rank)
```

with k=300.

### Optimizations

* gRPC Qdrant connections
* MMR for diversity
* Cached embeddings
* Async parallel retrieval

---

**Author:** Gaurav Singh
**License:** MIT

```

---

Would you like me to generate this as a **ready-to-download `README.md` file**?
```
