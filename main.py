import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import grpc
from dotenv import load_dotenv
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from rich.console import Console
from rich.panel import Panel

from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph

# Load environment variables
load_dotenv()

console = Console()

# --- Environment Variable Loading and Validation ---
required_env_vars = [
    "GOOGLE_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    console.print(f"[bold red]Error: Missing required environment variables: {', '.join(missing_vars)}[/bold red]")
    raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

google_api_key = os.getenv("GOOGLE_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# --- LLM and Embedding Model Initialization ---
console.print("[cyan]Initializing LLM and Embedding Model...[/cyan]")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=google_api_key,
)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=google_api_key,
)
console.print("[green]LLM and Embedding Model Initialized.[/green]")

def safe_for_console(text: str) -> str:
    """Sanitize text for Windows console rendering."""
    try:
        return re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)
    except Exception:
        return text

SHL_FILE = "shl_assessments.json"
QDRANT_COLLECTION_NAME = "shl_assessments"
# Determine embedding dimensionality dynamically
try:
    _probe_vec = embedding_model.embed_query("probe")
    EXPECTED_VECTOR_SIZE = len(_probe_vec) if isinstance(_probe_vec, list) else 768
except Exception:
    EXPECTED_VECTOR_SIZE = 768

# --- Load SHL Data ---
console.print(f"[cyan]Loading SHL data from {SHL_FILE}...[/cyan]")
with open(SHL_FILE, "r", encoding="utf-8") as f:
    shl_data = json.load(f)
console.print(f"[green]Loaded {len(shl_data)} entries from {SHL_FILE}.[/green]")

def build_documents(entries: List[Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []
    for entry in entries:
        name = entry.get("name", "")
        description = entry.get("description", "")
        url = entry.get("url", "")
        test_types = entry.get("test_types", [])
        duration = entry.get("duration", "N/A")
        remote = entry.get("remote_testing_support", "Unknown")
        adaptive = entry.get("adaptive_irt_support", "Unknown")

        # Whole-document representation (helps recall)
        full_text = (
            f"{name}\n"
            f"{description}\n"
            f"Test Types: {', '.join(test_types)}\n"
            f"Duration: {duration}\n"
            f"Remote Testing Support: {remote}\n"
            f"Adaptive/IRT Support: {adaptive}\n"
            f"URL: {url}"
        )
        docs.append(Document(page_content=full_text, metadata={**entry, "view":"full"}))

        # Title-only doc (high-signal)
        if name:
            docs.append(Document(page_content=name, metadata={**entry, "view":"title"}))

        # Short description doc (first 300 chars)
        if description:
            short_desc = description.strip()[:300]
            if short_desc:
                docs.append(Document(page_content=short_desc, metadata={**entry, "view":"short_description"}))

        # Signals doc (skills/test_types/flags condensed)
        signals_parts = []
        if test_types:
            signals_parts.append("TestTypes: " + ", ".join(test_types))
        if duration and duration != "N/A":
            signals_parts.append(f"Duration: {duration}")
        if remote:
            signals_parts.append(f"Remote: {remote}")
        if adaptive:
            signals_parts.append(f"Adaptive: {adaptive}")
        signals_text = " | ".join(signals_parts)
        if signals_text:
            docs.append(Document(page_content=signals_text, metadata={**entry, "view":"signals"}))
    return docs

# --- Prepare Documents (multi-embedding views) ---
console.print("[cyan]Preparing LangChain documents (multi-view)...[/cyan]")
base_documents = build_documents(shl_data)
console.print(f"[green]Prepared {len(base_documents)} documents (multi-view).[/green]")

# --- Apply Adaptive Chunking ---
console.print("[cyan]Applying adaptive chunking to documents for improved recall...[/cyan]")
short_splitter = RecursiveCharacterTextSplitter(chunk_size=320, chunk_overlap=80, separators=["\n\n", "\n", " ", ""])
long_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
chunked_documents: List[Document] = []
for d in base_documents:
    text = d.page_content or ""
    splitter = short_splitter if len(text) < 800 else long_splitter
    chunks = splitter.split_text(text)
    if not chunks:
        continue
    for c in chunks:
        chunked_documents.append(Document(page_content=c, metadata=d.metadata))
console.print(f"[green]Generated {len(chunked_documents)} document chunks (adaptive).[/green]")

# --- Qdrant Initialization (Optimized with Timing) ---
start_time_qdrant = time.perf_counter()
console.print(f"[cyan]Checking for Qdrant collection '{QDRANT_COLLECTION_NAME}' at {qdrant_url}...[/cyan]")
try:
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=True,
    )
    collection = client.get_collection(collection_name=QDRANT_COLLECTION_NAME)  # Check existence, inspect config
    # Handle both dict and attr-based response structures
    existing_vector_size = None
    try:
        if isinstance(collection, dict):
            vectors = collection.get("config", {}).get("params", {}).get("vectors")
            if isinstance(vectors, dict):
                existing_vector_size = vectors.get("size")
        else:
            cfg = getattr(collection, "config", None)
            params = getattr(cfg, "params", None) if cfg else None
            vectors = getattr(params, "vectors", None) if params else None
            existing_vector_size = getattr(vectors, "size", None) if vectors else None
    except Exception:
        existing_vector_size = None
    if existing_vector_size != EXPECTED_VECTOR_SIZE:
        console.print(f"[yellow]Existing collection vector size ({existing_vector_size}) "
                      f"does not match expected ({EXPECTED_VECTOR_SIZE}). Rebuilding collection...[/yellow]")
        from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=EXPECTED_VECTOR_SIZE, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=64, ef_construct=512),
        )
        start_time_create = time.perf_counter()
        vectorstore = QdrantVectorStore.from_documents(
            chunked_documents,
            embedding_model, 
            url=qdrant_url,
            prefer_grpc=True,
            api_key=qdrant_api_key,
            collection_name=QDRANT_COLLECTION_NAME,
        )
        end_time_create = time.perf_counter()
        console.print(f"[green]Recreated and populated collection in {end_time_create - start_time_create:.2f} seconds.[/green]")
        qdrant_op = "Recreated with correct vector size"
    else:
        console.print(f"[yellow]Found existing collection '{QDRANT_COLLECTION_NAME}'. Connecting...[/yellow]")
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION_NAME,
            embedding=embedding_model, 
        )
        qdrant_op = "Connected to existing"
except (UnexpectedResponse, ValueError, grpc._channel._InactiveRpcError) as e: 
    console.print(f"[yellow]Collection '{QDRANT_COLLECTION_NAME}' not found or connection error ({type(e).__name__}) ... Creating and populating...[/yellow]")
    start_time_create = time.perf_counter()
    from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=EXPECTED_VECTOR_SIZE, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=64, ef_construct=512),
    )
    vectorstore = QdrantVectorStore.from_documents(
        chunked_documents,
        embedding_model, 
        url=qdrant_url,
        prefer_grpc=True,
        api_key=qdrant_api_key,
        collection_name=QDRANT_COLLECTION_NAME,
    )
    end_time_create = time.perf_counter()
    console.print(f"[green]Created and populated collection in {end_time_create - start_time_create:.2f} seconds.[/green]")
    qdrant_op = "Created and populated new"

end_time_qdrant = time.perf_counter()
console.print(Panel(f"[green]Qdrant Setup Complete ({qdrant_op}).[/green]\nCollection: '{QDRANT_COLLECTION_NAME}'\nURL: {qdrant_url}\n[bold yellow]Total Time: {end_time_qdrant - start_time_qdrant:.2f} seconds[/bold yellow]", title="Vector Store Status", border_style="blue"))

# --- Retriever Initialization ---
console.print("[cyan]Initializing Retrievers...[/cyan]")
bm25_retriever = BM25Retriever.from_documents(chunked_documents)
bm25_retriever.k = 80

# Use MMR for dense retrieval and increase ef for better recall
dense_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 40,
        "fetch_k": 300,
        "lambda_mult": 0.0,
        "search_params": {"hnsw_ef": 1024},
    },
)

class HybridRetriever(BaseModel):
    dense: Any 
    sparse: Any

    async def invoke(self, query):
        # Get results from both retrievers asynchronously
        dense_task = self.dense.ainvoke(query)
        sparse_task = self.sparse.ainvoke(query)
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        # Reciprocal Rank Fusion (RRF) for robust hybrid merging
        k_fusion = 300
        rank_k = 300
        rrf_k = 300

        def rrf_score(rank: int, k: int = rrf_k) -> float:
            return 1.0 / (k + rank)

        id_key = lambda d: d.metadata.get('url') or d.metadata.get('name') or d.page_content[:50]
        scores: Dict[str, float] = {}
        id_to_doc: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_results[:rank_k], start=1):
            key = id_key(doc)
            id_to_doc[key] = doc
            scores[key] = scores.get(key, 0.0) + rrf_score(rank)
        for rank, doc in enumerate(sparse_results[:rank_k], start=1):
            key = id_key(doc)
            id_to_doc[key] = doc
            scores[key] = scores.get(key, 0.0) + rrf_score(rank)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        final_results: List[Document] = []
        for key, _ in ranked:
            final_results.append(id_to_doc[key])
            if len(final_results) >= 60:
                break
        return final_results

retriever = HybridRetriever(dense=dense_retriever, sparse=bm25_retriever)

query_prompt = PromptTemplate.from_template("""
Extract the following structured information from the job description below.

Fields to extract:
- role (job title or general function)
- skills (list of technologies, concepts, or traits expected)
- preferences (assessment-related preferences like adaptive, coding, remote etc.)
- duration (if mentioned)
- test_types (type of assessments expected like coding, numerical, etc.)

IMPORTANT: If the job description doesn't explicitly list all skills, preferences, or test types, infer them based on the role and industry standards. For example:
- For a "Java Developer" role, include skills like "Java", "Spring Boot", "SQL", etc. even if not explicitly mentioned
- For a "Data Scientist" role, include skills like "Python", "Machine Learning", "Statistics", etc.
- For technical roles, include preferences like "coding assessments" if appropriate
- For managerial roles, include skills like "Leadership", "Communication", etc.

Respond only in this format:
{{
    "role": "...",
    "skills": ["...", "..."],
    "preferences": ["...", "..."],
    "duration": "...",
    "test_types": ["...", "..."]
}}

Job description:
<job_description>
{job_description}
</job_description>

Important: When listing skills or preferences, identify multi-word technical terms (e.g., 'Java Script', 'SQL Server', 'Machine Learning') and keep them as single strings. Do not split them. Eg: 'Java Script' should be JavaScript and 'Machine Learning' should be MachineLearning
""")

expand_prompt = PromptTemplate.from_template("""
Given the following initial search facets for an assessment retrieval task:
Role: {role}
Skills: {skills}
Preferences: {preferences}
Test Types: {test_types}

Generate a compact list of 10-20 additional synonyms/related terms (single or multi-word) that improve recall for retrieval, covering frameworks, libraries, alternate names, and assessment keywords. Return as a JSON array of strings only.
""")

async def extract_query_info(state):
    query = state.input
    console.print(Panel(f"[cyan]Step 1: Extracting Info from Job Description:[/cyan]\n{safe_for_console(query)}", title="Workflow Step", border_style="blue"))
    prompt = query_prompt.format(job_description=query)
    
    start_time_llm_extract = time.perf_counter()
    response = (await llm.ainvoke(prompt)).content
    end_time_llm_extract = time.perf_counter()
    console.print(f"[magenta]LLM Info Extraction took: {end_time_llm_extract - start_time_llm_extract:.2f} seconds[/magenta]")
    
    response = re.sub(r"```json|```", "", response.strip())

    try:
        parsed = json.loads(response)
        console.print("[green]Successfully parsed LLM response for query info.[/green]")
    except Exception as e:
        console.print(f"[bold red]Failed to parse query info JSON:[/bold red] {e}")
        console.print(f"[red]Raw LLM Response:[/red]\n{response}")
        parsed = {"role": "", "skills": [], "preferences": [], "duration": "", "test_types": []}

    base_terms = [parsed.get('role', '')] + parsed.get("skills", []) + parsed.get("preferences", []) + parsed.get("test_types", [])
    query_str = " ".join(t for t in base_terms if t)
    # LLM-based query expansion
    try:
        exp_resp = (await llm.ainvoke(expand_prompt.format(
            role=parsed.get('role', ''),
            skills=", ".join(parsed.get('skills', [])),
            preferences=", ".join(parsed.get('preferences', [])),
            test_types=", ".join(parsed.get('test_types', [])),
        ))).content.strip()
        exp_resp = re.sub(r"```json|```", "", exp_resp)
        expanded = json.loads(exp_resp)
        if isinstance(expanded, list):
            query_str = f"{query_str} " + " ".join(expanded)
            console.print(f"[green]Added {len(expanded)} expansion terms to query.[/green]")
    except Exception as e:
        console.print(f"[yellow]Query expansion skipped due to parse error: {e}[/yellow]")

    console.print(f"[yellow]Generated Search Query:[/yellow] '{safe_for_console(query_str)}'")
    return {"query_info": query_str, "input": query}

async def perform_rag(state):
    query_info = state.query_info
    console.print(Panel(f"[cyan]Step 2: Performing Hybrid RAG for:[/cyan] '{safe_for_console(query_info)}'", title="Workflow Step", border_style="blue"))
    start_time_rag = time.perf_counter()
    # Use the async invoke method of HybridRetriever
    retrieved_docs = await retriever.invoke(query_info) 
    end_time_rag = time.perf_counter()
    console.print(f"[magenta]Hybrid Retrieval took: {end_time_rag - start_time_rag:.2f} seconds, got {len(retrieved_docs)} docs[/magenta]")
    return {"retrieved_docs": retrieved_docs, "query_info": query_info}

prompt_template = PromptTemplate.from_template("""
You are a helpful assistant tasked with selecting the most relevant SHL assessments for a given job.

Here is a summary of the job description:
{query}

Below are some SHL assessments (with their details):
{docs}

Please select the minimum 1 and maximum 10 most relevant assessments by their index numbers (Assessment 1, Assessment 2, etc.).

Respond ONLY with a JSON array of indices like:
```json
[1, 5, 8]
```

Do not include any other information in your response, just the array of indices.
""")

async def rerank_and_filter(state):
    query_info = state.query_info
    docs = state.retrieved_docs
    console.print(Panel(f"[cyan]Step 3: Re-ranking/Filtering {len(docs)} retrieved docs based on query:[/cyan] '{safe_for_console(query_info)}'", title="Workflow Step", border_style="blue"))

    # Optimized: Send only key info to LLM for re-ranking (cap at 50 docs)
    doc_strings = []
    for doc in docs[:50]:
        doc_strings.append(
            f"Name: {doc.metadata.get('name','')}\n"
            f"Description: {doc.metadata.get('description','')}\n"
            f"Test Types: {', '.join(doc.metadata.get('test_types', []))}\n"
            f"Duration: {doc.metadata.get('duration', 'N/A')}\n"
            f"Remote Testing Support: {doc.metadata.get('remote_testing_support', 'Unknown')}\n"
            f"Adaptive/IRT Support: {doc.metadata.get('adaptive_irt_support', 'Unknown')}"
        )

    doc_block = "\n\n".join([f"Assessment {i+1}:\n{s}" for i, s in enumerate(doc_strings)])

    prompt = prompt_template.format(query=query_info, docs=doc_block)
    
    start_time_llm_rerank = time.perf_counter()
    response = (await llm.ainvoke(prompt)).content.strip()
    end_time_llm_rerank = time.perf_counter()
    console.print(f"[magenta]LLM Re-ranking/Filtering took: {end_time_llm_rerank - start_time_llm_rerank:.2f} seconds[/magenta]")

    try:
        # Extract JSON array of indices
        match = re.search(r"```json\s*(\[.*?\])\s*```", response, re.DOTALL)
        if match:
            response = match.group(1).strip()
        if not match:
            match2 = re.search(r"(\[.*?\])", response, re.DOTALL)
            if match2:
                response = match2.group(1).strip()
        # Parse the indices
        indices = json.loads(response)
        if not isinstance(indices, list):
            raise json.JSONDecodeError("Not a list", response, 0)
    except json.JSONDecodeError as e:
        nums = re.findall(r"\d+", response)
        indices = [int(n) for n in nums][:10]
        console.print(f"[yellow]Parsed indices via fallback regex: {indices}[/yellow]")
    console.print(f"[green]Successfully parsed {len(indices)} assessment indices from LLM.[/green]")
    
    # Convert indices to actual assessment data
    recommendations = []
    for idx in indices:
        if 1 <= idx <= len(docs):
            doc = docs[idx-1]
            recommendations.append({
                "name": doc.metadata.get("name", ""),
                "url": doc.metadata.get("url", ""),
                "remote_testing_support": doc.metadata.get("remote_testing_support", "Unknown"),
                "adaptive_irt_support": doc.metadata.get("adaptive_irt_support", "Unknown"),
                "duration": doc.metadata.get("duration", "N/A"),
                "test_types": doc.metadata.get("test_types", [])
            })
        else:
            console.print(f"[yellow]Warning: Index {idx} is out of range (1-{len(docs)})[/yellow]")
    
    console.print(f"[green]Successfully converted {len(recommendations)} indices to full assessment data.[/green]")
    return {"final_recommendations": recommendations}

class GraphState(BaseModel):
    input: str
    query_info: Optional[str] = None
    retrieved_docs: Optional[List[Document]] = None
    final_recommendations: Optional[List[Dict[str, Any]]] = None

workflow = StateGraph(GraphState)
workflow.add_node("extract_info", extract_query_info)
workflow.add_node("rag", perform_rag)
workflow.add_node("filter", rerank_and_filter)

workflow.set_entry_point("extract_info")
workflow.add_edge("extract_info", "rag")
workflow.add_edge("rag", "filter")
workflow.set_finish_point("filter")

app = workflow.compile()

async def recommend_assessments(job_description: str
):
    console.print(Panel(f"[bold blue]Starting SHL Recommendation Workflow for Job Description:[/bold blue]\n{job_description}", title="Workflow Start", border_style="green"))
    start_time_workflow = time.perf_counter()
    # Use async invoke for the compiled graph
    result = await app.ainvoke({"input": job_description})
    end_time_workflow = time.perf_counter()
    console.print(Panel(f"[bold green]Workflow Complete.[/bold green]\n[bold yellow]Total Workflow Time: {end_time_workflow - start_time_workflow:.2f} seconds[/bold yellow]", title="Workflow End", border_style="green"))
    return result["final_recommendations"]

# --- Main Execution (Example) ---
if __name__ == "__main__":
    jd = """
    I am hiring for Java developers who can also collaborate effectively with my business teams. Looking
    for an assessment(s) that can be completed in 40 minutes
    """
    
    # Run the async function using asyncio.run
    final_recommendations = asyncio.run(recommend_assessments(jd))
    
    console.print(Panel("[bold blue]Final Recommendations:[/bold blue]", border_style="magenta"))
    console.print_json(data=final_recommendations)