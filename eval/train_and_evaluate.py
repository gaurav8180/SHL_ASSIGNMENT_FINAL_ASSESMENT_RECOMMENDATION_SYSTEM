"""
Training and Evaluation Script for SHL Assessment Recommendation System
Uses labeled training data to improve model performance and evaluate results
"""
import asyncio
import json
import pandas as pd
from typing import List, Dict, Any
from main import recommend_assessments
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import re
import unicodedata

console = Console()

def safe_for_console(text: str) -> str:
    """Normalize and sanitize text for Windows console rendering."""
    try:
        text = unicodedata.normalize("NFKC", str(text))
        return re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)
    except Exception:
        return str(text)

def load_training_data(file_path: str = "Gen_AI Dataset (1).xlsx") -> Dict[str, List[str]]:
    """Load labeled training data from Excel file.
    
    Returns:
        Dictionary mapping queries to list of assessment URLs
    """
    df = pd.read_excel(file_path)
    training_data = {}
    
    for _, row in df.iterrows():
        query = row['Query']
        assessment_url = row['Assessment_url']
        
        if query not in training_data:
            training_data[query] = []
        training_data[query].append(assessment_url)
    
    console.print(f"[green]Loaded {len(training_data)} training queries with labeled assessments[/green]")
    return training_data

def load_assessments_catalog(file_path: str = "shl_assessments.json") -> Dict[str, Dict[str, Any]]:
    """Load SHL assessments catalog and create URL to assessment mapping.
    
    Returns:
        Dictionary mapping assessment URL to assessment data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        assessments = json.load(f)
    
    url_to_assessment = {}
    for assessment in assessments:
        url = assessment.get('url', '')
        if url:
            url_to_assessment[url] = assessment
    
    console.print(f"[green]Loaded {len(url_to_assessment)} assessments from catalog[/green]")
    return url_to_assessment

def calculate_recall_at_k(recommended_urls: List[str], relevant_urls: List[str], k: int = 10) -> float:
    """Calculate Recall@K metric.
    
    Args:
        recommended_urls: List of recommended assessment URLs
        relevant_urls: List of relevant (ground truth) assessment URLs
        k: Number of top recommendations to consider
    
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not relevant_urls:
        return 0.0
    
    # Normalize URLs for comparison (remove trailing slashes, convert to lowercase, handle variations)
    def normalize_url(url):
        if not url:
            return ""
        # Remove trailing slashes, convert to lowercase
        url = url.rstrip('/').lower()
        # Handle URL variations (with/without www, http/https)
        # Remove protocol
        if url.startswith('https://'):
            url = url[8:]
        elif url.startswith('http://'):
            url = url[7:]
        # Remove www if present
        if url.startswith('www.'):
            url = url[4:]
        # Normalize URL encoding (e.g., %28 to (, %29 to ))
        import urllib.parse
        try:
            url = urllib.parse.unquote(url)
        except:
            pass
        # Remove trailing slash again after normalization
        url = url.rstrip('/')
        return url
    
    recommended_set = {normalize_url(url) for url in recommended_urls[:k] if url}
    relevant_set = {normalize_url(url) for url in relevant_urls if url}
    
    # Count how many relevant items are in top K recommendations
    relevant_retrieved = len(recommended_set.intersection(relevant_set))
    
    recall = relevant_retrieved / len(relevant_set) if relevant_set else 0.0
    return recall

async def evaluate_on_training_data(training_data: Dict[str, List[str]], 
                                   url_to_assessment: Dict[str, Dict[str, Any]],
                                   k: int = 10) -> Dict[str, Any]:
    """Evaluate the recommendation system on training data.
    
    Args:
        training_data: Dictionary mapping queries to relevant assessment URLs
        url_to_assessment: Dictionary mapping URLs to assessment data
        k: Number of recommendations to consider for Recall@K
    
    Returns:
        Dictionary containing evaluation metrics and detailed results
    """
    console.print(Panel("[bold cyan]Starting Evaluation on Training Data[/bold cyan]", border_style="blue"))
    
    results = []
    all_recalls = []
    
    for i, (query, relevant_urls) in enumerate(training_data.items(), 1):
        query_norm = unicodedata.normalize("NFKC", str(query))
        console.print(f"\n[cyan]Evaluating Query {i}/{len(training_data)}[/cyan]")
        console.print(f"[dim]{safe_for_console(query_norm[:100])}...[/dim]")
        
        # Add delay between queries to avoid rate limiting (except for first query)
        if i > 1:
            await asyncio.sleep(5)  # Wait 5 seconds between queries
        
        try:
            # Get recommendations
            recommendations = await recommend_assessments(query_norm)
            
            # Extract URLs from recommendations
            recommended_urls = [rec.get('url', '') for rec in recommendations if rec.get('url')]
            
            # Calculate Recall@K
            recall = calculate_recall_at_k(recommended_urls, relevant_urls, k)
            all_recalls.append(recall)
            
            # Find which recommended assessments are relevant
            recommended_set = {url.rstrip('/').lower() for url in recommended_urls[:k]}
            relevant_set = {url.rstrip('/').lower() for url in relevant_urls}
            relevant_found = recommended_set.intersection(relevant_set)
            
            results.append({
                'query': query,
                'num_relevant': len(relevant_urls),
                'num_recommended': len(recommended_urls),
                'num_relevant_found': len(relevant_found),
                'recall_at_k': recall,
                'relevant_urls': relevant_urls,
                'recommended_urls': recommended_urls[:k],
                'relevant_found': list(relevant_found)
            })
            
            console.print(f"[green]Recall@{k}: {recall:.3f} ({len(relevant_found)}/{len(relevant_urls)} relevant found)[/green]")
            
        except Exception as e:
            console.print(f"[bold red]Error evaluating query: {e}[/bold red]")
            all_recalls.append(0.0)
            results.append({
                'query': query,
                'num_relevant': len(relevant_urls),
                'num_recommended': 0,
                'num_relevant_found': 0,
                'recall_at_k': 0.0,
                'error': str(e)
            })
    
    # Calculate mean recall
    mean_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
    
    return {
        'mean_recall_at_k': mean_recall,
        'individual_recalls': all_recalls,
        'detailed_results': results,
        'num_queries': len(training_data)
    }

def print_evaluation_results(evaluation_results: Dict[str, Any]):
    """Print evaluation results in a formatted table."""
    console.print("\n")
    console.print(Panel("[bold green]Evaluation Results[/bold green]", border_style="green"))
    
    # Summary table
    table = Table(title="Summary Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Mean Recall@10", f"{evaluation_results['mean_recall_at_k']:.4f}")
    table.add_row("Number of Queries", str(evaluation_results['num_queries']))
    
    console.print(table)
    
    # Detailed results table
    detailed_table = Table(title="Detailed Results per Query")
    detailed_table.add_column("Query #", style="cyan")
    detailed_table.add_column("Relevant", style="yellow")
    detailed_table.add_column("Recommended", style="yellow")
    detailed_table.add_column("Found", style="yellow")
    detailed_table.add_column("Recall@10", style="green")
    
    for i, result in enumerate(evaluation_results['detailed_results'], 1):
        query_preview = result['query'][:50] + "..." if len(result['query']) > 50 else result['query']
        detailed_table.add_row(
            str(i),
            str(result.get('num_relevant', 0)),
            str(result.get('num_recommended', 0)),
            str(result.get('num_relevant_found', 0)),
            f"{result.get('recall_at_k', 0.0):.3f}"
        )
    
    console.print(detailed_table)
    
    # Print queries with low recall for analysis
    low_recall_queries = [
        (i, r) for i, r in enumerate(evaluation_results['detailed_results'], 1)
        if r.get('recall_at_k', 0.0) < 0.5
    ]
    
    if low_recall_queries:
        console.print("\n[bold yellow]Queries with Recall@10 < 0.5 (needs improvement):[/bold yellow]")
        for i, result in low_recall_queries:
            console.print(f"\n[red]Query {i}:[/red] {result['query'][:150]}...")
            console.print(f"  Recall@10: {result.get('recall_at_k', 0.0):.3f}")

async def main():
    """Main function to run training and evaluation."""
    console.print(Panel("[bold blue]SHL Assessment Recommendation System - Training & Evaluation[/bold blue]", 
                       border_style="blue"))
    
    # Load data
    training_data = load_training_data()
    url_to_assessment = load_assessments_catalog()
    
    # Evaluate on training data
    evaluation_results = await evaluate_on_training_data(training_data, url_to_assessment, k=10)
    
    # Print results
    print_evaluation_results(evaluation_results)
    
    # Save results to file
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[green]Evaluation results saved to evaluation_results.json[/green]")
    console.print(f"[bold green]Mean Recall@10: {evaluation_results['mean_recall_at_k']:.4f}[/bold green]")

if __name__ == "__main__":
    asyncio.run(main())

