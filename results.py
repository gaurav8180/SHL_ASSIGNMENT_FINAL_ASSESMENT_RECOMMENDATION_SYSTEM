"""
Prediction Script for SHL Assessment Recommendation System
Generates CSV output in required submission format
"""

import asyncio
import pandas as pd
import json
from main import recommend_assessments
from rich.console import Console
from rich.panel import Panel
import unicodedata
import re

console = Console()

def safe_for_console(text: str) -> str:
    """Normalize and sanitize text for Windows console rendering."""
    try:
        text = unicodedata.normalize("NFKC", str(text))
        return re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)
    except Exception:
        return str(text)

def load_test_queries(file_path: str = "Gen_AI Dataset (1).xlsx") -> list:
    """Load test queries from Excel file."""
    df = pd.read_excel(file_path)
    if 'Query' not in df.columns:
        raise ValueError("Input Excel file must have a 'Query' column.")
    
    queries = df['Query'].dropna().unique().tolist()
    console.print(f"[green]Loaded {len(queries)} unique test queries from Excel[/green]")
    return queries

async def generate_predictions(test_queries: list, top_k: int = 10) -> pd.DataFrame:
    """Generate recommendations for each query and return as DataFrame."""
    console.print(Panel("[bold cyan]Generating Predictions on Test Data[/bold cyan]", border_style="blue"))

    results = []
    for i, query in enumerate(test_queries, 1):
        query_norm = unicodedata.normalize("NFKC", str(query))
        console.print(f"\n[cyan]Processing Query {i}/{len(test_queries)}[/cyan]")
        console.print(f"[dim]{safe_for_console(query_norm[:100])}...[/dim]")

        # Add delay between queries to avoid rate limiting
        if i > 1:
            await asyncio.sleep(3)
        
        try:
            recommendations = await recommend_assessments(query_norm)
            top_recommendations = [rec.get('url', '') for rec in recommendations[:top_k] if rec.get('url')]
            
            for url in top_recommendations:
                results.append({
                    'Query': query,
                    'Assessment_url': url
                })
            
            console.print(f"[green]Generated {len(top_recommendations)} recommendations[/green]")
        
        except Exception as e:
            console.print(f"[bold red]Error generating predictions for query '{query}': {e}[/bold red]")
            # Append empty row if prediction fails
            results.append({
                'Query': query,
                'Assessment_url': ''
            })
    
    return pd.DataFrame(results)

async def main():
    """Main function to generate and save predictions."""
    console.print(Panel("[bold blue]SHL Assessment Recommendation System - Prediction Generator[/bold blue]", border_style="blue"))
    
    # Load test data
    test_queries = load_test_queries("Gen_AI Dataset (1).xlsx")
    
    # Generate predictions
    prediction_df = await generate_predictions(test_queries, top_k=10)
    
    # Save CSV in required submission format
    output_path = "submission_predictions.csv"
    prediction_df.to_csv(output_path, index=False)
    
    console.print(f"\n[bold green]Predictions saved successfully to {output_path}[/bold green]")
    console.print(f"[yellow]File format: Query, Assessment_url (matches Appendix 3 requirements)[/yellow]")

if __name__ == "__main__":
    asyncio.run(main())
