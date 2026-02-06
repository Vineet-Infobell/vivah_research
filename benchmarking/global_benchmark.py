"""
Global Benchmark Aggregator
---------------------------
Reads existing benchmark reports, aggregates scores across domains
(Profession, Education, Location), and identifies the top performing embedding models.

Version: 2.0
Improvements:
- **JSON support**: Prefers JSON files for 10x faster parsing
- Automatic fallback to HTML if JSON unavailable
- Robust column name matching (handles pandas HTML parsing variations)
- Validation for missing domain reports
- Enhanced error messages and debugging output
- Fixed indentation consistency
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime
import webbrowser

# Files to process (JSON preferred, HTML as fallback)
REPORTS = {
    'Profession': {
        'json': 'prof_results.json',
        'html': 'benchmark_report.html'
    },
    'Education': {
        'json': 'edu_results.json',
        'html': 'edu_benchmark_report.html'
    },
    'Location': {
        'json': 'loc_results.json',
        'html': 'loc_benchmark_report.html'
    }
}

def parse_json_report(file_path):
    """
    Parses JSON results file (faster and more reliable than HTML).
    Returns a DataFrame with benchmark results.
    """
    path = Path(file_path)
    if not path.exists():
        return None
        
    try:
        df = pd.read_json(str(path))
        print(f"   ✓ Loaded JSON with {len(df)} rows")
        return df
    except Exception as e:
        print(f"   ❌ Error reading JSON: {e}")
        return None

def parse_html_report(file_path):
    """
    Parses the 'Complete Results' table from the HTML report.
    Returns a DataFrame with columns: [Model, Dimension, Distance, nDCG, MRR, Precision@5, Recall@5]
    """
    path = Path(file_path)
    if not path.exists():
        print(f"⚠️  Report not found: {file_path}")
        return None
        
    try:
        # Read HTML tables
        tables = pd.read_html(str(path))
        
        # Find the 'Complete Results' table - must have both Model AND Dimension columns
        # This distinguishes it from the "Top 5" summary table which has Rank instead
        df = None
        for table in tables:
            # Check for required columns for the full results table
            has_model = 'Model' in table.columns
            has_dimension = 'Dimension' in table.columns or any('Dim' in str(col) for col in table.columns)
            has_ndcg = any('nDCG' in str(col) for col in table.columns)
            
            # Must have all three to be the full results table
            if has_model and has_dimension and has_ndcg:
                df = table
                print(f"   ✓ Found full results table with {len(table)} rows")
                break
        
        if df is None:
            print(f"❌ Could not find full results table in {file_path}")
            print(f"   Available tables: {len(tables)}")
            for i, table in enumerate(tables):
                print(f"   Table {i+1} columns: {list(table.columns)[:5]}")
            return None
            
        # Clean column names (remove arrow symbols and extra whitespace)
        df.columns = [str(c).replace(' ↕', '').replace('↕', '').strip() for c in df.columns]
        
        # Ensure numeric columns are numeric
        numeric_cols = ['Avg nDCG', 'Avg MRR', 'Avg Precision@5', 'Avg Recall@5']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Handle Dimension column (convert to numeric)
        if 'Dimension' in df.columns:
            df['Dimension'] = pd.to_numeric(df['Dimension'], errors='coerce')
        
        print(f"   ✓ Parsed table with {len(df)} rows and columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        print(f"❌ Error parsing {file_path}: {e}")
        return None

def generate_global_report():
    print("🚀 Starting Global Benchmark Aggregation...")
    print("=" * 60)
    
    all_data = []
    loaded_domains = []
    
    # 1. Parse each report (try JSON first, fall back to HTML)
    for domain, files in REPORTS.items():
        print(f"\n📂 Reading {domain} Report...")
        
        df = None
        
        # Try JSON first (faster!)
        json_file = files['json']
        print(f"   Trying JSON: {json_file}")
        df = parse_json_report(json_file)
        
        # Fall back to HTML if JSON not available
        if df is None:
            html_file = files['html']
            print(f"   Falling back to HTML: {html_file}")
            df = parse_html_report(html_file)
        
        if df is not None:
            # Add Domain column
            df['Domain'] = domain
            all_data.append(df)
            loaded_domains.append(domain)
            print(f"   ✅ Loaded {len(df)} rows from {domain}.")
        else:
            print(f"   ❌ Failed to load {domain} data.")

    if not all_data:
        print("\n❌ No data loaded. Please ensure HTML reports exist.")
        print("   Run the individual benchmark scripts first:")
        print("   - python prof_test.py")
        print("   - python edu_test.py")
        print("   - python loc_test.py")
        return
    
    # Validate domain coverage
    if len(loaded_domains) < len(REPORTS):
        missing = set(REPORTS.keys()) - set(loaded_domains)
        print(f"\n⚠️  WARNING: Only {len(loaded_domains)}/3 domains loaded!")
        print(f"   Missing: {', '.join(missing)}")
        print(f"   Results may be incomplete.\n")

    # 2. Combine Data
    full_df = pd.concat(all_data, ignore_index=True)
    
    # 3. Aggregation: Group by [Model, Dimension, Distance]
    # We define a 'Configuration' as Model + Dimension + Distance
    
    # Clean string columns for grouping
    full_df['Model'] = full_df['Model'].str.strip()
    full_df['Distance'] = full_df['Distance'].str.strip()
    
    grouped = full_df.groupby(['Model', 'Dimension', 'Distance'])
    
    # Calculate Global Stats
    global_df = grouped.agg({
        'Avg nDCG': ['mean', 'min', 'max', 'count'],
        'Avg MRR': 'mean'
    }).reset_index()
    
    # Flatten columns
    global_df.columns = ['Model', 'Dimension', 'Distance', 'Global nDCG', 'Min nDCG', 'Max nDCG', 'Domains Count', 'Global MRR']
    
    # Filter only fully tested configurations (must appear in all 3 loaded domains ideally, but we'll show count)
    # Sort by Global nDCG
    global_df = global_df.sort_values(by='Global nDCG', ascending=False)
    
    # 4. Find the Winner
    winner = global_df.iloc[0]
    
    print("\n" + "="*60)
    print(f"🏆 GLOBAL WINNER: {winner['Model']} (Dim: {int(winner['Dimension'])})")
    print(f"   Distance: {winner['Distance']}")
    print(f"   Score: {winner['Global nDCG']:.4f} (Avg across {int(winner['Domains Count'])} domain(s))")
    print(f"   Range: {winner['Min nDCG']:.4f} - {winner['Max nDCG']:.4f}")
    print(f"   Tested on: {', '.join(loaded_domains)}")
    print("="*60)
    
    # 5. Generate HTML
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Global Embedding Benchmark Report</title>
    <style>
        body {{ font-family: 'Inter', sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: #1e293b; padding: 2rem; border-radius: 12px; margin-bottom: 2rem; border: 1px solid #334155; }}
        h1 {{ background: linear-gradient(to right, #6366f1, #a855f7); -webkit-background-clip: text; color: transparent; margin-bottom: 0.5rem; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
        th {{ background: rgba(99,102,241,0.1); padding: 1rem; text-align: left; color: #818cf8; }}
        td {{ padding: 1rem; border-bottom: 1px solid #334155; }}
        tr:hover {{ background: rgba(255,255,255,0.02); }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }}
        .gold {{ color: #fbbf24; background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.2); }}
        .score {{ font-family: monospace; font-size: 1.1em; }}
    </style>
</head>
<body>
    <div class="container">
        <div style="text-align: center; margin-bottom: 3rem;">
            <h1>🌍 Global Embedding Benchmark</h1>
            <p style="color: #94a3b8;">Aggregated performance across Profession, Education, and Location domains</p>
            <p style="color: #64748b; font-size: 0.9em;">Generated: {timestamp}</p>
        </div>

        <div class="card" style="background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(168,85,247,0.1) 100%); border: 1px solid #6366f1;">
            <h2 style="color: #818cf8; margin-bottom: 1rem;">🏆 Top Performer: The "All-Rounder"</h2>
            <div style="display: flex; align-items: center; gap: 2rem; flex-wrap: wrap;">
                <div>
                    <span style="display: block; font-size: 0.9em; color: #94a3b8;">MODEL</span>
                    <strong style="font-size: 1.5em;">{winner['Model']}</strong>
                </div>
                <div>
                    <span style="display: block; font-size: 0.9em; color: #94a3b8;">DIMENSION</span>
                    <strong style="font-size: 1.5em;">{int(winner['Dimension'])}</strong>
                </div>
                <div>
                    <span style="display: block; font-size: 0.9em; color: #94a3b8;">DISTANCE</span>
                    <strong style="font-size: 1.5em;">{winner['Distance']}</strong>
                </div>
                <div style="margin-left: auto; text-align: right;">
                    <span style="display: block; font-size: 0.9em; color: #94a3b8;">GLOBAL nDCG</span>
                    <strong style="font-size: 2.5em; color: #4ade80;">{winner['Global nDCG']:.4f}</strong>
                </div>
            </div>
        </div>

        <div class="card">
            <h3>📊 Global Leaderboard</h3>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Dim</th>
                        <th>Distance</th>
                        <th>Global Score (nDCG)</th>
                        <th>Consistency (Min-Max)</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for i, (_, row) in enumerate(global_df.iterrows()):
        rank_icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
        score_color = "#4ade80" if row['Global nDCG'] > 0.8 else "#fbbf24" if row['Global nDCG'] > 0.7 else "#f87171"
        
        html += f"""
                    <tr>
                        <td>{rank_icon}</td>
                        <td><strong>{row['Model']}</strong></td>
                        <td>{int(row['Dimension'])}</td>
                        <td>{row['Distance']}</td>
                        <td class="score" style="color: {score_color}">{row['Global nDCG']:.4f}</td>
                        <td style="color: #94a3b8; font-size: 0.9em;">{row['Min nDCG']:.4f} - {row['Max nDCG']:.4f}</td>
                    </tr>
        """
    
    html += """
                </tbody>
            </table>
        </div>
        
        <div class="card">
            <h3>📂 Source Data</h3>
            <p style="color: #94a3b8;">This report was generated by aggregating benchmark results from:</p>
            <ul style="color: #cbd5e1; margin-top: 1rem;">
    """
    for domain, files in REPORTS.items():
         html += f"<li><strong>{domain}:</strong> {files['json']} (preferred) or {files['html']} (fallback)</li>"
         
    html += """
            </ul>
            <p style="color: #64748b; font-size: 0.85em; margin-top: 1rem;">
                💡 JSON files are preferred for faster, more reliable data parsing. HTML files are used as fallback.
            </p>
        </div>
    </div>
</body>
</html>
    """
    
    report_path = Path(__file__).parent / "global_benchmark_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
        
    print(f"\n✅ Global Report Generated: {report_path.absolute()}")
    
    # Open
    webbrowser.open(f'file:///{report_path.absolute()}')

if __name__ == "__main__":
    generate_global_report()
