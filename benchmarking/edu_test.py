from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances

# Inner product similarity (dot product)
def inner_product_similarity(a, b):
    return (a @ b.T)
from sklearn.metrics.pairwise import cosine_similarity
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.oauth2 import service_account
from typing import Dict, List, Tuple
import webbrowser
from datetime import datetime

import warnings
import numpy as np
warnings.filterwarnings('ignore')

print("✅ Imports successful")
# ========================
# 1. CONFIGURATION & SETUP
# ========================

# Load Credentials

# Load environment variables
import os  # <-- Add this line
env_path = Path('../../vivah_api/.env')
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env from: {env_path.absolute()}")
else:
    load_dotenv()
    print("⚠️  Using default .env")
# Get API Key
API_KEY = os.getenv("GEMINI_API_KEY")
# Get credentials path
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if CREDENTIALS_PATH and not os.path.isabs(CREDENTIALS_PATH):
    CREDENTIALS_PATH = str((Path('../../vivah_api') / CREDENTIALS_PATH).absolute())

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "triple-name-473801-t7")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

print(f"\n📋 Configuration:")
print(f"   Project: {PROJECT_ID}")
print(f"   Location: {LOCATION}")
print(f"   Credentials: {CREDENTIALS_PATH}")
print(f"   Exists: {os.path.exists(CREDENTIALS_PATH) if CREDENTIALS_PATH else False}")

# Initialize Google GenAI Client

# Create credentials
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
credentials = service_account.Credentials.from_service_account_file(
    CREDENTIALS_PATH, scopes=SCOPES
)

# Initialize GenAI client
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    credentials=credentials,
)

print("✅ Google GenAI client initialized successfully!")


CONFIGS_TO_TEST = [
    # Baseline Models (Standard 768)
    {'name': 'Text-Embed-004 (768)',  'id': 'text-embedding-004', 'dim': 768},
    {'name': 'Text-Embed-005 (768)',  'id': 'text-embedding-005', 'dim': 768},
    
    # Gemini 001 Variable Dimensions Test
    {'name': 'Gemini-001 (768)',   'id': 'gemini-embedding-001', 'dim': 768},
    {'name': 'Gemini-001 (1152)',  'id': 'gemini-embedding-001', 'dim': 1152},
    {'name': 'Gemini-001 (1536)',  'id': 'gemini-embedding-001', 'dim': 1536},
    {'name': 'Gemini-001 (3072)',  'id': 'gemini-embedding-001', 'dim': 3072},
]


# ========================
# 2. CORPUS (Education Database)
# ========================
CORPUS_EDUCATION = [
    # Engineering & Technology
    "B.Tech Computer Science", "B.E. Computer Science", "B.Tech Information Technology", "B.Tech Electronics",
    "M.Tech Computer Science", "MCA", "BCA", "Diploma in Computer Science", "Ph.D. Computer Science",
    "B.Tech Mechanical", "B.E. Mechanical", "B.Tech Civil", "B.E. Civil", "M.Tech Mechanical",
    "B.Sc Computer Science",
    
    # Medical & Health
    "MBBS", "MD General Medicine", "MS General Surgery", "BDS", "MDS", "BAMS", "BHMS",
    "B.Pharma", "M.Pharma", "B.Sc Nursing", "M.Sc Nursing", "Physiotherapy (BPT)",
    
    # Business & Management
    "MBA", "BBA", "PGDM", "MBA Finance", "MBA Marketing", "B.Com", "B.Com Hons",
    "M.Com", "CA", "CS", "CFA", "ICWA",
    
    # Arts & Humanities
    "B.A. English", "M.A. English", "B.A. History", "M.A. History", "B.A. Economics", "M.A. Economics",
    "B.A. Psychology", "M.A. Psychology", "Bachelor of Social Work (BSW)", "Master of Social Work (MSW)",
    "B.A. Sociology",
    
    # Science
    "B.Sc Physics", "M.Sc Physics", "B.Sc Chemistry", "M.Sc Chemistry", "B.Sc Mathematics", "M.Sc Mathematics",
    "B.Sc Biotechnology", "M.Sc Biotechnology", "Ph.D. Physics", "Ph.D. Biotechnology",
    
    # Design & Media
    "B.Des", "M.Des", "Bachelor of Fine Arts (BFA)", "Master of Fine Arts (MFA)",
    "B.Sc Animation", "Bachelor of Journalism", "Master of Journalism",
    
    # Education
    "B.Ed", "M.Ed", "D.Ed"
]
print(f"✅ Corpus Loaded: {len(CORPUS_EDUCATION)} Unique Degrees")

# ========================
# 3. TEST SCENARIOS (Tiered Ground Truth)
# ========================
# Each query has 3 tiers of relevance:
#   Tier 1 (weight 3.0): Highly relevant - Perfect matches
#   Tier 2 (weight 2.0): Relevant - Strong semantic similarity
#   Tier 3 (weight 1.0): Somewhat relevant - Related/adjacent roles
#   Not listed (weight 0.0): Irrelevant
TEST_SCENARIOS = [

    # ================= TECH =================
    {
        "query": "Computer Science Graduate",
        "ground_truth": {
            "tier_1": ["B.Tech Computer Science", "B.E. Computer Science", "B.Sc Computer Science"],
            "tier_2": ["MCA", "M.Tech Computer Science", "B.Tech Information Technology", "BCA"],
            "tier_3": ["Diploma in Computer Science", "B.Tech Electronics"]
        }
    },
    
    # ================= MEDICAL =================
    {
        "query": "Doctor",
        "ground_truth": {
            "tier_1": ["MBBS", "MD General Medicine", "MS General Surgery"],
            "tier_2": ["BDS", "BAMS", "BHMS"],
            "tier_3": ["B.Pharma", "Physiotherapy (BPT)", "B.Sc Nursing"]
        }
    },
    
    # ================= BUSINESS =================
    {
        "query": "MBA",
        "ground_truth": {
            "tier_1": ["MBA", "PGDM", "MBA Finance", "MBA Marketing"],
            "tier_2": ["BBA", "M.Com"],
            "tier_3": ["B.Com", "CA", "CFA"]
        }
    },
    
    # ================= ARTS =================
    {
        "query": "Psychology Major",
        "ground_truth": {
            "tier_1": ["B.A. Psychology", "M.A. Psychology"],
            "tier_2": ["Bachelor of Social Work (BSW)", "Master of Social Work (MSW)"],
            "tier_3": ["B.A. Sociology", "B.Ed"]
        }
    },
    
    # ================= FINANCE =================
    {
        "query": "Accounts Professional",
        "ground_truth": {
            "tier_1": ["CA", "B.Com Hons", "M.Com"],
            "tier_2": ["B.Com", "MBA Finance", "ICWA"],
            "tier_3": ["BBA", "CS", "CFA"]
        }
    }
]



# ========================
# 4. WEIGHTED METRICS CALCULATION (Tiered)
# ========================
TIER_WEIGHTS = {
    'tier_1': 3.0,  # Highly relevant
    'tier_2': 2.0,  # Relevant
    'tier_3': 1.0   # Somewhat relevant
    # Not in any tier: 0.0 (irrelevant)
}

def calculate_metrics(retrieved_jobs, ground_truth_tiered, k=5):
    """
    Calculate weighted metrics based on tiered ground truth.
    
    Args:
        retrieved_jobs: List of retrieved job titles (ordered by model)
        ground_truth_tiered: Dict with 'tier_1', 'tier_2', 'tier_3' keys
        k: Number of results to evaluate (default 5)
    
    Returns:
        precision, recall, mrr, ndcg (all weighted)
    """
    retrieved_k = retrieved_jobs[:k]
    
    # Create job -> weight mapping
    job_weights = {}
    all_relevant_jobs = []
    for tier, jobs in ground_truth_tiered.items():
        weight = TIER_WEIGHTS.get(tier, 0.0)
        for job in jobs:
            job_weights[job] = weight
            all_relevant_jobs.append(job)
    
    # Calculate max possible score (for normalization)
    all_weights_sorted = sorted(job_weights.values(), reverse=True)
    max_score_at_k = sum(all_weights_sorted[:k])
    
    # Weighted Precision@k
    retrieved_score = sum(job_weights.get(job, 0.0) for job in retrieved_k)
    precision = retrieved_score / max_score_at_k if max_score_at_k > 0 else 0
    
    # Weighted Recall@k (how much of total relevance we captured)
    total_relevance = sum(job_weights.values())
    recall = retrieved_score / total_relevance if total_relevance > 0 else 0
    
    # Weighted MRR (first highly relevant result)
    mrr = 0
    for i, job in enumerate(retrieved_jobs):
        weight = job_weights.get(job, 0.0)
        if weight > 0:  # Any relevant result
            mrr = (weight / 3.0) / (i + 1)  # Normalize by max weight
            break
    
    # Weighted nDCG@k
    dcg = 0
    for i, job in enumerate(retrieved_k):
        weight = job_weights.get(job, 0.0)
        dcg += weight / np.log2(i + 2)
    
    # Ideal DCG (best possible ranking)
    idcg = 0
    for i, weight in enumerate(all_weights_sorted[:k]):
        idcg += weight / np.log2(i + 2)
    
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return precision, recall, mrr, ndcg

# ========================
# 5. HTML REPORT GENERATOR
# ========================
def generate_html_report(all_results, query_details, timestamp):
    """Generate interactive HTML report with tabs, charts, and detailed analysis"""
    
    # Sort results
    df = pd.DataFrame(all_results)
    df_sorted = df.sort_values(by=["Avg nDCG", "Avg MRR"], ascending=False)
    
    # Get winner
    winner = df_sorted.iloc[0]
    
    # Prepare data for charts
    chart_data = df.groupby(['Model', 'Dim'])['Avg nDCG'].max().reset_index()
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Education Embedding Benchmark Report</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        :root {{
            --primary: #6366f1;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --border: #334155;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid var(--border);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .header .timestamp {{
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        
        .winner-card {{
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: 0 20px 60px rgba(99, 102, 241, 0.3);
        }}
        
        .winner-card h2 {{
            font-size: 1.8rem;
            margin-bottom: 1rem;
        }}
        
        .winner-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        
        .stat {{
            background: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            display: block;
        }}
        
        .stat-label {{
            font-size: 0.85rem;
            opacity: 0.9;
            display: block;
            margin-top: 0.25rem;
        }}
        
        .tabs {{
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
            border-bottom: 2px solid var(--border);
            flex-wrap: wrap;
        }}
        
        .tab {{
            padding: 1rem 2rem;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 1rem;
            font-weight: 500;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
        }}
        
        .tab:hover {{
            color: var(--text-primary);
        }}
        
        .tab.active {{
            color: var(--primary);
            border-bottom-color: var(--primary);
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
            animation: fadeIn 0.3s;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .card {{
            background: var(--bg-card);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border);
        }}
        
        .card h3 {{
            color: var(--primary);
            margin-bottom: 1rem;
            font-size: 1.3rem;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        
        th {{
            background: rgba(99, 102, 241, 0.1);
            padding: 1rem;
            text-align: left;
            font-weight: 600;
            cursor: pointer;
            user-select: none;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        
        th:hover {{
            background: rgba(99, 102, 241, 0.2);
        }}
        
        td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
        }}
        
        tr:hover {{
            background: rgba(255, 255, 255, 0.03);
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        
        .badge-excellent {{
            background: var(--success);
            color: white;
        }}
        
        .badge-good {{
            background: #3b82f6;
            color: white;
        }}
        
        .badge-fair {{
            background: var(--warning);
            color: white;
        }}
        
        .badge-poor {{
            background: var(--danger);
            color: white;
        }}
        
        .query-section {{
            margin-bottom: 2rem;
        }}
        
        .query-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem;
            background: rgba(99, 102, 241, 0.1);
            border-radius: 10px;
            cursor: pointer;
            margin-bottom: 1rem;
        }}
        
        .query-header:hover {{
            background: rgba(99, 102, 241, 0.15);
        }}
        
        .query-header h4 {{
            font-size: 1.2rem;
            color: var(--primary);
        }}
        
        .ground-truth {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 1rem 0;
        }}
        
        .gt-item {{
            background: rgba(16, 185, 129, 0.2);
            color: var(--success);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            border: 1px solid var(--success);
        }}
        
        .retrieved-list {{
            list-style: none;
            padding: 0;
        }}
        
        .retrieved-list li {{
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 5px;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .match-icon {{
            font-size: 1.2rem;
        }}
        
        .match {{ color: var(--success); }}
        .miss {{ color: var(--danger); }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 2rem 0;
        }}
        
        .model-comparison {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        
        .model-card {{
            background: rgba(255, 255, 255, 0.03);
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid var(--border);
        }}
        
        .model-card h5 {{
            color: var(--primary);
            margin-bottom: 0.5rem;
        }}
        
        .collapse-btn {{
            background: none;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 1.5rem;
            transition: transform 0.3s;
        }}
        
        .collapse-btn.open {{
            transform: rotate(180deg);
        }}
        
        .collapsible-content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }}
        
        .collapsible-content.open {{
            max-height: 2000px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏆 Education Embedding Benchmark Report</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </div>
        
        <div class="winner-card">
            <h2>🥇 Overall Winner</h2>
            <p style="font-size: 1.3rem; margin: 0.5rem 0;">{winner['Model']} + {winner['Distance'].title()}</p>
            <div class="winner-stats">
                <div class="stat">
                    <span class="stat-value">{winner['Avg nDCG']:.4f}</span>
                    <span class="stat-label">nDCG</span>
                </div>
                <div class="stat">
                    <span class="stat-value">{winner['Avg MRR']:.4f}</span>
                    <span class="stat-label">MRR</span>
                </div>
                <div class="stat">
                    <span class="stat-value">{winner['Avg Precision@5']:.4f}</span>
                    <span class="stat-label">Precision@5</span>
                </div>
                <div class="stat">
                    <span class="stat-value">{winner['Avg Recall@5']:.4f}</span>
                    <span class="stat-label">Recall@5</span>
                </div>
            </div>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('summary')">📊 Summary</button>
            <button class="tab" onclick="showTab('queries')">🔍 Per-Query Analysis</button>
            <button class="tab" onclick="showTab('analysis')">🔬 Deep Analysis</button>
            <button class="tab" onclick="showTab('full')">📋 Full Results</button>
        </div>
        
        <div id="summary" class="tab-content active">
            <div class="card">
                <h3>📈 Performance by Dimension</h3>
                <div class="chart-container">
                    <canvas id="dimensionChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h3>🎯 Distance Metric Comparison</h3>
                <div class="chart-container">
                    <canvas id="distanceChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h3>🏅 Top 5 Configurations</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Model</th>
                            <th>Distance</th>
                            <th>nDCG</th>
                            <th>MRR</th>
                            <th>Precision@5</th>
                            <th>Recall@5</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    # Add top 5 rows
    for idx, row in df_sorted.head(5).iterrows():
        badge_class = 'excellent' if row['Avg nDCG'] > 0.9 else 'good' if row['Avg nDCG'] > 0.8 else 'fair'
        html += f"""
                        <tr>
                            <td>{'🥇' if idx == df_sorted.index[0] else '🥈' if idx == df_sorted.index[1] else '🥉' if idx == df_sorted.index[2] else f"#{list(df_sorted.index).index(idx) + 1}"}</td>
                            <td><strong>{row['Model']}</strong></td>
                            <td>{row['Distance'].title()}</td>
                            <td><span class="badge badge-{badge_class}">{row['Avg nDCG']:.4f}</span></td>
                            <td>{row['Avg MRR']:.4f}</td>
                            <td>{row['Avg Precision@5']:.4f}</td>
                            <td>{row['Avg Recall@5']:.4f}</td>
                        </tr>
"""
    
    html += """
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="queries" class="tab-content">
"""
    
    # Add per-query analysis
    for query_name, details in query_details.items():
        query_id = query_name.replace(' ', '_')
        
        # Flatten ground truth for display with tier info
        gt_tiered = details['ground_truth']
        
        html += f"""
            <div class="query-section">
                <div class="query-header" onclick="toggleCollapse('{query_id}')">
                    <h4>Query: "{query_name}"</h4>
                    <button class="collapse-btn" id="btn_{query_id}">▼</button>
                </div>
                <div class="collapsible-content" id="content_{query_id}">
                    <div class="card">
                        <p><strong>Ground Truth (Tiered by Relevance):</strong></p>
                        <div class="ground-truth">
"""
        
        # Display ground truth with tier badges
        tier_colors = {
            'tier_1': 'background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%); color: #000;',  # Gold
            'tier_2': 'background: linear-gradient(135deg, #c0c0c0 0%, #e8e8e8 100%); color: #000;',  # Silver
            'tier_3': 'background: linear-gradient(135deg, #cd7f32 0%, #e6a85c 100%); color: #fff;'   # Bronze
        }
        tier_labels = {'tier_1': '⭐⭐⭐', 'tier_2': '⭐⭐', 'tier_3': '⭐'}
        
        for tier in ['tier_1', 'tier_2', 'tier_3']:
            if tier in gt_tiered and gt_tiered[tier]:
                for gt in gt_tiered[tier]:
                    html += f'                            <span class="gt-item" style="{tier_colors[tier]}" title="{tier_labels[tier]} {tier.replace("_", " ").title()}">{gt}</span>\n'
        
        html += f"""
                        </div>
                        
                        <h4 style="margin-top: 2rem; margin-bottom: 1rem;">🏆 Top 3 Performers</h4>
                        <div class="model-comparison">
"""
        
        # Show top 3 models for this query
        for model_result in details['top_models'][:3]:
            # Calculate weighted score
            weighted_score = model_result.get('weighted_score', 0)
            max_score = model_result.get('max_score', 15)  # 3+3+3+2+2 for top 5
            
            badge_class = 'excellent' if model_result['ndcg'] > 0.9 else 'good' if model_result['ndcg'] > 0.8 else 'fair'
            
            html += f"""
                            <div class="model-card">
                                <h5>{model_result['model']} + {model_result['distance'].title()}</h5>
                                <p><span class="badge badge-{badge_class}">nDCG: {model_result['ndcg']:.4f}</span> | Score: {weighted_score:.1f}/{max_score:.1f}</p>
                                <ul class="retrieved-list">
"""
            for i, job in enumerate(model_result['retrieved'][:5], 1):
                # Determine tier and weight
                job_tier = None
                job_weight = 0
                for tier in ['tier_1', 'tier_2', 'tier_3']:
                    if tier in gt_tiered and job in gt_tiered[tier]:
                        job_tier = tier
                        job_weight = TIER_WEIGHTS[tier]
                        break
                
                if job_weight > 0:
                    icon = '✓'
                    css_class = 'match'
                    tier_label = tier_labels.get(job_tier, '')
                    html += f'                                    <li><span class="match-icon {css_class}">{icon}</span> {i}. {job} <small style="opacity:0.7">({tier_label} {job_weight:.1f}pts)</small></li>\n'
                else:
                    icon = '✗'
                    css_class = 'miss'
                    html += f'                                    <li><span class="match-icon {css_class}">{icon}</span> {i}. {job} <small style="opacity:0.7">(0pts)</small></li>\n'
            
            html += """
                                </ul>
                            </div>
"""
        
        html += f"""
                        </div>
                        
                        <h4 style="margin-top: 2rem; margin-bottom: 1rem;">📊 All Configurations</h4>
                        <table id="table_{query_id}" style="font-size: 0.9rem;">
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Model</th>
                                    <th>Distance</th>
                                    <th>nDCG</th>
                                    <th>Score</th>
                                    <th>Retrieved Top-5</th>
                                </tr>
                            </thead>
                            <tbody>
"""
        
        # Show ALL models for this query
        for rank, model_result in enumerate(details['top_models'], 1):
            weighted_score = model_result.get('weighted_score', 0)
            max_score = model_result.get('max_score', 15)
            badge_class = 'excellent' if model_result['ndcg'] > 0.9 else 'good' if model_result['ndcg'] > 0.8 else 'fair' if model_result['ndcg'] > 0.7 else 'poor'
            
            # Create retrieved list with tier indicators
            retrieved_html = ""
            for i, job in enumerate(model_result['retrieved'][:5], 1):
                job_weight = 0
                for tier in ['tier_1', 'tier_2', 'tier_3']:
                    if tier in gt_tiered and job in gt_tiered[tier]:
                        job_weight = TIER_WEIGHTS[tier]
                        break
                
                if job_weight > 0:
                    color = 'var(--success)'
                    icon = '✓'
                    retrieved_html += f'<span style="color: {color};">{icon}</span> {i}. {job} <small>({job_weight:.1f}pts)</small><br>'
                else:
                    color = 'var(--danger)'
                    icon = '✗'
                    retrieved_html += f'<span style="color: {color};">{icon}</span> {i}. {job} <small>(0pts)</small><br>'
            
            rank_icon = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else f'#{rank}'
            
            html += f"""
                                <tr>
                                    <td>{rank_icon}</td>
                                    <td><strong>{model_result['model']}</strong></td>
                                    <td>{model_result['distance'].title()}</td>
                                    <td><span class="badge badge-{badge_class}">{model_result['ndcg']:.4f}</span></td>
                                    <td>{weighted_score:.1f}/{max_score:.1f}</td>
                                    <td style="line-height: 1.8;">{retrieved_html}</td>
                                </tr>
"""
        
        html += """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
"""
    
    html += """
        </div>
        
        <div id="analysis" class="tab-content">
            <div class="card">
                <h3>🔬 Deep Analysis & Insights</h3>
                <p style="color: var(--text-secondary); margin-bottom: 2rem;">
                    Comprehensive performance analysis across all configurations.
                </p>
            </div>
            
            <div class="card">
                <h3>📊 Performance by Model</h3>
                <div id="modelAnalysis" style="margin-top: 1rem;">
                    <p style="color: var(--text-secondary);">Loading analysis...</p>
                </div>
            </div>
            
            <div class="card">
                <h3>📏 Performance by Distance Metric</h3>
                <div id="distanceAnalysis" style="margin-top: 1rem;">
                    <p style="color: var(--text-secondary);">Loading analysis...</p>
                </div>
            </div>
            
            <div class="card">
                <h3>🎯 Performance by Dimension</h3>
                <div id="dimensionAnalysis" style="margin-top: 1rem;">
                    <p style="color: var(--text-secondary);">Loading analysis...</p>
                </div>
            </div>
            
            <div class="card">
                <h3>💡 Key Recommendations</h3>
                <div id="recommendations" style="margin-top: 1rem;">
                    <p style="color: var(--text-secondary);">Analyzing results...</p>
                </div>
            </div>
        </div>
        
        <div id="full" class="tab-content">
            <div class="card">
                <h3>📋 Complete Results (All Configurations)</h3>
                <table id="fullTable">
                    <thead>
                        <tr>
                            <th onclick="sortTable(0)">Model</th>
                            <th onclick="sortTable(1)">Dimension</th>
                            <th onclick="sortTable(2)">Distance</th>
                            <th onclick="sortTable(3)">Avg nDCG ↕</th>
                            <th onclick="sortTable(4)">Avg MRR ↕</th>
                            <th onclick="sortTable(5)">Avg Precision@5 ↕</th>
                            <th onclick="sortTable(6)">Avg Recall@5 ↕</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    # Add all results
    for _, row in df_sorted.iterrows():
        badge_class = 'excellent' if row['Avg nDCG'] > 0.9 else 'good' if row['Avg nDCG'] > 0.8 else 'fair' if row['Avg nDCG'] > 0.7 else 'poor'
        html += f"""
                        <tr>
                            <td><strong>{row['Model']}</strong></td>
                            <td>{row['Dim']}</td>
                            <td>{row['Distance'].title()}</td>
                            <td><span class="badge badge-{badge_class}">{row['Avg nDCG']:.4f}</span></td>
                            <td>{row['Avg MRR']:.4f}</td>
                            <td>{row['Avg Precision@5']:.4f}</td>
                            <td>{row['Avg Recall@5']:.4f}</td>
                        </tr>
"""
    
    html += """
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        // Tab switching
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
            
            // Populate Deep Analysis when shown
            if (tabName === 'analysis' && typeof populateDeepAnalysis === 'function') {
                populateDeepAnalysis();
            }
        }
        
        // Collapse/Expand
        function toggleCollapse(id) {
            const content = document.getElementById('content_' + id);
            const btn = document.getElementById('btn_' + id);
            content.classList.toggle('open');
            btn.classList.toggle('open');
        }
        
        // Table sorting
        function sortTable(columnIndex) {
            const table = document.getElementById('fullTable');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            rows.sort((a, b) => {
                let aVal = a.cells[columnIndex].textContent.trim();
                let bVal = b.cells[columnIndex].textContent.trim();
                
                // Extract numeric values from badges
                if (columnIndex >= 3) {
                    aVal = parseFloat(aVal);
                    bVal = parseFloat(bVal);
                    return bVal - aVal; // Descending for metrics
                }
                
                return aVal.localeCompare(bVal);
            });
            
            rows.forEach(row => tbody.appendChild(row));
        }
        
        // Charts
        const chartColors = {
            primary: '#6366f1',
            success: '#10b981',
            warning: '#f59e0b',
            danger: '#ef4444',
            info: '#3b82f6'
        };
        
        // Dimension Chart
"""
    
    # Prepare chart data
    dim_data = df.groupby('Dim')['Avg nDCG'].max().sort_index()
    html += f"""
        new Chart(document.getElementById('dimensionChart'), {{
            type: 'line',
            data: {{
                labels: {[int(x) for x in dim_data.index]},
                datasets: [{{
                    label: 'Max nDCG by Dimension',
                    data: {[float(round(v, 4)) for v in dim_data.values]},
                    borderColor: chartColors.primary,
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ labels: {{ color: '#f1f5f9' }} }},
                    title: {{ display: true, text: 'Impact of Embedding Dimension on Performance', color: '#f1f5f9', font: {{ size: 16 }} }}
                }},
                scales: {{
                    y: {{ 
                        beginAtZero: false,
                        grid: {{ color: 'rgba(255, 255, 255, 0.1)' }},
                        ticks: {{ color: '#94a3b8' }}
                    }},
                    x: {{ 
                        grid: {{ color: 'rgba(255, 255, 255, 0.1)' }},
                        ticks: {{ color: '#94a3b8' }}
                    }}
                }}
            }}
        }});
        
        // Distance Metric Chart
"""
    
    dist_data = df.groupby('Distance')['Avg nDCG'].mean().sort_values(ascending=False)
    html += f"""
        new Chart(document.getElementById('distanceChart'), {{
            type: 'bar',
            data: {{
                labels: {[d.title() for d in dist_data.index]},
                datasets: [{{
                    label: 'Average nDCG',
                    data: {[float(round(v, 4)) for v in dist_data.values]},
                    backgroundColor: [
                        chartColors.primary,
                        chartColors.success,
                        chartColors.warning,
                        chartColors.info
                    ],
                    borderWidth: 0,
                    borderRadius: 8
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    title: {{ display: true, text: 'Average Performance by Distance Metric', color: '#f1f5f9', font: {{ size: 16 }} }}
                }},
                scales: {{
                    y: {{ 
                        beginAtZero: false,
                        grid: {{ color: 'rgba(255, 255, 255, 0.1)' }},
                        ticks: {{ color: '#94a3b8' }}
                    }},
                    x: {{ 
                        grid: {{ display: false }},
                        ticks: {{ color: '#94a3b8' }}
                    }}
                }}
            }}
        }});
    """
    
    html += """
        
        // ===== DEEP ANALYSIS TAB POPULATION =====
        function populateDeepAnalysis() {
            const table = document.getElementById('fullTable');
            if (!table) return;
            
            const rows = Array.from(table.querySelectorAll('tbody tr'));
            const data = rows.map(row => ({
                model: row.cells[0].textContent.trim(),
                dim: parseInt(row.cells[1].textContent),
                distance: row.cells[2].textContent.trim(),
                ndcg: parseFloat(row.cells[3].textContent.match(/[0-9.]+/)[0])
            }));
            
            // === Performance by Model ===
            const byModel = {};
            data.forEach(d => {
                if (!byModel[d.model]) byModel[d.model] = [];
                byModel[d.model].push(d);
            });
            
            let modelHTML = '<table style="width:100%"><thead><tr><th>Model</th><th>Avg nDCG</th><th>Best Distance</th><th>Configs</th></tr></thead><tbody>';
            Object.keys(byModel).sort((a, b) => {
                const avgA = byModel[a].reduce((s, d) => s + d.ndcg, 0) / byModel[a].length;
                const avgB = byModel[b].reduce((s, d) => s + d.ndcg, 0) / byModel[b].length;
                return avgB - avgA;
            }).forEach(model => {
                const modelData = byModel[model];
                const avg = (modelData.reduce((s, d) => s + d.ndcg, 0) / modelData.length).toFixed(4);
                const best = modelData.reduce((a, b) => a.ndcg > b.ndcg ? a : b);
                const badge = avg > 0.9 ? 'excellent' : avg > 0.8 ? 'good' : 'fair';
                modelHTML += '<tr>' +
                    '<td><strong>' + model + '</strong></td>' +
                    '<td><span class="badge badge-' + badge + '">' + avg + '</span></td>' +
                    '<td>' + best.distance + '</td>' +
                    '<td>' + modelData.length + '</td>' +
                '</tr>';
            });
            modelHTML += '</tbody></table>';
            document.getElementById('modelAnalysis').innerHTML = modelHTML;
            
            // === Performance by Distance ===
            const byDistance = {};
            data.forEach(d => {
                if (!byDistance[d.distance]) byDistance[d.distance] = [];
                byDistance[d.distance].push(d);
            });
            
            let distHTML = '<table style="width:100%"><thead><tr><th>Distance Metric</th><th>Avg nDCG</th><th>Std Dev</th><th>Best Model</th></tr></thead><tbody>';
            Object.keys(byDistance).sort((a, b) => {
                const avgA = byDistance[a].reduce((s, d) => s + d.ndcg, 0) / byDistance[a].length;
                const avgB = byDistance[b].reduce((s, d) => s + d.ndcg, 0) / byDistance[b].length;
                return avgB - avgA;
            }).forEach(dist => {
                const distData = byDistance[dist];
                const avg = distData.reduce((s, d) => s + d.ndcg, 0) / distData.length;
                const variance = distData.reduce((s, d) => s + Math.pow(d.ndcg - avg, 2), 0) / distData.length;
                const stdDev = Math.sqrt(variance).toFixed(4);
                const best = distData.reduce((a, b) => a.ndcg > b.ndcg ? a : b);
                const badge = avg > 0.9 ? 'excellent' : avg > 0.8 ? 'good' : 'fair';
                distHTML += '<tr>' +
                    '<td><strong>' + dist + '</strong></td>' +
                    '<td><span class="badge badge-' + badge + '">' + avg.toFixed(4) + '</span></td>' +
                    '<td>' + stdDev + '</td>' +
                    '<td>' + best.model + '</td>' +
                '</tr>';
            });
            distHTML += '</tbody></table>';
            document.getElementById('distanceAnalysis').innerHTML = distHTML;
            
            // === Performance by Dimension ===
            const byDim = {};
            data.forEach(d => {
                if (!byDim[d.dim]) byDim[d.dim] = [];
                byDim[d.dim].push(d);
            });
            
            let dimHTML = '<table style="width:100%"><thead><tr><th>Dimension</th><th>Avg nDCG</th><th>Best Config</th></tr></thead><tbody>';
            Object.keys(byDim).sort((a, b) => parseInt(a) - parseInt(b)).forEach(dim => {
                const dimData = byDim[dim];
                const avg = (dimData.reduce((s, d) => s + d.ndcg, 0) / dimData.length).toFixed(4);
                const best = dimData.reduce((a, b) => a.ndcg > b.ndcg ? a : b);
                const badge = avg > 0.9 ? 'excellent' : avg > 0.8 ? 'good' : 'fair';
                dimHTML += '<tr>' +
                    '<td><strong>' + dim + 'D</strong></td>' +
                    '<td><span class="badge badge-' + badge + '">' + avg + '</span></td>' +
                    '<td>' + best.model + ' + ' + best.distance + '</td>' +
                '</tr>';
            });
            dimHTML += '</tbody></table>';
            document.getElementById('dimensionAnalysis').innerHTML = dimHTML;
            
            // === Recommendations ===
            const bestOverall = data.reduce((a, b) => {
                if (Math.abs(a.ndcg - b.ndcg) < 0.000001) {
                    if (a.distance.toLowerCase().includes('cosine')) return a;
                    if (b.distance.toLowerCase().includes('cosine')) return b;
                    return a;
                }
                return a.ndcg > b.ndcg ? a : b;
            });
            const bestDist = Object.keys(byDistance).reduce((a, b) => {
                const avgA = byDistance[a].reduce((s, d) => s + d.ndcg, 0) / byDistance[a].length;
                const avgB = byDistance[b].reduce((s, d) => s + d.ndcg, 0) / byDistance[b].length;
                
                // Tie-breaker logic
                if (Math.abs(avgA - avgB) < 0.000001) {
                    if (a.toLowerCase().includes('cosine')) return a;
                    if (b.toLowerCase().includes('cosine')) return b;
                    return a; 
                }
                return avgA > avgB ? a : b;
            });
            const bestDim = Object.keys(byDim).reduce((a, b) => {
                const avgA = byDim[a].reduce((s, d) => s + d.ndcg, 0) / byDim[a].length;
                const avgB = byDim[b].reduce((s, d) => s + d.ndcg, 0) / byDim[b].length;
                return avgA > avgB ? a : b;
            });
            
            const recHTML = 
                '<div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid var(--border);">' +
                    '<h4 style="color: var(--primary); margin-bottom: 0.75rem;">🥇 Production Recommendation</h4>' +
                    '<p style="line-height: 1.8;">Use <strong>' + bestOverall.model + ' with ' + bestOverall.distance + ' distance</strong> for production. It achieves the highest nDCG (' + bestOverall.ndcg.toFixed(4) + ').</p>' +
                '</div>' +
                '<div style="background: rgba(16, 185, 129, 0.1); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid var(--border);">' +
                    '<h4 style="color: var(--success); margin-bottom: 0.75rem;">📏 Best Distance Metric</h4>' +
                    '<p style="line-height: 1.8;"><strong>' + bestDist + '</strong> performs best on average across all models.</p>' +
                '</div>' +
                '<div style="background: rgba(245, 158, 11, 0.1); padding: 1.5rem; border-radius: 10px; border: 1px solid var(--border);">' +
                    '<h4 style="color: var(--warning); margin-bottom: 0.75rem;">🎯 Optimal Dimension</h4>' +
                    '<p style="line-height: 1.8;"><strong>' + bestDim + 'D embeddings</strong> provide the best average performance.</p>' +
                '</div>';
            document.getElementById('recommendations').innerHTML = recHTML;
        }
        

    </script>
</body>
</html>
"""

    
    return html

# ========================
# 6. BENCHMARK ENGINE
# ========================
def run_benchmark():
    print(f"\n🚀 STARTING SEMANTIC BENCHMARK (Short Queries)")
    print("=" * 100)
    
    final_results = []
    query_details = {}  # Store per-query rankings

    for config in CONFIGS_TO_TEST:
        model_name = config['name']
        model_id = config['id']
        dim = config['dim']
        
        print(f"\n🔵 Processing: {model_name} (Dim: {dim}) ...", end=" ")
        
        # A. Embed Corpus
        try:
            doc_resp = client.models.embed_content(
                model=model_id, contents=CORPUS_EDUCATION,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=dim)
            )
            corpus_vecs = np.array([e.values for e in doc_resp.embeddings])
        except Exception as e:
            print(f"❌ Error embedding corpus: {e}")
            continue
        print("Done.")

        # Storage for this model's performance across all queries
        model_stats = {
            'cosine': {'P': [], 'R': [], 'MRR': [], 'nDCG': []},
            'l2': {'P': [], 'R': [], 'MRR': [], 'nDCG': []},
            'l1': {'P': [], 'R': [], 'MRR': [], 'nDCG': []},
            'inner_product': {'P': [], 'R': [], 'MRR': [], 'nDCG': []}
        }

        # B. Loop Scenarios
        for scenario in TEST_SCENARIOS:
            query = scenario['query']
            gt_tiered = scenario['ground_truth']
            
            # Initialize query details if first time
            if query not in query_details:
                query_details[query] = {
                    'ground_truth': gt_tiered,
                    'top_models': []
                }
            
            try:
                # Embed Query
                q_resp = client.models.embed_content(
                    model=model_id, contents=query,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=dim)
                )
                q_vec = np.array(q_resp.embeddings[0].values).reshape(1, -1)
                
                # C. Calculate All Distances
                distances = {
                    'cosine': cosine_similarity(q_vec, corpus_vecs)[0],
                    'l2': -euclidean_distances(q_vec, corpus_vecs)[0],
                    'l1': -manhattan_distances(q_vec, corpus_vecs)[0],
                    'inner_product': np.dot(corpus_vecs, q_vec.T).flatten()
                }

                # D. Compute Metrics for each Distance
                for dist_name, scores in distances.items():
                    top_indices = np.argsort(scores)[::-1]
                    retrieved_jobs = [CORPUS_EDUCATION[i] for i in top_indices]
                    
                    p, r, mrr, ndcg = calculate_metrics(retrieved_jobs, gt_tiered, k=5)
                    
                    model_stats[dist_name]['P'].append(p)
                    model_stats[dist_name]['R'].append(r)
                    model_stats[dist_name]['MRR'].append(mrr)
                    model_stats[dist_name]['nDCG'].append(ndcg)
                    
                    # Calculate weighted score for top-5 (for HTML display)
                    weighted_score = 0
                    for job in retrieved_jobs[:5]:
                        for tier in ['tier_1', 'tier_2', 'tier_3']:
                            if tier in gt_tiered and job in gt_tiered[tier]:
                                weighted_score += TIER_WEIGHTS[tier]
                                break
                    
                    # Calculate max possible score
                    all_weights = []
                    for tier in ['tier_1', 'tier_2', 'tier_3']:
                        if tier in gt_tiered:
                            all_weights.extend([TIER_WEIGHTS[tier]] * len(gt_tiered[tier]))
                    all_weights.sort(reverse=True)
                    max_score = sum(all_weights[:5])
                    
                    # Store for per-query analysis
                    query_details[query]['top_models'].append({
                        'model': model_name,
                        'distance': dist_name,
                        'ndcg': ndcg,
                        'retrieved': retrieved_jobs[:5],
                        'weighted_score': weighted_score,
                        'max_score': max_score
                    })
                    
            except Exception as e:
                print(f"   ❌ Query Error ({query}): {e}")

        # E. Aggregate Results for this Config
        for dist_name, metrics in model_stats.items():
            final_results.append({
                'Model': f"{model_name}",
                'Dim': dim,
                'Distance': dist_name,
                'Avg Precision@5': np.mean(metrics['P']),
                'Avg Recall@5': np.mean(metrics['R']),
                'Avg MRR': np.mean(metrics['MRR']),
                'Avg nDCG': np.mean(metrics['nDCG'])
            })

    # Sort query details by nDCG
    for query in query_details:
        query_details[query]['top_models'].sort(key=lambda x: x['ndcg'], reverse=True)

    # ========================
    # 7. CONSOLE SUMMARY
    # ========================
    print("\n\n" + "=" * 110)
    print(f"{'🏆 BENCHMARK COMPLETE':^110}")
    print("=" * 110)
    
    df_res = pd.DataFrame(final_results)
    if not df_res.empty:
        df_res = df_res.sort_values(by=["Avg nDCG", "Avg MRR"], ascending=False)
        winner = df_res.iloc[0]
        
        print(f"\n🥇 WINNER: {winner['Model']} + {winner['Distance'].title()}")
        print(f"   nDCG: {winner['Avg nDCG']:.4f} | MRR: {winner['Avg MRR']:.4f} | P@5: {winner['Avg Precision@5']:.4f}\n")
        
        print("📊 Top 5 Configurations:")
        print(df_res.head(5).to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    else:
        print("No results generated.")
    
    # ========================
    # 8. SAVE STRUCTURED DATA (JSON)
    # ========================
    json_path = Path(__file__).parent / "edu_results.json"
    df_res.to_json(json_path, orient='records', indent=2)
    print(f"\n✅ JSON Data Saved: {json_path.absolute()}")
    
    # ========================
    # 9. GENERATE HTML REPORT
    # ========================
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    html_content = generate_html_report(final_results, query_details, timestamp)
    
    report_path = Path(__file__).parent / "edu_benchmark_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML Report Generated: {report_path.absolute()}")
    print("=" * 110)
    
    # Auto-open in browser
    try:
        webbrowser.open(f'file:///{report_path.absolute()}')
        print("🌐 Opening report in browser...")
    except:
        print("⚠️  Please open the HTML file manually")
    
    return df_res

if __name__ == "__main__":
    run_benchmark()