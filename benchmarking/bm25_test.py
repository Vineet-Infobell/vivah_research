
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from datetime import datetime
import webbrowser
import os

print("✅ Imports successful")

# ========================
# 1. CORPUS (Professions Database)
# ========================
CORPUS_JOBS = [
    # Software / IT
    "Software Engineer", "Senior Software Developer", "Backend Developer", "Frontend Developer",
    "Full Stack Developer", "Data Scientist", "Machine Learning Engineer", "DevOps Engineer",
    "Product Manager", "Project Manager", "CTO", "System Administrator", "Cloud Architect", "SDE",
    
    # Medical
    "Doctor", "Surgeon", "General Physician", "Cardiologist", "Dentist", "Physiotherapist",
    "Nurse", "Pharmacist", "Medical Researcher", "Neurologist", "Pediatrician", "Radiologist",
    
    # Legal & Finance
    "Corporate Lawyer", "Advocate", "Judge", "Legal Consultant", "Chartered Accountant",
    "Investment Banker", "Financial Analyst", "Bank Manager", "Tax Consultant",
    
    # Creative & Design
    "Graphic Designer", "UI/UX Designer", "Art Director", "Fashion Designer", "Architect",
    "Interior Designer", "Writer", "Content Creator", "Journalist", "Video Editor",
    
    # Education & Academia
    "Professor", "School Teacher", "Lecturer", "Research Scholar", "Principal", "Academic Coordinator",
    
    # Engineering (Non-IT)
    "Civil Engineer", "Mechanical Engineer", "Electrical Engineer", "Automobile Engineer",
    
    # Business & Ops
    "HR Manager", "Marketing Manager", "Sales Executive", "Operations Head", "Business Analyst"
]
print(f"✅ Corpus Loaded: {len(CORPUS_JOBS)} Unique Professions")

# ========================
# 2. TEST SCENARIOS (Tiered Ground Truth)
# ========================
TEST_SCENARIOS = [
    {
        "query": "Soware Devoper",
        "ground_truth": {
            "tier_1": ["Software Engineer", "Senior Software Developer", "Full Stack Developer"],
            "tier_2": ["Backend Developer", "Frontend Developer", "SDE"],
            "tier_3": ["DevOps Engineer", "Machine Learning Engineer", "Cloud Architect", "System Administrator"]
        }
    },
    {
        "query": "Data Scientist",
        "ground_truth": {
            "tier_1": ["Data Scientist", "Machine Learning Engineer"],
            "tier_2": ["Business Analyst", "Software Engineer"],
            "tier_3": ["Cloud Architect", "DevOps Engineer", "Product Manager", "Research Scholar"]
        }
    },
    {
        "query": "Doctor",
        "ground_truth": {
            "tier_1": ["Doctor", "General Physician"],
            "tier_2": ["Surgeon", "Cardiologist", "Pediatrician", "Neurologist"],
            "tier_3": ["Radiologist", "Medical Researcher", "Pharmacist"]
        }
    },
    {
        "query": "Dentist",
        "ground_truth": {
            "tier_1": ["Dentist"],
            "tier_2": ["Doctor", "General Physician", "Surgeon"],
            "tier_3": ["Medical Researcher", "Pharmacist", "Physiotherapist"]
        }
    },
    {
        "query": "Designer",
        "ground_truth": {
            "tier_1": ["Graphic Designer", "UI/UX Designer", "Interior Designer"],
            "tier_2": ["Fashion Designer", "Art Director", "Architect"],
            "tier_3": ["Content Creator", "Video Editor", "Writer"]
        }
    },
    {
        "query": "Lawyer",
        "ground_truth": {
            "tier_1": ["Corporate Lawyer", "Advocate", "Legal Consultant"],
            "tier_2": ["Judge", "Tax Consultant", "Chartered Accountant"],
            "tier_3": [] 
        }
    },
    {
        "query": "Professor",
        "ground_truth": {
            "tier_1": ["Professor", "Lecturer", "Research Scholar"],
            "tier_2": ["Academic Coordinator", "Principal", "School Teacher"],
            "tier_3": ["Medical Researcher", "Writer"]
        }
    }
]

# ========================
# 3. METRICS SCORING
# ========================
TIER_WEIGHTS = {'tier_1': 3.0, 'tier_2': 2.0, 'tier_3': 1.0}

def calculate_metrics(retrieved_jobs, ground_truth_tiered, k=5):
    retrieved_k = retrieved_jobs[:k]
    job_weights = {}
    for tier, jobs in ground_truth_tiered.items():
        weight = TIER_WEIGHTS.get(tier, 0.0)
        for job in jobs:
            job_weights[job] = weight
            
    all_weights_sorted = sorted(job_weights.values(), reverse=True)
    max_score_at_k = sum(all_weights_sorted[:k])
    
    # Precision
    retrieved_score = sum(job_weights.get(job, 0.0) for job in retrieved_k)
    precision = retrieved_score / max_score_at_k if max_score_at_k > 0 else 0
    
    # MRR
    mrr = 0
    for i, job in enumerate(retrieved_jobs):
        weight = job_weights.get(job, 0.0)
        if weight > 0:
            mrr = (weight / 3.0) / (i + 1)
            break
            
    # nDCG
    dcg = 0
    for i, job in enumerate(retrieved_k):
        weight = job_weights.get(job, 0.0)
        dcg += weight / np.log2(i + 2)
        
    idcg = 0
    for i, weight in enumerate(all_weights_sorted[:k]):
        idcg += weight / np.log2(i + 2)
        
    ndcg = dcg / idcg if idcg > 0 else 0
    return precision, mrr, ndcg

# ========================
# 4. EXECUTION (BM25)
# ========================
print("\n🚀 Preparing BM25 Index...")

# Tokenization
tokenized_corpus = [doc.lower().split() for doc in CORPUS_JOBS]

# Initialize BM25Okapi
bm25 = BM25Okapi(tokenized_corpus)

results_summary = []
query_details = {}

print("✅ Indexing Complete. Running benchmarks...")

all_results = []

for scenario in TEST_SCENARIOS:
    query = scenario['query']
    gt = scenario['ground_truth']
    
    tokenized_query = query.lower().split()
    
    # Get scores
    scores = bm25.get_scores(tokenized_query)
    
    # Rank documents
    # Indices of top scores
    top_indices = np.argsort(scores)[::-1]
    
    retrieved_docs = []
    # Filter out 0 scores? Usually BM25 gives 0 for no match.
    # We'll take top 10 regardless, but usually we care about non-zero.
    
    for idx in top_indices:
        if scores[idx] > 0:
            retrieved_docs.append(CORPUS_JOBS[idx])
    
    # Fallback if less than k
    if len(retrieved_docs) < 5:
        # Just to fill up for metric calc? No, metric calc handles list slicing.
        pass
        
    # Calculate Metrics
    precision, mrr, ndcg = calculate_metrics(retrieved_docs, gt, k=5)
    
    res_obj = {
        'Scenario': query,
        'Retrieved': retrieved_docs[:5],
        'nDCG': round(ndcg, 4),
        'MRR': round(mrr, 4)
    }
    all_results.append(res_obj)
    
    print(f"--- Query: {query} ---")
    print(f"   Top 5: {retrieved_docs[:5]}")
    print(f"   nDCG: {ndcg:.3f} | MRR: {mrr:.3f}\n")

# Calculate Average
avg_ndcg = np.mean([r['nDCG'] for r in all_results])
avg_mrr = np.mean([r['MRR'] for r in all_results])

print("==========================================")
print(f"🏆 BM25 Overall Performance")
print(f"   Avg nDCG: {avg_ndcg:.4f}")
print(f"   Avg MRR:  {avg_mrr:.4f}")
print("==========================================")

# ========================
# 5. HTML REPORT GENERATION
# ========================
html_rows = ""
for res in all_results:
    retrieved_list = "".join([f"<li>{item}</li>" for item in res['Retrieved']])
    row = f"""
    <div class="card">
        <h3>Query: "{res['Scenario']}"</h3>
        <div class="stats">
            <span class="badge" style="background: #4f46e5;">nDCG: {res['nDCG']}</span>
            <span class="badge" style="background: #10b981;">MRR: {res['MRR']}</span>
        </div>
        <p><strong>Top 5 Retrieved:</strong></p>
        <ul>{retrieved_list}</ul>
    </div>
    """
    html_rows += row

full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>BM25 Benchmark Report</title>
    <style>
        body {{ font-family: sans-serif; background: #0f172a; color: white; padding: 2rem; }}
        .header {{ text-align: center; margin-bottom: 2rem; background: #1e293b; padding: 2rem; border-radius: 12px; }}
        .card {{ background: #1e293b; margin-bottom: 1rem; padding: 1.5rem; border-radius: 8px; border: 1px solid #334155; }}
        .stats {{ margin: 10px 0; }}
        .badge {{ padding: 4px 10px; border-radius: 4px; font-weight: bold; margin-right: 10px; }}
        ul {{ list-style: none; padding: 0; }}
        li {{ background: rgba(255,255,255,0.05); padding: 8px; margin: 4px 0; border-radius: 4px; }}
        h3 {{ color: #a5b4fc; margin-top: 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 BM25 Benchmark Results</h1>
        <div class="stats">
            <span class="badge" style="background: #f59e0b; font-size: 1.2rem;">Avg nDCG: {avg_ndcg:.4f}</span>
            <span class="badge" style="background: #8b5cf6; font-size: 1.2rem;">Avg MRR: {avg_mrr:.4f}</span>
        </div>
        <p>Comparison against Semantic Search</p>
    </div>
    {html_rows}
</body>
</html>
"""

with open("bm25_report.html", "w", encoding="utf-8") as f:
    f.write(full_html)
    
webbrowser.open(f'file:///{os.path.abspath("bm25_report.html")}')
print("✅ Report Generated: bm25_report.html")
