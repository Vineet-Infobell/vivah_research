"""
APPROACH 3: Cross-Encoder Re-ranking + Hard Filters
---------------------------------------------------
Flow:
1. Hard Filters: Gender, Religion, Age Range, Location (Exact)
2. Semantic Search: Get Top 50 candidates
3. Cross-Encoder: Re-rank Top 50 using 'cross-encoder/ms-marco-MiniLM-L-6-v2'
4. Return Top 10
"""

import psycopg2
import json
import time
from sentence_transformers import CrossEncoder, util
import numpy as np # Needed for sigmoid
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

from pathlib import Path

from google.oauth2 import service_account

# Load environment from parent directory
env_path = Path('../../vivah_api/.env')
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env from: {env_path.absolute()}")
else:
    load_dotenv()
    print("⚠️ Using current directory .env")

# Setup Google GenAI (using service account like load_data.py)
print("🔑 Initializing Google GenAI...")

CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if CREDENTIALS_PATH and not os.path.isabs(CREDENTIALS_PATH):
    CREDENTIALS_PATH = str((Path('../../vivah_api') / CREDENTIALS_PATH).absolute())

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "triple-name-473801-t7")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=SCOPES)

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    credentials=credentials
)

# Load Cross-Encoder
print("⏳ Loading Cross-Encoder...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') 
print("✅ Cross-Encoder Ready!")

print("="*80)
print("🎯 APPROACH 3: Cross-Encoder Re-ranking + Hard Filters")
print("="*80)

# Database connection
conn = psycopg2.connect(
    host="localhost",
    port="5433",
    database="postgres",
    user="postgres",
    password="matchpass"
)
cur = conn.cursor()

# Load ground truth
with open("ground_truth_refined.json", 'r', encoding='utf-8') as f:
    ground_truth = json.load(f)

# NDCG Calculation Helpers
import math

def calculate_dcg(relevances):
    dcg = 0
    for i, rel in enumerate(relevances):
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg

def calculate_ndcg(predicted_ids, ground_truth_candidates, k=10):
    relevance_map = {c['user_id']: c['relevance_score'] for c in ground_truth_candidates}
    relevances = [relevance_map.get(pid, 0) for pid in predicted_ids[:k]]
    actual_dcg = calculate_dcg(relevances)
    ideal_relevances = sorted(list(relevance_map.values()), reverse=True)[:k]
    ideal_dcg = calculate_dcg(ideal_relevances)
    if ideal_dcg == 0: return 0.0
    return actual_dcg / ideal_dcg

def gaussian_age_score(target_age, candidate_age):
    """
    Balanced age decay (from Approach 1 optimization)
    - Older: 10.62% decay per year (Base 0.8938)
    - Younger: 9.09% decay per year (Base 0.9091)
    """
    diff = candidate_age - target_age
    if diff == 0: return 1.0
    if diff > 0: return 1.0 * (0.8938 ** diff)  # Older
    else: return 1.0 * (0.9091 ** abs(diff))   # Younger


def create_embedding(text: str) -> list:
    """Generate 1152-dim embedding using gemini-embedding-001"""
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=str(text),
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=1152
            )
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"Embedding Error: {e}")
        return None

def search_matches(query: str, filters: dict, criteria: dict, searcher: dict, top_k: int = 10):
    start_time = time.time()
    
    # Generate query embedding
    emb_start = time.time()
    query_vec = create_embedding(query)
    emb_time = (time.time() - emb_start) * 1000
    
    if not query_vec:
        return None, 0, 0, 0
    
    vec_str = '[' + ','.join(map(str, query_vec)) + ']'
    
    # CALCULATE AGE RANGE
    if 'age_min' in filters and 'age_max' in filters:
        age_min = filters['age_min']
        age_max = filters['age_max']
    elif 'preferred_age' in filters:
        age_min = filters['preferred_age'] - 3
        age_max = filters['preferred_age'] + 3
    else:
        # Default fallback
        age_min = searcher['age'] - 3
        age_max = searcher['age'] + 5

    # BUILD SQL QUERY
    where_clauses = [
        f"gender = '{filters['gender']}'",
        f"religion = '{filters['religion']}'",
        f"age BETWEEN {age_min} AND {age_max}"
    ]
    
    # Add Location Filter (if specified and not flexible)
    if criteria.get('location') and not criteria.get('location_flexible'):
        where_clauses.append(f"location = '{criteria['location']}'")
        
    where_sql = " AND ".join(where_clauses)
    
    # GET MORE CANDIDATES FOR RE-RANKING (Top 50)
    sql = f"""
        SELECT 
            user_id, name, gender, age, religion, location, education, job_title,
            1 - (user_vector <=> '{vec_str}'::vector) as similarity
        FROM users
        WHERE {where_sql}
        ORDER BY user_vector <=> '{vec_str}'::vector
        LIMIT 50;
    """
    
    db_start = time.time()
    cur.execute(sql)
    candidates = cur.fetchall()
    db_time = (time.time() - db_start) * 1000
    
    if not candidates:
        return [], 0, 0, 0
    
    # PREPARE FOR CROSS-ENCODER
    # Pair: (Query, Candidate Profile Text)
    pairs = []
    candidate_map = {}
    
    for c in candidates:
        user_id, name, gender, age, religion, location, education, job_title, similarity = c
        
        # Profile string for re-ranking
        profile_text = f"{job_title} with {education} living in {location}. Age {age}, {religion} {gender}."
        
        pairs.append([query, profile_text])
        
        candidate_map[len(pairs)-1] = {
            "user_id": user_id,
            "name": name,
            "age": age,
            "location": location,
            "education": education,
            "job_title": job_title,
            "semantic_score": round(float(similarity), 4)
        }
    
    # RE-RANK SCORES
    rerank_start = time.time()
    
    # Cross-Encoder outputs logits (unbounded). We need to normalize them to 0-1 for weighted sum.
    # We use Sigmoid: 1 / (1 + exp(-x))
    raw_scores = cross_encoder.predict(pairs)
    sigmoid_scores = 1 / (1 + np.exp(-raw_scores))
    
    rerank_time = (time.time() - rerank_start) * 1000
    
    # Use Pure Cross-Encoder Score for Ranking
    scored_candidates = []
    
    for i, ce_score in enumerate(sigmoid_scores):
        cand = candidate_map[i]
        
        # Pure cross-encoder score (100%)
        final_score = float(ce_score)
        
        cand['cross_encoder_score'] = round(float(ce_score), 4)
        cand['final_score'] = round(float(final_score), 4)
        
        scored_candidates.append(cand)
    
    # Sort by Cross-Encoder Score
    scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Take top K
    final_results = scored_candidates[:top_k]
    
    total_time = (time.time() - start_time) * 1000
    
    # Add rerank time to latency check logic
    # (Just updating db_time to include rerank for simplicity in return signature)
    db_time += rerank_time 
    
    return final_results, total_time, emb_time, db_time

# Run all test scenarios
results = []
matches_data = [] # For HTML
total_latency = 0

print("\n🚀 Running Test Scenarios...\n")

for scenario in ground_truth:
    print(f"Scenario {scenario['scenario_id']}: {scenario['scenario_name']}")
    
    matches, latency, emb_time, db_time = search_matches(
        scenario['query'],
        scenario['filters'],
        scenario['criteria'],
        scenario['searcher'],
        top_k=10
    )
    
    matches_data.append(matches)
    
    # Calculate Metrics (NDCG, Precision, Recall)
    gt_candidates = scenario['ground_truth_candidates']
    gt_ids = set([m['user_id'] for m in gt_candidates if m['relevance_score'] > 0]) # All relevant
    
    predicted_id_list = [m['user_id'] for m in matches]
    predicted_set = set(predicted_id_list)
    
    # Hits
    hits = gt_ids.intersection(predicted_set)
    
    # Metrics
    precision = (len(hits) / 10) * 100
    recall = (len(hits) / len(gt_ids)) * 100 if len(gt_ids) > 0 else 0
    ndcg = calculate_ndcg(predicted_id_list, gt_candidates, k=10)
    
    results.append({
        "scenario_id": scenario['scenario_id'],
        "scenario_name": scenario['scenario_name'],
        "latency_ms": round(latency, 2),
        "ndcg": round(ndcg, 3),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "top_10_ids": predicted_id_list
    })
    
    total_latency += latency
    
    print(f"   ⏱️  Latency: {latency:.2f}ms")
    print(f"   🎯 NDCG: {ndcg:.3f} | Precision: {precision:.1f}% | Recall: {recall:.1f}%")
    print(f"   Top 3: {[m['user_id'] for m in matches[:3]]}\n")

# Summary
avg_latency = total_latency / len(results) if results else 0
avg_ndcg = sum(r['ndcg'] for r in results) / len(results) if results else 0
avg_precision = sum(r['precision'] for r in results) / len(results) if results else 0
avg_recall = sum(r['recall'] for r in results) / len(results) if results else 0

print("="*80)
print("📊 OVERALL RESULTS")
print("="*80)
print(f"Average Latency:   {avg_latency:.2f}ms")
print(f"Average NDCG:      {avg_ndcg:.3f}")
print(f"Average Precision: {avg_precision:.1f}%")

# Save JSON results
output = {
    "approach": "Approach 3: Hard Filters + Age Range + Semantic + Cross-Encoder",
    "configuration": {
        "filters": ["gender", "religion", "age_range"],
        "ranking": "pure_cross_encoder",
        "ce_weight": 1.0
    },
    "summary": {
        "average_latency_ms": round(avg_latency, 2),
        "average_ndcg": round(avg_ndcg, 3),
        "average_precision": round(avg_precision, 1),
        "average_recall": round(avg_recall, 1),
        "total_scenarios": len(results)
    },
    "scenarios": results
}

with open("approach_3_results.json", 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ JSON Results saved to: approach_3_results.json")

# Generate Detailed HTML Report
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Approach 3: Hard Filters + Age Range + Cross-Encoder - Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            padding: 40px; 
            border-radius: 16px; 
            box-shadow: 0 20px 60px rgba(0,0,0,0.3); 
        }}
        
        h1 {{ 
            color: #2d3748; 
            text-align: center; 
            margin-bottom: 10px; 
            font-size: 2.5em;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .subtitle {{
            text-align: center;
            color: #718096;
            margin-bottom: 40px;
            font-size: 1.1em;
        }}
        
        .hero-section {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 40px;
            text-align: center;
        }}
        
        .hero-section h2 {{ 
            font-size: 3.5em; 
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        .hero-section p {{ 
            font-size: 1.3em; 
            opacity: 0.95; 
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .stat-card {{
            background: #f7fafc;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            border-left: 4px solid #10b981;
            transition: transform 0.2s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }}
        
        .stat-card .label {{
            display: block;
            font-size: 0.85em;
            color: #718096;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }}
        
        .stat-card .value {{
            display: block;
            font-size: 2em;
            font-weight: bold;
            color: #2d3748;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #2d3748;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #10b981;
        }}
        
        .config-box {{
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 40px;
            border-left: 4px solid #10b981;
        }}
        
        .config-box h3 {{
            color: #065f46;
            margin-bottom: 15px;
        }}
        
        .config-item {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #a7f3d0;
        }}
        
        .config-item:last-child {{
            border-bottom: none;
        }}
        
        .config-label {{
            font-weight: 600;
            color: #064e3b;
        }}
        
        .config-value {{
            color: #065f46;
            font-family: 'Courier New', monospace;
        }}
        
        .chart-container {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        
        .scenario {{
            background: #f9fafb;
            margin-bottom: 40px;
            border: 2px solid #e5e7eb;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .scenario h2 {{
            color: #059669;
            margin-bottom: 20px;
            font-size: 1.5em;
        }}
        
        .metrics {{
            display: flex;
            gap: 15px;
            margin: 15px 0 25px 0;
            flex-wrap: wrap;
        }}
        
        .metric {{
            background: #d1fae5;
            padding: 12px 20px;
            border-radius: 8px;
            color: #065f46;
            font-weight: 600;
            font-size: 0.95em;
        }}
        
        .comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
        }}
        
        .comparison h3 {{
            color: #374151;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}
        
        .profile-card {{
            border: 2px solid #e5e7eb;
            padding: 15px;
            margin-bottom: 12px;
            border-radius: 8px;
            background: white;
            transition: all 0.2s;
        }}
        
        .profile-card:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transform: translateX(5px);
        }}
        
        .match-yes {{
            border-left: 5px solid #10b981;
            background: linear-gradient(90deg, #ecfdf5 0%, white 100%);
        }}
        
        .match-no {{
            border-left: 5px solid #ef4444;
            background: linear-gradient(90deg, #fef2f2 0%, white 100%);
        }}
        
        .score-pill {{
            display: inline-block;
            background: #e5e7eb;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            margin-right: 8px;
            font-weight: 500;
        }}
        
        .insights-box {{
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border-left: 4px solid #f59e0b;
            padding: 25px;
            border-radius: 8px;
            margin-top: 30px;
        }}
        
        .insights-box h3 {{
            color: #92400e;
            margin-bottom: 15px;
        }}
        
        .insights-box ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .insights-box li {{
            padding: 10px 0;
            color: #78350f;
            font-size: 1.05em;
        }}
        
        .insights-box li:before {{
            content: "💡 ";
            margin-right: 10px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }}
        
        .badge-success {{
            background-color: #d1fae5;
            color: #065f46;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Approach 3: Hard Filters + Age Range + Cross-Encoder</h1>
        <div class="subtitle">Hard filters (Gender, Religion, Age Range) → Semantic (Top 50) → Pure Cross-Encoder Re-rank → Top 10</div>
        
        <!-- Hero Section -->
        <div class="hero-section">
            <h2>{avg_ndcg:.3f}</h2>
            <p>Average NDCG Score (Ranking Quality)</p>
            <p style="font-size: 0.9em; margin-top: 10px; opacity: 0.8;">
                {len(results)} scenarios tested | Average latency: {avg_latency:.0f}ms
            </p>
        </div>
        
        <!-- Statistics Grid -->
        <div class="stats-grid">
            <div class="stat-card">
                <span class="label">Avg NDCG</span>
                <span class="value">{avg_ndcg:.3f}</span>
            </div>
            <div class="stat-card">
                <span class="label">Avg Precision</span>
                <span class="value">{avg_precision:.1f}%</span>
            </div>
            <div class="stat-card">
                <span class="label">Avg Recall</span>
                <span class="value">{avg_recall:.1f}%</span>
            </div>
            <div class="stat-card">
                <span class="label">Avg Latency</span>
                <span class="value">{avg_latency:.0f}ms</span>
            </div>
        </div>
        
        <!-- Configuration Section -->
        <div class="section">
            <h2 class="section-title">⚙️ Algorithm Configuration</h2>
            <div class="config-box">
                <h3>Pure Cross-Encoder Ranking</h3>
                <div class="config-item">
                    <span class="config-label">Hard Filters:</span>
                    <span class="config-value">Gender, Religion, Age Range</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Ranking Method:</span>
                    <span class="config-value">Pure Cross-Encoder (100%)</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Cross-Encoder Model:</span>
                    <span class="config-value">ms-marco-MiniLM-L-6-v2</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Candidates Retrieved:</span>
                    <span class="config-value">Top 50 → Top 10</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Age Handling:</span>
                    <span class="config-value">Hard filter (range only), not scored</span>
                </div>
            </div>
        </div>
        
        <!-- Performance Charts -->
        <div class="section">
            <h2 class="section-title">📊 Performance Metrics</h2>
            <div class="chart-container">
                <canvas id="performanceChart" height="80"></canvas>
            </div>
        </div>
"""

for i, result in enumerate(results):
    scenario = ground_truth[i]
    gt_candidates = scenario['ground_truth_candidates']
    gt_ids = set([m['user_id'] for m in gt_candidates if m['relevance_score'] > 0])
    pred_ids = set(result['top_10_ids'])
    matched_ids = gt_ids.intersection(pred_ids)

    html_content += f"""
        <div class="scenario">
            <h2>{scenario['scenario_name']}</h2>
            <div class="metrics">
                <div class="metric">Latency: {result['latency_ms']}ms</div>
                <div class="metric">NDCG: {result['ndcg']}</div>
                <div class="metric">Precision: {result['precision']}%</div>
            </div>
            
            <div class="comparison">
                <div>
                    <h3>✅ Predicted Results (Top 10)</h3>
    """
    
    for p in matches_data[i][:10]:
        is_match = p['user_id'] in gt_ids
        cls = "match-yes" if is_match else "match-no"
        html_content += f"""
                    <div class="profile-card {cls}">
                        <strong>{p['name']} (ID: {p['user_id']})</strong><br>
                        {p['job_title']}, {p['location']}<br>
                        <div style="margin-top:5px;">
                            <span class="score-pill">Final: {p['final_score']}</span>
                            <span class="score-pill">CE: {p['cross_encoder_score']}</span>
                        </div>
                    </div>
        """
        
    html_content += """
                </div>
                <div>
                    <h3>📋 Ground Truth (All Valid)</h3>
    """
    
    for p in gt_candidates:
        is_found = p['user_id'] in pred_ids
        status = "✓ Found" if is_found else "○ Missed"
        style = "color:green" if is_found else "color:gray"
        html_content += f"""
                    <div class="profile-card">
                        <div style="float:right; {style}; font-weight:bold;">{status}</div>
                        <strong>{p['name']} (ID: {p['user_id']})</strong> - Tier {p['relevance_score']}<br>
                        {p['job_title']}, {p['location']}
                    </div>
        """
        
    html_content += """
                </div>
            </div>
        </div>
    """

# Add insights section
html_content += f"""
        <!-- Key Insights -->
        <div class="insights-box">
            <h3>🔑 Key Insights</h3>
            <ul>
                <li><strong>Pure Cross-Encoder:</strong> 100% cross-encoder ranking provides direct query-candidate comparison for superior semantic understanding.</li>
                <li><strong>Two-Stage Approach:</strong> Initial vector search (top 50) → Cross-encoder re-ranking (top 10) balances speed and accuracy.</li>
                <li><strong>Age Range Filter:</strong> Age is filtered as a range (hard filter), not scored - ensures age compatibility upfront.</li>
                <li><strong>Hard Filters First:</strong> Gender, Religion, and Age Range filtered before semantic matching for efficiency.</li>
                <li><strong>No Age Scoring:</strong> Simplified approach focuses purely on semantic relevance after essential filters.</li>
            </ul>
        </div>
    </div>
    
    <script>
        // Performance Chart
        const ctx = document.getElementById('performanceChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {[f"Scenario {r['scenario_id']}" for r in results]},
                datasets: [
                    {{
                        label: 'NDCG',
                        data: {[r['ndcg'] for r in results]},
                        backgroundColor: 'rgba(16, 185, 129, 0.7)',
                        borderColor: 'rgba(16, 185, 129, 1)',
                        borderWidth: 2
                    }},
                    {{
                        label: 'Precision (%)',
                        data: {[r['precision']/100 for r in results]},
                        backgroundColor: 'rgba(59, 130, 246, 0.7)',
                        borderColor: 'rgba(59, 130, 246, 1)',
                        borderWidth: 2
                    }},
                    {{
                        label: 'Recall (%)',
                        data: {[r['recall']/100 for r in results]},
                        backgroundColor: 'rgba(245, 158, 11, 0.7)',
                        borderColor: 'rgba(245, 158, 11, 1)',
                        borderWidth: 2
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: true, position: 'top' }},
                    title: {{ 
                        display: true,
                        text: 'Performance Across All Scenarios',
                        font: {{ size: 16 }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1.1,
                        title: {{ display: true, text: 'Score' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

with open("approach_3_report.html", "w", encoding='utf-8') as f:
    f.write(html_content)

print("\n✅ Detailed HTML Report saved to: approach_3_report.html")
import webbrowser
webbrowser.open('approach_3_report.html')
print("✅ Report saved & opened!")

cur.close()
conn.close()
