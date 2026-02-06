"""
APPROACH 2: Hard Filters + Semantic Search
------------------------------------------
Flow:
1. Hard Filters: 
   - Gender (Exact)
   - Religion (Exact)
   - Age (Range from filters OR -3/+5 years)
2. Semantic Search: Profession + Education + Location
3. Top 10 Results (sorted by semantic similarity only)

Note: Pure semantic ranking without age scoring. Location is semantic, not filtered.
"""

import psycopg2
import json
import time
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

print("="*80)
print("🎯 APPROACH 2: Hard Filters + Semantic Search")
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

# Load ground truth (using refined version)
with open("ground_truth_refined.json", 'r', encoding='utf-8') as f:
    ground_truth = json.load(f)

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
    """
    Search using hard filters + semantic search
    """
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
        
    where_sql = " AND ".join(where_clauses)
    
    sql = f"""
        SELECT 
            user_id, name, gender, age, religion, location, education, job_title,
            1 - (user_vector <=> '{vec_str}'::vector) as similarity
        FROM users
        WHERE {where_sql}
        ORDER BY user_vector <=> '{vec_str}'::vector
        LIMIT {top_k};
    """
    
    # Execute search
    db_start = time.time()
    cur.execute(sql)
    candidates = cur.fetchall()
    db_time = (time.time() - db_start) * 1000
    
    # Format results
    results = []
    for candidate in candidates:
        user_id, name, gender, age, religion, location, education, job_title, similarity = candidate
        
        results.append({
            "user_id": user_id,
            "name": name,
            "age": age,
            "location": location,
            "education": education,
            "job_title": job_title,
            "semantic_score": round(float(similarity), 4)
        })
    
    total_time = (time.time() - start_time) * 1000
    
    return results, total_time, emb_time, db_time

# Helper function to calculate NDCG
def calculate_ndcg(predicted_ids, ground_truth_matches, k=10):
    """Calculate NDCG@k score"""
    # Create relevance map from ground truth
    relevance_map = {m['user_id']: m.get('relevance', 3) for m in ground_truth_matches}
    
    # Get relevance scores for predicted results
    predicted_relevance = [relevance_map.get(pid, 0) for pid in predicted_ids[:k]]
    
    # Calculate DCG
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(predicted_relevance))
    
    # Calculate IDCG (ideal DCG)
    ideal_relevance = sorted([m.get('relevance', 3) for m in ground_truth_matches], reverse=True)[:k]
    idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))
    
    return dcg / idcg if idcg > 0 else 0.0

import numpy as np

# Run all test scenarios
results = []
matches_data = []  # For HTML report
total_latency = 0

print("\n🚀 Running Scenarios...\n")

for scenario in ground_truth:
    print(f"Scenario {scenario['scenario_id']}: {scenario['scenario_name']}")
    
    # Search
    matches, latency, emb_time, db_time = search_matches(
        scenario['query'],
        scenario['filters'],
        scenario['criteria'],
        scenario['searcher'],
        top_k=10
    )
    
    if matches is None:
        print(f"   ❌ Failed\n")
        continue
        
    matches_data.append(matches)
    
    # Calculate metrics
    gt_ids = set([m['user_id'] for m in scenario['ground_truth_candidates'][:10]])
    predicted_ids = [m['user_id'] for m in matches]
    predicted_set = set(predicted_ids)
    
    overlap = len(gt_ids.intersection(predicted_set))
    precision = (overlap / len(predicted_ids)) * 100 if predicted_ids else 0
    recall = (overlap / len(gt_ids)) * 100 if gt_ids else 0
    
    # Calculate NDCG
    ndcg = calculate_ndcg(predicted_ids, scenario['ground_truth_candidates'], k=10)
    
    results.append({
        "scenario_id": scenario['scenario_id'],
        "scenario_name": scenario['scenario_name'],
        "latency_ms": round(latency, 2),
        "embedding_time_ms": round(emb_time, 2),
        "db_time_ms": round(db_time, 2),
        "ndcg": round(ndcg, 3),
        "precision": round(precision, 1),
        "recall": round(recall, 1),
        "overlap": overlap,
        "top_10_ids": predicted_ids
    })
    
    total_latency += latency
    
    print(f"   ⏱️  Latency: {int(latency)}ms")
    print(f"   🎯 NDCG: {ndcg:.3f} | Precision: {precision:.1f}% | Recall: {recall:.1f}%\n")

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
print(f"Average Recall:    {avg_recall:.1f}%")

# Save detailed results
output = {
    "approach": "Approach 2: Hard Filters + Semantic Search",
    "configuration": {
        "filters": ["gender", "religion", "age_range"],
        "ranking": "pure_semantic"
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

with open("approach_2_results.json", 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ JSON Results saved to: approach_2_results.json")

# ========================
# GENERATE HTML REPORT
# ========================

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Matchmaking Report - Approach 2</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #9333ea 0%, #7e22ce 100%);
            color: #333; 
            padding: 40px 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        
        .hero {{
            background: white;
            border-radius: 20px;
            padding: 60px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 40px;
        }}
        .hero h1 {{ 
            font-size: 3em; 
            background: linear-gradient(135deg, #9333ea 0%, #7e22ce 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
        }}
        .hero .subtitle {{ font-size: 1.2em; color: #666; margin-bottom: 30px; }}
        .hero .main-metric {{
            font-size: 4em;
            font-weight: bold;
            color: #9333ea;
            margin: 20px 0;
        }}
        .hero .main-label {{ font-size: 1.3em; color: #888; text-transform: uppercase; letter-spacing: 2px; }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }}
        .stat-card:hover {{ transform: translateY(-5px); }}
        .stat-card .value {{ font-size: 3em; font-weight: bold; color: #9333ea; }}
        .stat-card .label {{ font-size: 1em; color: #666; margin-top: 10px; text-transform: uppercase; letter-spacing: 1px; }}
        
        .config-box {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .config-box h2 {{
            color: #9333ea;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .config-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #9333ea;
        }}
        .config-item .config-label {{ font-weight: bold; color: #9333ea; margin-bottom: 5px; }}
        .config-item .config-value {{ font-size: 1.3em; color: #333; }}
        
        .chart-container {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .chart-container h2 {{ color: #9333ea; margin-bottom: 20px; }}
        
        .scenario {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .scenario-header {{
            border-bottom: 3px solid #9333ea;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }}
        .scenario-header h2 {{ color: #9333ea; font-size: 1.8em; margin-bottom: 10px; }}
        
        .metrics-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #9333ea 0%, #7e22ce 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-card .metric-value {{ font-size: 2em; font-weight: bold; }}
        .metric-card .metric-label {{ font-size: 0.9em; opacity: 0.9; margin-top: 5px; }}
        
        .results-section {{ margin: 30px 0; }}
        .results-section h3 {{ 
            color: #9333ea; 
            margin-bottom: 20px; 
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e0e0;
        }}
        
        .profile-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .profile-card {{
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            background: #fafafa;
            transition: all 0.3s;
        }}
        .profile-card:hover {{ box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
        .profile-card.matched {{ border-color: #9333ea; background: #f3e8ff; }}
        .profile-card.missed {{ border-color: #ef4444; background: #fef2f2; }}
        
        .profile-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            font-weight: bold;
            color: #9333ea;
        }}
        .profile-id {{ font-size: 0.85em; color: #888; }}
        .profile-details {{ color: #555; line-height: 1.6; }}
        .profile-details div {{ margin: 5px 0; }}
        
        .score-pills {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 10px;
        }}
        .score-pill {{
            background: #9333ea;
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        
        .match-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
            margin-top: 8px;
        }}
        .match-badge.found {{ background: #10b981; color: white; }}
        .match-badge.missed {{ background: #ef4444; color: white; }}
        
        .insights-box {{
            background: linear-gradient(135deg, #9333ea 0%, #7e22ce 100%);
            color: white;
            border-radius: 15px;
            padding: 30px;
            margin-top: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .insights-box h2 {{ margin-bottom: 20px; font-size: 2em; }}
        .insights-box ul {{ list-style: none; padding: 0; }}
        .insights-box li {{
            padding: 10px 0;
            padding-left: 30px;
            position: relative;
            line-height: 1.6;
        }}
        .insights-box li:before {{
            content: "✓";
            position: absolute;
            left: 0;
            font-weight: bold;
            font-size: 1.2em;
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="hero">
        <h1>🎯 Approach 2: Hard Filters + Semantic Search</h1>
        <p class="subtitle">Hard filters on Gender, Religion, Age Range → Pure semantic similarity ranking (Location is semantic)</p>
        <div class="main-metric">{avg_ndcg:.3f}</div>
        <div class="main-label">Average NDCG Score</div>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="value">{avg_latency:.0f}ms</div>
            <div class="label">Avg Latency</div>
        </div>
        <div class="stat-card">
            <div class="value">{avg_precision:.1f}%</div>
            <div class="label">Avg Precision</div>
        </div>
        <div class="stat-card">
            <div class="value">{avg_recall:.1f}%</div>
            <div class="label">Avg Recall</div>
        </div>
    </div>
    
    <div class="config-box">
        <h2>⚙️ Configuration</h2>
        <div class="config-grid">
            <div class="config-item">
                <div class="config-label">Hard Filters</div>
                <div class="config-value">Gender, Religion, Age Range</div>
            </div>
            <div class="config-item">
                <div class="config-label">Ranking Method</div>
                <div class="config-value">Pure Semantic (100%)</div>
            </div>
            <div class="config-item">
                <div class="config-label">Candidates Retrieved</div>
                <div class="config-value">Top 10</div>
            </div>
        </div>
        <p style="margin-top: 20px; color: #666; line-height: 1.8;">
            <strong>Approach:</strong> This method applies strict filters (gender, religion, age range), 
            then ranks candidates using pure semantic similarity based on profession, education, and location. 
            Location is part of semantic matching, not a hard filter. No age scoring or additional weighting is applied.
        </p>
    </div>
    
    <div class="chart-container">
        <h2>📊 Performance Across Scenarios</h2>
        <canvas id="performanceChart" width="400" height="150"></canvas>
    </div>
"""

# Add scenario results
for i, result in enumerate(results):
    scenario = ground_truth[i]
    
    # Get matched IDs
    gt_ids = set([m['user_id'] for m in scenario['ground_truth_candidates']])
    pred_ids = result['top_10_ids']
    matched_ids = gt_ids.intersection(pred_ids)
    
    # Get target age info
    age_info = ""
    if 'preferred_age' in scenario['filters']:
        age_info = f", Target Age {scenario['filters']['preferred_age']}"
    elif 'age_min' in scenario['filters'] and 'age_max' in scenario['filters']:
        age_info = f", Age {scenario['filters']['age_min']}-{scenario['filters']['age_max']}"
    
    html_content += f"""
    <div class="scenario">
        <div class="scenario-header">
            <h2>Scenario {result['scenario_id']}: {result['scenario_name']}</h2>
            <p><strong>Query:</strong> {scenario['query']}</p>
            <p><strong>Filters:</strong> {scenario['filters']['gender']}, {scenario['filters']['religion']}{age_info}</p>
        </div>
        
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-value">{result['latency_ms']:.0f}ms</div>
                <div class="metric-label">Latency</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{result['ndcg']:.3f}</div>
                <div class="metric-label">NDCG</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{result['precision']:.1f}%</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{result['recall']:.1f}%</div>
                <div class="metric-label">Recall</div>
            </div>
        </div>
        
        <div class="results-section">
            <h3>🔍 Predicted Top 10 Results</h3>
            <div class="profile-grid">
"""
    # Predicted results
    for idx, profile in enumerate(matches_data[i][:10], 1):
        is_matched = profile['user_id'] in matched_ids
        card_class = "matched" if is_matched else ""
        
        html_content += f"""
                <div class="profile-card {card_class}">
                    <div class="profile-header">
                        <span>#{idx} {profile['name']}</span>
                        <span class="profile-id">ID: {profile['user_id']}</span>
                    </div>
                    <div class="profile-details">
                        <div>👔 {profile['job_title']}</div>
                        <div>🎓 {profile['education']}</div>
                        <div>📍 {profile['location']} · {profile['age']} years</div>
                    </div>
                    <div class="score-pills">
                        <span class="score-pill">Semantic: {profile['semantic_score']:.3f}</span>
                    </div>
                </div>
"""
    html_content += """
            </div>
        </div>
        
        <div class="results-section">
            <h3>🎯 Ground Truth (Top 15 Expected Matches)</h3>
            <div class="profile-grid">
"""
    
    # Ground truth
    for idx, profile in enumerate(scenario['ground_truth_candidates'], 1):
        is_found = profile['user_id'] in matched_ids
        card_class = "matched" if is_found else "missed"
        badge_class = "found" if is_found else "missed"
        badge_text = "✓ Found" if is_found else "○ Missed"
        relevance = profile.get('relevance_score', 3)
        
        html_content += f"""
                <div class="profile-card {card_class}">
                    <div class="profile-header">
                        <span>#{idx} {profile['name']}</span>
                        <span class="profile-id">ID: {profile['user_id']}</span>
                    </div>
                    <div class="profile-details">
                        <div>👔 {profile['job_title']}</div>
                        <div>🎓 {profile['education']}</div>
                        <div>📍 {profile['location']} · {profile['age']} years</div>
                    </div>
                    <span class="match-badge {badge_class}">{badge_text}</span>
                    <span class="score-pill" style="margin-left: 8px;">Relevance: {relevance}</span>
                </div>
"""
    
    html_content += """
            </div>
        </div>
    </div>
"""

# Add Chart.js script and insights
html_content += f"""
    <div class="insights-box">
        <h2>💡 Key Insights</h2>
        <ul>
            <li><strong>Pure Semantic Approach:</strong> 100% semantic similarity ranking without age scoring or additional weights</li>
            <li><strong>Essential Hard Filters:</strong> Gender, religion, age range - Location is semantic</li>
            <li><strong>Simplicity:</strong> Straightforward approach - minimal filters + semantic search, easy to understand</li>
            <li><strong>Performance:</strong> Average NDCG of {avg_ndcg:.3f} with {avg_precision:.1f}% precision in {avg_latency:.0f}ms</li>
            <li><strong>Flexibility:</strong> Location matching is flexible through semantic similarity</li>
        </ul>
    </div>
</div>

<script>
const ctx = document.getElementById('performanceChart').getContext('2d');
const data = {{
    labels: {[f"S{r['scenario_id']}" for r in results]},
    datasets: [
        {{
            label: 'NDCG',
            data: {[r['ndcg'] for r in results]},
            backgroundColor: 'rgba(147, 51, 234, 0.8)',
            borderColor: 'rgba(147, 51, 234, 1)',
            borderWidth: 2
        }},
        {{
            label: 'Precision (%)',
            data: {[r['precision']/100 for r in results]},
            backgroundColor: 'rgba(126, 34, 206, 0.8)',
            borderColor: 'rgba(126, 34, 206, 1)',
            borderWidth: 2
        }}
    ]
}};

new Chart(ctx, {{
    type: 'bar',
    data: data,
    options: {{
        responsive: true,
        maintainAspectRatio: true,
        plugins: {{
            legend: {{
                display: true,
                position: 'top'
            }},
            title: {{
                display: false
            }}
        }},
        scales: {{
            y: {{
                beginAtZero: true,
                max: 1.0,
                ticks: {{
                    callback: function(value) {{
                        return value.toFixed(2);
                    }}
                }}
            }}
        }}
    }}
}});
</script>

</body>
</html>
"""

with open("approach_2_report.html", "w", encoding='utf-8') as f:
    f.write(html_content)

print("✅ HTML Report saved: approach_2_report.html")

cur.close()
conn.close()
