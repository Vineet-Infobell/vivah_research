"""
APPROACH 6: Age Range (Hard) + Semantic + Gaussian Age Scoring
----------------------------------------------------------------
Flow:
1. Hard Filters: Gender, Religion, Age Range (-3/+5)
2. Semantic Search: Get candidates
3. Gaussian Age Scoring: Decay from preferred age
4. Final Score: 87.46% Semantic + 12.54% Age
5. Top 10 Results

Concept: Double age consideration - filter out incompatible ages, 
then score remaining candidates for fine-grained age preferences.
Best of Approach 1 (Gaussian) + Approach 2 (Age Range Filter)
"""

import psycopg2
import json
import time
import math
from google import genai
from google.genai import types
from google.oauth2 import service_account
import os
from pathlib import Path
from dotenv import load_dotenv

# ========================
# 1. SETUP
# ========================

# Load .env
env_path = Path(__file__).parent.parent.parent / "vivah_api" / ".env"
load_dotenv(env_path)
print(f"✅ Loaded .env from: {env_path}")

CREDENTIALS_PATH = str(Path(__file__).parent.parent.parent / "vivah_api" / "service-account.json")
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "triple-name-473801-t7")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=SCOPES)

print("🔑 Initializing Google GenAI...")
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    credentials=credentials
)

print("="*80)
print("🎯 APPROACH 6: Age Range + Semantic + Gaussian Age")
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

# NDCG Calculation Helpers
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
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0

# Gaussian Age Scoring Function
def gaussian_age_score(preferred_age, candidate_age):
    """
    Optimized Gaussian scoring with asymmetric decay.
    Older base (0.8938) = 10.62% decay per year
    Younger base (0.9091) = 9.09% decay per year
    """
    age_diff = candidate_age - preferred_age
    if age_diff >= 0:  # Candidate is older or same age
        base = 0.8938
        score = base ** abs(age_diff)
    else:  # Candidate is younger
        base = 0.9091
        score = base ** abs(age_diff)
    return score

# ========================
# 2. MATCHMAKING FUNCTION
# ========================

def matchmaking_approach_6(searcher, criteria, top_k=10):
    """
    Approach 6: Hard age filter + Semantic + Gaussian Age Scoring
    """
    start_time = time.time()
    
    # Generate query embedding - using jobs and educations from criteria
    jobs = criteria.get('jobs', ['Software Developer'])
    educations = criteria.get('educations', ['B.Tech'])
    location = criteria.get('location', 'Bangalore')
    
    query = f"{jobs[0]} with {educations[0]} in {location}"
    
    emb_start = time.time()
    query_vec = create_embedding(query)
    emb_time = (time.time() - emb_start) * 1000
    
    if not query_vec:
        return [], 0, 0, 0
    
    vec_str = '[' + ','.join(map(str, query_vec)) + ']'
    
    # PREPARE FILTERS
    filters = {
        'gender': 'Female' if searcher['gender'] == 'Male' else 'Male',
        'religion': criteria.get('religion', searcher['religion'])
    }
    
    # Age Range Filter (-3/+5 years from preferred age)
    preferred_age = criteria.get('preferred_age', searcher['age'])
    age_min = preferred_age - 3
    age_max = preferred_age + 5
    
    # BUILD SQL QUERY with Hard Filters
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
        LIMIT 50;
    """
    
    db_start = time.time()
    cur.execute(sql)
    candidates = cur.fetchall()
    db_time = (time.time() - db_start) * 1000
    
    if not candidates:
        return [], 0, 0, 0
    
    # Calculate Final Weighted Score with Gaussian Age
    target_age = preferred_age
    
    scored_candidates = []
    for c in candidates:
        user_id, name, gender, age, religion, location, education, job_title, similarity = c
        
        # Calculate age score
        age_score = gaussian_age_score(target_age, age)
        
        # Weighted combination: 87.46% Semantic + 12.54% Age
        final_score = (float(similarity) * 0.8746) + (age_score * 0.1254)
        
        scored_candidates.append({
            "user_id": user_id,
            "name": name,
            "age": age,
            "location": location,
            "education": education,
            "job_title": job_title,
            "semantic_score": round(float(similarity), 4),
            "age_score": round(float(age_score), 4),
            "final_score": round(float(final_score), 4)
        })
    
    # Sort by final score
    scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Take top K
    final_results = scored_candidates[:top_k]
    
    total_time = (time.time() - start_time) * 1000
    
    return final_results, total_time, emb_time, db_time

# ========================
# 3. RUN TEST SCENARIOS
# ========================

print("\n🚀 Running Scenarios...\n")

results = []
matches_data = []

for i, scenario in enumerate(ground_truth, 1):
    scenario_id = scenario['scenario_id']
    searcher = scenario['searcher']
    criteria = scenario['criteria']
    ground_truth_candidates = scenario['ground_truth_candidates']
    
    # Run matchmaking
    matches, latency, emb_time, db_time = matchmaking_approach_6(searcher, criteria, top_k=10)
    matches_data.append(matches)
    
    # Calculate metrics
    predicted_ids = [m['user_id'] for m in matches]
    gt_ids = [c['user_id'] for c in ground_truth_candidates if c['relevance_score'] > 0]
    
    ndcg = calculate_ndcg(predicted_ids, ground_truth_candidates, k=10)
    
    relevant_count = sum(1 for pid in predicted_ids if pid in gt_ids)
    precision = (relevant_count / len(predicted_ids)) * 100 if predicted_ids else 0
    recall = (relevant_count / len(gt_ids)) * 100 if gt_ids else 0
    
    jobs_str = criteria.get('jobs', [''])[0] if criteria.get('jobs') else 'Unknown'
    location_str = criteria.get('location', 'Unknown')
    
    print(f"Scenario {scenario_id}: {jobs_str} in {location_str}")
    print(f"   ⏱️  Latency: {latency:.0f}ms")
    print(f"   🎯 NDCG: {ndcg:.3f} | Precision: {precision:.1f}% | Recall: {recall:.1f}%\n")
    
    results.append({
        "scenario_id": scenario_id,
        "latency_ms": round(latency, 2),
        "embedding_time_ms": round(emb_time, 2),
        "db_time_ms": round(db_time, 2),
        "ndcg": round(ndcg, 3),
        "precision": round(precision, 1),
        "recall": round(recall, 1),
        "predicted_ids": predicted_ids,
        "ground_truth_ids": gt_ids
    })

# ========================
# 4. CALCULATE OVERALL METRICS
# ========================

avg_latency = sum(r['latency_ms'] for r in results) / len(results)
avg_ndcg = sum(r['ndcg'] for r in results) / len(results)
avg_precision = sum(r['precision'] for r in results) / len(results)
avg_recall = sum(r['recall'] for r in results) / len(results)

print("="*80)
print("📊 OVERALL RESULTS")
print("="*80)
print(f"Average Latency:   {avg_latency:.2f}ms")
print(f"Average NDCG:      {avg_ndcg:.3f}")
print(f"Average Precision: {avg_precision:.1f}%")
print(f"Average Recall:    {avg_recall:.1f}%")

# Save JSON results
output = {
    "approach": "Approach 6: Age Range + Semantic + Gaussian Age",
    "configuration": {
        "filters": ["gender", "religion", "age_range"],
        "semantic_weight": 0.8746,
        "age_weight": 0.1254,
        "older_base": 0.8938,
        "younger_base": 0.9091
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

with open("approach_6_results.json", 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ JSON Results saved to: approach_6_results.json")

# ========================
# 5. GENERATE HTML REPORT
# ========================

html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Approach 6: Age Range + Semantic + Gaussian Age - Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
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
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
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
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 40px;
            text-align: center;
        }}
        
        .hero-section h2 {{ 
            font-size: 4em; 
            margin-bottom: 10px; 
            font-weight: bold;
        }}
        
        .hero-section p {{
            font-size: 1.2em;
            opacity: 0.95;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
            padding: 25px;
            border-radius: 12px;
            border-left: 5px solid #f59e0b;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .stat-card h3 {{
            color: #92400e;
            font-size: 0.9em;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .stat-card .value {{
            color: #b45309;
            font-size: 2.5em;
            font-weight: bold;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            color: #d97706;
            font-size: 1.8em;
            margin-bottom: 20px;
            border-bottom: 3px solid #f59e0b;
            padding-bottom: 10px;
        }}
        
        .config-box {{
            background: #fffbeb;
            border: 2px solid #fbbf24;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 30px;
        }}
        
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .config-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #fcd34d;
        }}
        
        .config-label {{
            color: #92400e;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        
        .config-value {{
            color: #b45309;
            font-family: 'Courier New', monospace;
            font-size: 1.1em;
        }}
        
        .chart-container {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        
        .scenario {{
            background: #fefce8;
            margin-bottom: 40px;
            border: 2px solid #fde047;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .scenario h2 {{
            color: #ca8a04;
            margin-bottom: 20px;
            font-size: 1.5em;
        }}
        
        .metrics {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }}
        
        .metric {{
            background: white;
            padding: 12px 20px;
            border-radius: 8px;
            border: 2px solid #fde047;
            font-weight: 600;
        }}
        
        .comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 25px;
        }}
        
        .profile-card {{
            background: white;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 4px solid #d1d5db;
        }}
        
        .profile-card.match-yes {{
            border-left-color: #10b981;
            background-color: #d1fae5;
        }}
        
        .profile-card.match-no {{
            border-left-color: #ef4444;
            background-color: #fee2e2;
        }}
        
        .score-pill {{
            display: inline-block;
            background: #fbbf24;
            color: #78350f;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            margin-right: 5px;
        }}
        
        .insights-box {{
            background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
            border: 2px solid #f59e0b;
            padding: 30px;
            border-radius: 12px;
            margin-top: 40px;
        }}
        
        .insights-box h3 {{
            color: #92400e;
            margin-bottom: 20px;
            font-size: 1.5em;
        }}
        
        .insights-box ul {{
            list-style: none;
            padding: 0;
        }}
        
        .insights-box li {{
            padding: 12px 0;
            border-bottom: 1px solid #fde68a;
            color: #451a03;
            line-height: 1.8;
        }}
        
        .insights-box li:last-child {{
            border-bottom: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Approach 6: Age Range + Semantic + Gaussian Age</h1>
        <p class="subtitle">Hard filters (Gender, Religion, Age Range) → Semantic → Gaussian Age Scoring → Top 10</p>
        
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
                <h3>Precision</h3>
                <div class="value">{avg_precision:.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>Recall</h3>
                <div class="value">{avg_recall:.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>Avg Latency</h3>
                <div class="value">{avg_latency:.0f}ms</div>
            </div>
        </div>
        
        <!-- Configuration Section -->
        <div class="section">
            <h2 class="section-title">⚙️ Algorithm Configuration</h2>
            <div class="config-box">
                <h3 style="margin-bottom: 20px; color: #92400e;">Double Age Consideration</h3>
                <div class="config-grid">
                    <div class="config-item">
                        <div class="config-label">Hard Filters</div>
                        <div class="config-value">Gender, Religion, Age Range</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">Semantic Weight</div>
                        <div class="config-value">87.46%</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">Age Weight</div>
                        <div class="config-value">12.54%</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">Age Range</div>
                        <div class="config-value">-3 to +5 years</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">Older Decay</div>
                        <div class="config-value">10.62% per year</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">Younger Decay</div>
                        <div class="config-value">9.09% per year</div>
                    </div>
                </div>
                <p style="margin-top: 20px; color: #78350f; line-height: 1.8;">
                    <strong>Approach:</strong> This method combines hard age filtering with Gaussian age scoring. 
                    First, incompatible ages are filtered out (-3/+5 range), then remaining candidates are scored 
                    with fine-grained age preferences. Final score: 87.46% semantic + 12.54% age.
                </p>
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

# Add each scenario
for i, result in enumerate(results):
    scenario = ground_truth[i]
    matches = matches_data[i]
    gt_ids = [c['user_id'] for c in scenario['ground_truth_candidates'] if c['relevance_score'] > 0]
    
    jobs_str = scenario['criteria'].get('jobs', ['Unknown'])[0]
    age_info = f", Age: {scenario['filters'].get('preferred_age', scenario['searcher']['age'])}"
    
    html_content += f"""
        <div class="scenario">
            <h2>Scenario {result['scenario_id']}: {jobs_str}</h2>
            <p><strong>Filters:</strong> {scenario['filters']['gender']}, {scenario['filters']['religion']}{age_info}, Age Range: {scenario['filters'].get('preferred_age', scenario['searcher']['age'])-3} to {scenario['filters'].get('preferred_age', scenario['searcher']['age'])+5}</p>
            <p><strong>Location:</strong> {scenario['criteria'].get('location', 'Any')}</p>
            
            <div class="metrics">
                <div class="metric">⏱️ Latency: {result['latency_ms']:.0f}ms</div>
                <div class="metric">🎯 NDCG: {result['ndcg']:.3f}</div>
                <div class="metric">✅ Precision: {result['precision']:.1f}%</div>
                <div class="metric">📊 Recall: {result['recall']:.1f}%</div>
            </div>
            
            <div class="comparison">
                <div>
                    <h3>✅ Predicted Results (Top 10)</h3>
    """
    
    for p in matches[:10]:
        is_match = p['user_id'] in gt_ids
        cls = "match-yes" if is_match else "match-no"
        html_content += f"""
                    <div class="profile-card {cls}">
                        <strong>{p['name']} (ID: {p['user_id']})</strong><br>
                        {p['job_title']}, {p['location']}<br>
                        Age: {p['age']}<br>
                        <div style="margin-top:5px;">
                            <span class="score-pill">Final: {p['final_score']}</span>
                            <span class="score-pill">Semantic: {p['semantic_score']}</span>
                            <span class="score-pill">Age: {p['age_score']}</span>
                        </div>
                    </div>
        """
        
    html_content += """
                </div>
                <div>
                    <h3>🎯 Ground Truth (Expected)</h3>
    """
    
    for gt in scenario['ground_truth_candidates']:
        if gt['relevance_score'] > 0:
            is_found = gt['user_id'] in [p['user_id'] for p in matches]
            cls = "match-yes" if is_found else "match-no"
            html_content += f"""
                    <div class="profile-card {cls}">
                        <strong>{gt['name']} (ID: {gt['user_id']})</strong><br>
                        Relevance: {gt['relevance_score']}<br>
                        {"✅ Found" if is_found else "❌ Not Found"}
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
                <li><strong>Double Age Consideration:</strong> Age is both filtered (hard range) AND scored (Gaussian) for optimal balance.</li>
                <li><strong>Balanced Weights:</strong> 87.46% semantic + 12.54% age scoring (Bayesian optimized parameters).</li>
                <li><strong>Smart Age Handling:</strong> Filter removes incompatible ages (-3/+5), then Gaussian scoring fine-tunes within range.</li>
                <li><strong>Best of Both Worlds:</strong> Combines Approach 1's age scoring with Approach 2's age filtering.</li>
                <li><strong>Performance:</strong> Average NDCG of {avg_ndcg:.3f} with {avg_precision:.1f}% precision in {avg_latency:.0f}ms.</li>
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
                        backgroundColor: 'rgba(245, 158, 11, 0.7)',
                        borderColor: 'rgba(217, 119, 6, 1)',
                        borderWidth: 2
                    }},
                    {{
                        label: 'Precision (%)',
                        data: {[r['precision']/100 for r in results]},
                        backgroundColor: 'rgba(251, 191, 36, 0.7)',
                        borderColor: 'rgba(245, 158, 11, 1)',
                        borderWidth: 2
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1,
                        ticks: {{
                            callback: function(value) {{
                                return (value * 100).toFixed(0) + '%';
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'NDCG vs Precision by Scenario',
                        font: {{ size: 16 }}
                    }},
                    legend: {{
                        display: true,
                        position: 'top'
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

with open("approach_6_report.html", 'w', encoding='utf-8') as f:
    f.write(html_content)

print("✅ HTML Report saved: approach_6_report.html")

conn.close()
