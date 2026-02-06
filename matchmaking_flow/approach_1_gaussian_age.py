"""
APPROACH 1: Semantic Search + Gaussian Age Scoring
-------------------------------------------------
Flow:
1. Hard Filters: Gender + Religion
2. Semantic Search: Profession + Education + Location
3. Gaussian Age Scoring: 
   - Increasing: 100% → 80% → 64% → 51.2% (20% deduction)
   - Decreasing: 100% → 60% → 36% → 21.6% (40% deduction)
4. Top 10 Results
"""

import psycopg2
import json
import time
import webbrowser
from google import genai
from google.genai import types
from google.oauth2 import service_account
import os
from pathlib import Path
from dotenv import load_dotenv

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

print("✅ GenAI Client Ready\n")

print("="*80)
print("🎯 APPROACH 1: Semantic Search + Gaussian Age Scoring")
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

# ========================
# 3. LOAD GROUND TRUTH
# ========================
with open('ground_truth_refined.json', 'r', encoding='utf-8') as f:
    ground_truth = json.load(f)

# NDCG Calculation Helpers
import math

def calculate_dcg(relevances):
    dcg = 0
    for i, rel in enumerate(relevances):
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg

def calculate_ndcg(predicted_ids, ground_truth_candidates, k=10):
    # Map ID -> Relevance Score (Default 0)
    relevance_map = {c['user_id']: c['relevance_score'] for c in ground_truth_candidates}
    
    # Get relevances for predicted list (up to k)
    relevances = [relevance_map.get(pid, 0) for pid in predicted_ids[:k]]
    
    # Calculate DCG
    actual_dcg = calculate_dcg(relevances)
    
    # Calculate Ideal DCG (IDCG) - sort known relevant items descending
    ideal_relevances = sorted(list(relevance_map.values()), reverse=True)[:k]
    ideal_dcg = calculate_dcg(ideal_relevances)
    
    if ideal_dcg == 0: return 0.0
    return actual_dcg / ideal_dcg

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

def gaussian_age_score(target_age: int, candidate_age: int) -> float:
    """
    Calculate age compatibility score using Gaussian decay
    BALANCED WEIGHTS (More realistic than pure Bayesian):
    - Older: 10.62% decay per year (Base 0.8938)
    - Younger: 9.09% decay per year (Base 0.9091)
    """
    if target_age == candidate_age:
        return 1.0
    
    diff = candidate_age - target_age
    
    if diff > 0:  # Candidate is older
        # Balanced: 10.62% decay per year (Base 0.8938)
        score = 1.0 * (0.8938 ** diff)
    else:  # Candidate is younger
        # Balanced: 9.09% decay per year (Base 0.9091)
        score = 1.0 * (0.9091 ** abs(diff))
    
    return score

def search_matches(query: str, filters: dict, target_age: int, top_k: int = 10):
    """
    Search for matches using semantic search + Gaussian age scoring
    Returns detailed scoring for transparency
    """
    start_time = time.time()
    
    # Generate query embedding
    emb_start = time.time()
    query_vec = create_embedding(query)
    emb_time = (time.time() - emb_start) * 1000
    
    if not query_vec:
        return None, 0, 0
    
    vec_str = '[' + ','.join(map(str, query_vec)) + ']'
    
    # Build SQL with hard filters (Gender + Religion only)
    sql = f"""
        SELECT 
            user_id, name, gender, age, religion, location, education, job_title,
            1 - (user_vector <=> '{vec_str}'::vector) as similarity
        FROM users
        WHERE gender = '{filters['gender']}'
          AND religion = '{filters['religion']}'
        ORDER BY user_vector <=> '{vec_str}'::vector
        LIMIT 20;
    """
    
    # Execute search
    db_start = time.time()
    cur.execute(sql)
    candidates = cur.fetchall()
    db_time = (time.time() - db_start) * 1000
    
    # Apply Gaussian age scoring with detailed breakdown
    scored_results = []
    for candidate in candidates:
        user_id, name, gender, age, religion, location, education, job_title, similarity = candidate
        
        age_score = gaussian_age_score(target_age, age)
        # Balanced weights: 87.46% semantic + 12.54% age
        final_score = (similarity * 0.8746) + (age_score * 0.1254)
        
        scored_results.append({
            "user_id": user_id,
            "name": name,
            "age": age,
            "location": location,
            "education": education,
            "job_title": job_title,
            "semantic_score": round(float(similarity), 4),
            "age_score": round(age_score, 4),
            "final_score": round(final_score, 4)
        })
    
    # Sort by final score and take top K
    scored_results.sort(key=lambda x: x['final_score'], reverse=True)
    top_results = scored_results[:top_k]
    
    total_time = (time.time() - start_time) * 1000
    
    return top_results, total_time, emb_time, db_time

# Run all test scenarios
results = []
matches_data = []  # Store matches for HTML generation
total_latency = 0

print("\n🚀 Running Test Scenarios...\n")

for scenario in ground_truth:
    print(f"Scenario {scenario['scenario_id']}: {scenario['scenario_name']}")
    
    # Get target age - use preferred_age from filters if available, else searcher's age
    target_age = scenario['filters'].get('preferred_age', scenario['searcher']['age'])
    
    # Search
    matches, latency, emb_time, db_time = search_matches(
        scenario['query'],
        scenario['filters'],
        target_age,
        top_k=10
    )
    
    if matches is None:
        print(f"   ❌ Failed\n")
        continue
    
    # Store matches for HTML generation
    matches_data.append(matches)
    
    # Calculate accuracy (how many ground truth IDs in top 10)
    # Calculate Metrics (NDCG, Precision, Recall)
    # Get all GT candidates (up to 20)
    gt_candidates = scenario['ground_truth_candidates']
    gt_ids = set([m['user_id'] for m in gt_candidates if m['relevance_score'] > 0]) # All relevant IDs
    
    predicted_id_list = [m['user_id'] for m in matches]
    predicted_set = set(predicted_id_list)
    
    # Hits (Any Tier)
    hits = gt_ids.intersection(predicted_set)
    
    # Precision@10: Fraction of relevant items in top 10
    precision = (len(hits) / 10) * 100
    
    # Recall@10: Fraction of total relevant items found
    recall = (len(hits) / len(gt_ids)) * 100 if len(gt_ids) > 0 else 0
    
    # NDCG@10
    ndcg = calculate_ndcg(predicted_id_list, gt_candidates, k=10)
    
    results.append({
        "scenario_id": scenario['scenario_id'],
        "scenario_name": scenario['scenario_name'],
        "latency_ms": round(latency, 2),
        "embedding_time_ms": round(emb_time, 2),
        "db_time_ms": round(db_time, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "ndcg": round(ndcg, 3),
        "top_10_ids": predicted_id_list
    })
    
    total_latency += latency
    
    print(f"   ⏱️  Latency: {latency:.2f}ms")
    print(f"   🎯 NDCG: {ndcg:.3f} | Precision: {precision:.1f}% | Recall: {recall:.1f}%")
    print(f"   Top 3: {[m['user_id'] for m in matches[:3]]}\n")

# Summary
# Summary
avg_latency = total_latency / len(results) if results else 0
avg_ndcg = sum(r['ndcg'] for r in results) / len(results) if results else 0
avg_precision = sum(r['precision'] for r in results) / len(results) if results else 0
avg_recall = sum(r['recall'] for r in results) / len(results) if results else 0

print("="*80)
print("📊 OVERALL RESULTS (Using Refined Ground Truth)")
print("="*80)
print(f"Average Latency:   {avg_latency:.2f}ms")
print(f"Average NDCG@10:   {avg_ndcg:.3f} (Rank Quality)")
print(f"Average Precision: {avg_precision:.1f}% (Relevance)")
print(f"Average Recall:    {avg_recall:.1f}% (Coverage)")
print(f"Total Scenarios:   {len(results)}")

# ========================
# 4. SAVE JSON RESULTS
# ========================
json_output = {
    "approach": "Approach 1: Semantic Search + Gaussian Age Scoring",
    "summary": {
        "average_latency_ms": round(avg_latency, 2),
        "average_ndcg": round(avg_ndcg, 3),
        "average_precision": round(avg_precision, 2),
        "average_recall": round(avg_recall, 2),
        "total_scenarios": len(results)
    },
    "scenarios": []
}

for i, scenario in enumerate(ground_truth):
    result = results[i]
    scenario_data = {
        "scenario_id": scenario['scenario_id'],
        "scenario_name": scenario['scenario_name'],
        "query": scenario['query'],
        "searcher": scenario['searcher'],
        "filters": scenario['filters'],
        "performance": {
            "latency_ms": result['latency_ms'],
            "ndcg": result['ndcg'],
            "precision": result['precision'],
            "recall": result['recall']
        },
        "ground_truth_ids": [m['user_id'] for m in scenario['ground_truth_candidates']],
        "predicted_top_10": result['top_10_ids'],
        "predicted_profiles": matches_data[i]
    }
    json_output['scenarios'].append(scenario_data)

json_filename = 'approach_1_results.json'
with open(json_filename, 'w', encoding='utf-8') as f:
    json.dump(json_output, f, indent=2, ensure_ascii=False)

print(f"\n✅ JSON Results saved to: {json_filename}")

# Generate HTML Report
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Approach 1: Gaussian Age Scoring - Results</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{ 
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-align: center;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
        .summary-card h3 {{ font-size: 2em; margin-bottom: 5px; }}
        .summary-card p {{ opacity: 0.9; }}
        .scenario {{
            margin-bottom: 50px;
            border: 2px solid #e0e0e0;
            border-radius: 15px;
            padding: 30px;
            background: #f9f9f9;
        }}
        .scenario-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
        }}
        .scenario-header h2 {{ margin-bottom: 10px; }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }}
        .metric-value {{ font-size: 1.8em; font-weight: bold; color: #667eea; }}
        .metric-label {{ color: #666; font-size: 0.9em; margin-top: 5px; }}
        .profiles-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 20px;
        }}
        .profile-section h3 {{
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        .profile-card {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 12px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .profile-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .profile-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .profile-name {{
            font-weight: bold;
            color: #667eea;
            font-size: 1.1em;
        }}
        .profile-id {{
            background: #667eea;
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
        }}
        .profile-details {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            color: #666;
            font-size: 0.9em;
        }}
        .detail-item {{
            display: flex;
            gap: 5px;
        }}
        .detail-label {{ font-weight: 600; color: #333; }}
        .match-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
            margin-top: 8px;
        }}
        .match-yes {{ background: #10b981; color: white; }}
        .match-no {{ background: #ef4444; color: white; }}
        @media (max-width: 768px) {{
            .profiles-comparison {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Approach 1: Gaussian Age Scoring</h1>
        <p class="subtitle">Semantic Search + Gaussian Age Decay (20% increasing, 40% decreasing)</p>
        
        <!-- Approach Logic -->
        <div style="background: #f0f7ff; border-left: 4px solid #667eea; padding: 20px; margin-bottom: 30px; border-radius: 8px;">
            <h3 style="color: #667eea; margin-bottom: 15px;">📋 Approach Logic</h3>
            <ol style="line-height: 1.8; color: #333;">
                <li><strong>Hard Filters:</strong> Gender + Religion (NO age filter)</li>
                <li><strong>Semantic Search:</strong> Query embedding generated using <code>gemini-embedding-001</code> (1152D)</li>
                <li><strong>Vector Search:</strong> Top 100 candidates retrieved using cosine similarity</li>
                <li><strong>Gaussian Age Scoring:</strong>
                    <ul style="margin-left: 20px; margin-top: 8px;">
                        <li>Same age = 100%</li>
                        <li>Older candidates: 20% deduction per year (100% → 80% → 64% → 51.2%...)</li>
                        <li>Younger candidates: 40% deduction per year (100% → 60% → 36% → 21.6%...)</li>
                    </ul>
                </li>
                <li><strong>Final Score:</strong> (Semantic × 0.8) + (Age × 0.2) = Weighted Sum</li>
                <li><strong>Output:</strong> Top 10 profiles ranked by final score</li>
            </ol>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <h3>{avg_latency:.1f}ms</h3>
                <p>Avg Latency</p>
            </div>
            <div class="summary-card">
                <h3>{avg_ndcg:.3f}</h3>
                <p>Avg NDCG</p>
            </div>
            <div class="summary-card">
                <h3>{len(results)}</h3>
                <p>Scenarios</p>
            </div>
            <div class="summary-card">
                <h3>1152D</h3>
                <p>Embedding Size</p>
            </div>
        </div>
"""

for i, result in enumerate(results):
    scenario = ground_truth[i]
    
    # Get matched IDs
    # Get matched IDs for highlighting
    gt_candidates = scenario['ground_truth_candidates']
    gt_ids = set([m['user_id'] for m in gt_candidates if m['relevance_score'] > 0])
    
    pred_ids = set(result['top_10_ids']) 
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
                <p style="margin-top: 10px;">
                    <strong>Searcher:</strong> {scenario['searcher']['gender']}, 
                    Age {scenario['searcher']['age']}, 
                    {scenario['searcher']['religion']}
                </p>
                <p><strong>Looking for:</strong> {scenario['filters']['gender']}, {scenario['filters']['religion']}{age_info}</p>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{result['latency_ms']}ms</div>
                    <div class="metric-label">Total Latency</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result['embedding_time_ms']}ms</div>
                    <div class="metric-label">Embedding</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result['db_time_ms']}ms</div>
                    <div class="metric-label">Database</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result['ndcg']}</div>
                    <div class="metric-label">NDCG Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result['precision']}%</div>
                    <div class="metric-label">Precision</div>
                </div>
            </div>
            
            <div class="profiles-comparison">
                <div class="profile-section">
                    <h3>✅ Expected (Ground Truth)</h3>
"""
    
    # Ground truth profiles (Show top 10 best out of 20 for display)
    # Ground truth profiles (Show ALL 20)
    for idx, profile in enumerate(scenario['ground_truth_candidates'], 1):
        is_matched = profile['user_id'] in matched_ids
        badge_class = "match-yes" if is_matched else "match-no"
        badge_text = "✓ Found" if is_matched else "✗ Missed"
        
        html_content += f"""
                    <div class="profile-card">
                        <div class="profile-header">
                            <span class="profile-name">#{idx}. {profile['name']}</span>
                            <span class="profile-id">ID: {profile['user_id']}</span>
                        </div>
                        <div class="profile-details">
                            <div class="detail-item">
                                <span class="detail-label">Job:</span>
                                <span>{profile['job_title']}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Age:</span>
                                <span>{profile['age']}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Edu:</span>
                                <span>{profile['education']}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">City:</span>
                                <span>{profile['location']}</span>
                            </div>
                        </div>
                        <span class="match-badge {badge_class}">{badge_text}</span>
                    </div>
"""
    
    html_content += """
                </div>
                <div class="profile-section">
                    <h3>🔍 Predicted (Our Results)</h3>
"""
    
    # Use stored matches data for this scenario
    scenario_matches = matches_data[i]
    for idx, profile in enumerate(scenario_matches[:10], 1):
        is_matched = profile['user_id'] in matched_ids
        badge_class = "match-yes" if is_matched else "match-no"
        badge_text = "✓ Match" if is_matched else "✗ Extra"
        
        html_content += f"""
                    <div class="profile-card">
                        <div class="profile-header">
                            <span class="profile-name">#{idx}. {profile['name']}</span>
                            <span class="profile-id">ID: {profile['user_id']}</span>
                        </div>
                        <div class="profile-details">
                            <div class="detail-item">
                                <span class="detail-label">Job:</span>
                                <span>{profile['job_title']}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Age:</span>
                                <span>{profile['age']}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Edu:</span>
                                <span>{profile['education']}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">City:</span>
                                <span>{profile['location']}</span>
                            </div>
                        </div>
                        <div style="margin-top: 12px; padding: 10px; background: #f0f7ff; border-radius: 6px; font-size: 0.85em;">
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; text-align: center;">
                                <div>
                                    <div style="font-weight: bold; color: #667eea;">{profile['semantic_score']}</div>
                                    <div style="color: #666; font-size: 0.9em;">Semantic</div>
                                </div>
                                <div>
                                    <div style="font-weight: bold; color: #f59e0b;">{profile['age_score']}</div>
                                    <div style="color: #666; font-size: 0.9em;">Age Score</div>
                                </div>
                                <div>
                                    <div style="font-weight: bold; color: #10b981;">{profile['final_score']}</div>
                                    <div style="color: #666; font-size: 0.9em;">Final</div>
                                </div>
                            </div>
                        </div>
                        <span class="match-badge {badge_class}">{badge_text}</span>
                    </div>
"""
    
    html_content += """
                </div>
            </div>
        </div>
"""

html_content += """
    </div>
</body>
</html>
"""

# Save HTML report
report_path = "approach_1_report.html"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n✅ HTML Report saved to: {report_path}")

# Open in browser
import os
abs_path = os.path.abspath(report_path)
webbrowser.open('file://' + abs_path)
print(f"🌐 Opening report in browser...")

cur.close()
conn.close()
