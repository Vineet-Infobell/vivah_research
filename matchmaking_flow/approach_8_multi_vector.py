"""
APPROACH 8: Multi-Vector Approach with Separate Embeddings
-----------------------------------------------------------
Flow:
1. Hard Filters: Gender, Religion
2. Generate 3 Separate Embeddings:
   - Profession Embedding (from job_title query)
   - Education Embedding (from education query)
   - Location Embedding (from location query)
3. Calculate 3 Similarity Scores for each candidate
4. Weighted Combination: 50% Profession + 30% Education + 20% Location
5. Top 10 Results

Concept: Fine-grained control over each attribute's importance.
Research-oriented approach with separate vector spaces.
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
print("🎯 APPROACH 8: Multi-Vector with Separate Embeddings")
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

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot_product / (mag1 * mag2)

# ========================
# 2. MATCHMAKING FUNCTION
# ========================

# Attribute weights (optimized)
PROFESSION_WEIGHT = 0.40  # 40% - Most important
EDUCATION_WEIGHT = 0.35   # 35% - Almost equal importance
LOCATION_WEIGHT = 0.15    # 15% - Lower priority
AGE_WEIGHT = 0.10         # 10% - Age consideration

# Gaussian Age Scoring Function
def gaussian_age_score(preferred_age, candidate_age):
    """
    Optimized Gaussian scoring with asymmetric decay.
    """
    age_diff = candidate_age - preferred_age
    if age_diff >= 0:  # Candidate is older
        base = 0.8938
        score = base ** abs(age_diff)
    else:  # Candidate is younger
        base = 0.9091
        score = base ** abs(age_diff)
    return score

def matchmaking_approach_8(searcher, criteria, top_k=10):
    """
    Approach 8: Multi-Vector with separate embeddings for each attribute
    """
    start_time = time.time()
    
    # Extract queries with more context
    jobs = criteria.get('jobs', ['Software Developer'])
    educations = criteria.get('educations', ['B.Tech'])
    location = criteria.get('location', 'Bangalore')
    
    # Create richer query strings with context
    profession_query = f"{jobs[0]} professional working in technology"
    education_query = f"Educated with {educations[0]} degree in engineering or computer science"
    location_query = f"Living in {location} city in India"
    
    # Generate 3 separate embeddings
    emb_start = time.time()
    
    profession_vec = create_embedding(profession_query)
    education_vec = create_embedding(education_query)
    location_vec = create_embedding(location_query)
    
    emb_time = (time.time() - emb_start) * 1000
    
    if not profession_vec or not education_vec or not location_vec:
        return [], 0, 0, 0
    
    # PREPARE FILTERS (only Gender + Religion)
    filters = {
        'gender': 'Female' if searcher['gender'] == 'Male' else 'Male',
        'religion': criteria.get('religion', searcher['religion'])
    }
    
    # Get preferred age for scoring
    preferred_age = criteria.get('preferred_age', searcher['age'])
    
    # BUILD SQL QUERY with minimal hard filters
    where_clauses = [
        f"gender = '{filters['gender']}'",
        f"religion = '{filters['religion']}'"
    ]
    
    where_sql = " AND ".join(where_clauses)
    
    # Get candidates with their 3 separate vectors
    sql = f"""
        SELECT 
            user_id, name, gender, age, religion, location, education, job_title,
            profession_vector, education_vector, location_vector
        FROM users
        WHERE {where_sql}
        AND profession_vector IS NOT NULL 
        AND education_vector IS NOT NULL 
        AND location_vector IS NOT NULL
        LIMIT 200;
    """
    
    db_start = time.time()
    cur.execute(sql)
    candidates = cur.fetchall()
    db_time = (time.time() - db_start) * 1000
    
    if not candidates:
        return [], 0, 0, 0
    
    # Calculate multi-vector scores
    scored_candidates = []
    
    for c in candidates:
        user_id, name, gender, age, religion, location, education, job_title, prof_vec_str, edu_vec_str, loc_vec_str = c
        
        # Parse separate vectors
        prof_user_vec = json.loads(prof_vec_str) if isinstance(prof_vec_str, str) else prof_vec_str
        edu_user_vec = json.loads(edu_vec_str) if isinstance(edu_vec_str, str) else edu_vec_str
        loc_user_vec = json.loads(loc_vec_str) if isinstance(loc_vec_str, str) else loc_vec_str
        
        # Calculate 3 separate similarity scores with matching vector spaces
        profession_sim = cosine_similarity(profession_vec, prof_user_vec)
        education_sim = cosine_similarity(education_vec, edu_user_vec)
        location_sim = cosine_similarity(location_vec, loc_user_vec)
        
        # Calculate age score
        age_score = gaussian_age_score(preferred_age, age)
        
        # Weighted combination: 40% Prof + 35% Edu + 15% Loc + 10% Age
        final_score = (
            profession_sim * PROFESSION_WEIGHT +
            education_sim * EDUCATION_WEIGHT +
            location_sim * LOCATION_WEIGHT +
            age_score * AGE_WEIGHT
        )
        
        scored_candidates.append({
            "user_id": user_id,
            "name": name,
            "age": age,
            "location": location,
            "education": education,
            "job_title": job_title,
            "profession_score": round(float(profession_sim), 4),
            "education_score": round(float(education_sim), 4),
            "location_score": round(float(location_sim), 4),
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
    matches, latency, emb_time, db_time = matchmaking_approach_8(searcher, criteria, top_k=10)
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
    "approach": "Approach 8: Multi-Vector with Separate Embeddings",
    "configuration": {
        "filters": ["gender", "religion"],
        "profession_weight": PROFESSION_WEIGHT,
        "education_weight": EDUCATION_WEIGHT,
        "location_weight": LOCATION_WEIGHT,
        "age_weight": AGE_WEIGHT,
        "ranking": "multi_vector_weighted_with_age"
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

with open("approach_8_results.json", 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ JSON Results saved to: approach_8_results.json")

# ========================
# 5. GENERATE HTML REPORT
# ========================

html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Approach 8: Multi-Vector - Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
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
            background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
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
            background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
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
            background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
            padding: 25px;
            border-radius: 12px;
            border-left: 5px solid #8b5cf6;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .stat-card h3 {{
            color: #5b21b6;
            font-size: 0.9em;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .stat-card .value {{
            color: #6d28d9;
            font-size: 2.5em;
            font-weight: bold;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            color: #6d28d9;
            font-size: 1.8em;
            margin-bottom: 20px;
            border-bottom: 3px solid #8b5cf6;
            padding-bottom: 10px;
        }}
        
        .config-box {{
            background: #faf5ff;
            border: 2px solid #c4b5fd;
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
            border: 1px solid #ddd6fe;
        }}
        
        .config-label {{
            color: #5b21b6;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        
        .config-value {{
            color: #6d28d9;
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
            background: #fdf4ff;
            margin-bottom: 40px;
            border: 2px solid #e9d5ff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .scenario h2 {{
            color: #7c3aed;
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
            border: 2px solid #e9d5ff;
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
            background: #c4b5fd;
            color: #5b21b6;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            margin-right: 5px;
        }}
        
        .insights-box {{
            background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
            border: 2px solid #8b5cf6;
            padding: 30px;
            border-radius: 12px;
            margin-top: 40px;
        }}
        
        .insights-box h3 {{
            color: #5b21b6;
            margin-bottom: 20px;
            font-size: 1.5em;
        }}
        
        .insights-box ul {{
            list-style: none;
            padding: 0;
        }}
        
        .insights-box li {{
            padding: 12px 0;
            border-bottom: 1px solid #ddd6fe;
            color: #3730a3;
            line-height: 1.8;
        }}
        
        .insights-box li:last-child {{
            border-bottom: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Approach 8: Multi-Vector with Separate Embeddings</h1>
        <p class="subtitle">Hard filters (Gender, Religion) → 3 Separate Embeddings (Profession, Education, Location) → Weighted Combination → Top 10</p>
        
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
                <h3 style="margin-bottom: 20px; color: #5b21b6;">Multi-Vector Weighted Approach</h3>
                <div class="config-grid">
                    <div class="config-item">
                        <div class="config-label">Hard Filters</div>
                        <div class="config-value">Gender, Religion Only</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">Profession Weight</div>
                        <div class="config-value">40%</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">Education Weight</div>
                        <div class="config-value">35%</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">Location Weight</div>
                        <div class="config-value">15%</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">Age Weight</div>
                        <div class="config-value">10%</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">Embedding Model</div>
                        <div class="config-value">gemini-embedding-001</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">Vector Dimension</div>
                        <div class="config-value">1152</div>
                    </div>
                </div>
                <p style="margin-top: 20px; color: #4c1d95; line-height: 1.8;">
                    <strong>Approach:</strong> This optimized multi-vector method generates separate embeddings with rich context 
                    for profession, education, and location. Each attribute gets its own vector space and optimized weight. 
                    Final score: 40% profession + 35% education + 15% location + 10% age (Gaussian). 
                    Provides fine-grained control over attribute importance.
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
    
    html_content += f"""
        <div class="scenario">
            <h2>Scenario {result['scenario_id']}: {jobs_str}</h2>
            <p><strong>Filters:</strong> {scenario['filters']['gender']}, {scenario['filters']['religion']}</p>
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
                            <span class="score-pill">Prof: {p['profession_score']}</span>
                            <span class="score-pill">Edu: {p['education_score']}</span>
                            <span class="score-pill">Loc: {p['location_score']}</span>
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
                <li><strong>Optimized Multi-Vector:</strong> Separate embeddings with rich context for profession (40%), education (35%), location (15%), and age (10%).</li>
                <li><strong>Rich Query Context:</strong> Enhanced embeddings with descriptive context instead of single words for better semantic matching.</li>
                <li><strong>Minimal Hard Filters:</strong> Only gender and religion filtered - maximum flexibility for semantic matching.</li>
                <li><strong>Balanced Weights:</strong> Optimized weights with profession and education getting higher priority, plus age consideration.</li>
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
                        backgroundColor: 'rgba(139, 92, 246, 0.7)',
                        borderColor: 'rgba(109, 40, 217, 1)',
                        borderWidth: 2
                    }},
                    {{
                        label: 'Precision (%)',
                        data: {[r['precision']/100 for r in results]},
                        backgroundColor: 'rgba(196, 181, 253, 0.7)',
                        borderColor: 'rgba(139, 92, 246, 1)',
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

with open("approach_8_report.html", 'w', encoding='utf-8') as f:
    f.write(html_content)

print("✅ HTML Report saved: approach_8_report.html")

conn.close()
