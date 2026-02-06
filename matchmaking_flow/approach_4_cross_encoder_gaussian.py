"""
APPROACH 4: Semantic + Gaussian + Cross-Encoder
-----------------------------------------------
Flow:
1. Hard Filters: Gender + Religion ONLY (Soft Age/Location)
2. Semantic Search: Top 50 results
3. Gaussian Scoring: Adjust scores based on Age Match
4. Cross-Encoder: Re-rank Top 20 from Gaussian step
5. Return Top 10
"""

import psycopg2
import json
import time
import math
from sentence_transformers import CrossEncoder
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

from pathlib import Path

from google.oauth2 import service_account

load_dotenv()
env_path = Path('../../vivah_api/.env')
if env_path.exists(): load_dotenv(env_path)

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
print("⏳ Loading Cross-Encoder...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✅ Ready!")

print("="*80)
print("🎯 APPROACH 4: Semantic + Gaussian + Cross-Encoder")
print("="*80)

conn = psycopg2.connect(host="localhost", port="5433", database="postgres", user="postgres", password="matchpass")
cur = conn.cursor()

# Load ground truth with refined version
with open("ground_truth_refined.json", 'r', encoding='utf-8') as f:
    ground_truth = json.load(f)

# NDCG Calculation
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

def create_embedding(text):
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=str(text),
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=1152)
        )
        return response.embeddings[0].values
    except: return None

def gaussian_age_score(target_age, candidate_age):
    """
    Balanced age decay (from optimization)
    - Older: 10.62% decay per year (Base 0.8938)
    - Younger: 9.09% decay per year (Base 0.9091)
    """
    diff = candidate_age - target_age
    if diff > 0: return 1.0 * (0.8938 ** diff)  # Older
    else: return 1.0 * (0.9091 ** abs(diff))   # Younger

def search_matches(query, filters, target_age, top_k=10):
    start_time = time.time()
    
    # 1. Embedding
    query_vec = create_embedding(query)
    vec_str = '[' + ','.join(map(str, query_vec)) + ']'
    
    # 2. SQL - Gender/Religion Filters ONLY
    sql = f"""
        SELECT user_id, name, gender, age, religion, location, education, job_title,
            1 - (user_vector <=> '{vec_str}'::vector) as similarity
        FROM users
        WHERE gender = '{filters['gender']}' AND religion = '{filters['religion']}'
        ORDER BY user_vector <=> '{vec_str}'::vector
        LIMIT 50;
    """
    cur.execute(sql)
    candidates = cur.fetchall()
    
    # 3. Gaussian Scoring (Stage 1: Top 50 → Top 20)
    scored_candidates = []
    for c in candidates:
        user_id, name, gender, age, religion, location, education, job_title, similarity = c
        age_score = gaussian_age_score(target_age, age)
        # Stage 1: Balanced weights 87.46% semantic + 12.54% age
        final_score = (float(similarity) * 0.8746) + (age_score * 0.1254)
        
        scored_candidates.append({
            "user_id": user_id, "name": name, "age": age, 
            "location": location, "education": education, "job_title": job_title,
            "gaussian_score": final_score,
            "profile_text": f"{job_title} with {education} in {location}. Age {age}."
        })
        
    # Sort and take top 20 for re-ranking
    scored_candidates.sort(key=lambda x: x['gaussian_score'], reverse=True)
    top_20 = scored_candidates[:20]
    
    # 4. Cross-Encoder Re-ranking (Stage 2: Top 20 → Top 10)
    pairs = [[query, c['profile_text']] for c in top_20]
    raw_ce_scores = cross_encoder.predict(pairs)
    
    # Apply sigmoid normalization
    import numpy as np
    ce_scores = 1 / (1 + np.exp(-raw_ce_scores))
    
    # Combine CE + Age (87.46% CE + 12.54% Age)
    for i, score in enumerate(ce_scores):
        ce_score_norm = float(score)
        age_score = gaussian_age_score(target_age, top_20[i]['age'])
        final_combined = (ce_score_norm * 0.8746) + (age_score * 0.1254)
        
        top_20[i]['ce_score'] = round(ce_score_norm, 4)
        top_20[i]['age_score'] = round(age_score, 4)
        top_20[i]['final_combined_score'] = round(final_combined, 4)
        
    top_20.sort(key=lambda x: x['final_combined_score'], reverse=True)
    
    return top_20[:top_k], (time.time() - start_time) * 1000

# Run Test
results = []
matches_data = []
total_latency = 0

print("\n🚀 Running Scenarios...\n")
for scenario in ground_truth:
    print(f"Scenario {scenario['scenario_id']}: {scenario['scenario_name']}")
    target_age = scenario['filters'].get('preferred_age', scenario['searcher']['age'])
    
    matches, latency = search_matches(scenario['query'], scenario['filters'], target_age)
    matches_data.append(matches)
    
    # Calculate metrics
    gt_candidates = scenario['ground_truth_candidates']
    gt_ids = set([m['user_id'] for m in gt_candidates if m['relevance_score'] > 0])
    pred_ids = [m['user_id'] for m in matches]
    pred_ids_set = set(pred_ids)
    
    overlap = len(gt_ids.intersection(pred_ids_set))
    precision = (overlap / 10) * 100
    recall = (overlap / len(gt_ids)) * 100 if len(gt_ids) > 0 else 0
    ndcg = calculate_ndcg(pred_ids, gt_candidates, k=10)
    
    results.append({
        "scenario_id": scenario['scenario_id'],
        "scenario_name": scenario['scenario_name'],
        "latency_ms": round(latency, 2),
        "ndcg": round(ndcg, 3),
        "precision": round(precision, 1),
        "recall": round(recall, 1)
    })
    
    total_latency += latency
    print(f"   ⏱️  Latency: {latency:.0f}ms")
    print(f"   🎯 NDCG: {ndcg:.3f} | Precision: {precision:.1f}% | Recall: {recall:.1f}%\n")

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

# Save JSON results
output = {
    "approach": "Approach 4: Two-Stage (Semantic + Cross-Encoder) with Gaussian Age",
    "configuration": {
        "stage1_semantic_weight": 0.8746,
        "stage1_age_weight": 0.1254,
        "stage2_ce_weight": 0.8746,
        "stage2_age_weight": 0.1254,
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

with open("approach_4_results.json", 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ JSON Results saved to: approach_4_results.json")

# Generate Detailed HTML Report (Same as Approach 1)
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Approach 4: Gaussian + Cross-Encoder - Detailed Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
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
            color: 2d3748; 
            text-align: center; 
            margin-bottom: 10px; 
            font-size: 2.5em;
            background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
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
            background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
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
            border-left: 4px solid #8b5cf6;
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
            border-bottom: 3px solid #8b5cf6;
        }}
        
        .config-box {{
            background: linear-gradient(135deg, #faf5ff 0%, #ede9fe 100%);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 40px;
            border-left: 4px solid #8b5cf6;
        }}
        
        .config-box h3 {{
            color: #5b21b6;
            margin-bottom: 15px;
        }}
        
        .config-item {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #ddd6fe;
        }}
        
        .config-label {{
            font-weight: 600;
            color: #6b21a8;
        }}
        
        .config-value {{
            color: #7c3aed;
            font-family: 'Courier New', monospace;
        }}
        
        .chart-container {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Approach 4 Report</h1>
        <div class="subtitle">Two-Stage: Gaussian Scoring → Cross-Encoder Re-ranking</div>
        
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
                <h3>Balanced Two-Stage Approach</h3>
                <div class="config-item">
                    <span class="config-label">Stage 1 - Gaussian Scoring:</span>
                    <span class="config-value">87.46% Semantic + 12.54% Age</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Stage 2 - Cross-Encoder:</span>
                    <span class="config-value">87.46% CE + 12.54% Age</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Older Age Decay:</span>
                    <span class="config-value">10.62% per year (Base: 0.8938)</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Younger Age Decay:</span>
                    <span class="config-value">9.09% per year (Base: 0.9091)</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Hard Filters:</span>
                    <span class="config-value">Gender + Religion Only (Soft Age/Location)</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Process Flow:</span>
                    <span class="config-value">Top 50 → Gaussian Top 20 → CE Top 10</span>
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
        
        <!-- Detailed Scenario Results -->
        <div class="section">
            <h2 class="section-title">🔍 Detailed Scenario Results</h2>
"""

# Add scenario-wise comparisons
for i, result in enumerate(results):
    scenario = ground_truth[i]
    gt_candidates = scenario['ground_truth_candidates']
    gt_ids = set([m['user_id'] for m in gt_candidates if m['relevance_score'] > 0])
    pred_ids = set([m['user_id'] for m in matches_data[i]])
    
    html_content += f"""
            <div style="background: #f9fafb; margin-bottom: 40px; border: 2px solid #e5e7eb; padding: 30px; border-radius: 12px;">
                <h3 style="color: #7c3aed; margin-bottom: 20px; font-size: 1.3em;">{scenario['scenario_name']}</h3>
                <div style="display: flex; gap: 15px; margin: 15px 0 25px 0; flex-wrap: wrap;">
                    <div style="background: #ddd6fe; padding: 12px 20px; border-radius: 8px; color: #5b21b6; font-weight: 600;">
                        Latency: {result['latency_ms']}ms
                    </div>
                    <div style="background: #ddd6fe; padding: 12px 20px; border-radius: 8px; color: #5b21b6; font-weight: 600;">
                        NDCG: {result['ndcg']}
                    </div>
                    <div style="background: #ddd6fe; padding: 12px 20px; border-radius: 8px; color: #5b21b6; font-weight: 600;">
                        Precision: {result['precision']}%
                    </div>
                    <div style="background: #ddd6fe; padding: 12px 20px; border-radius: 8px; color: #5b21b6; font-weight: 600;">
                        Recall: {result['recall']}%
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px;">
                    <div>
                        <h4 style="color: #374151; margin-bottom: 15px; font-size: 1.1em;">✅ Predicted Results (Top 10)</h4>
    """
    
    for p in matches_data[i]:
        is_match = p['user_id'] in gt_ids
        border_color = "#8b5cf6" if is_match else "#ef4444"
        bg_gradient = "linear-gradient(90deg, #faf5ff 0%, white 100%)" if is_match else "linear-gradient(90deg, #fef2f2 0%, white 100%)"
        
        html_content += f"""
                        <div style="border: 2px solid #e5e7eb; border-left: 5px solid {border_color}; background: {bg_gradient}; padding: 15px; margin-bottom: 12px; border-radius: 8px; transition: all 0.2s;">
                            <strong>{p['name']} (ID: {p['user_id']})</strong><br>
                            {p['job_title']}, {p['education']}<br>
                            {p['location']}, Age {p['age']}<br>
                            <div style="margin-top: 8px;">
                                <span style="display: inline-block; background: #e5e7eb; padding: 4px 10px; border-radius: 12px; font-size: 0.85em; margin-right: 8px;">
                                    Final: {p.get('final_combined_score', 0):.3f}
                                </span>
                                <span style="display: inline-block; background: #e5e7eb; padding: 4px 10px; border-radius: 12px; font-size: 0.85em; margin-right: 8px;">
                                    CE: {p.get('ce_score', 0):.3f}
                                </span>
                                <span style="display: inline-block; background: #e5e7eb; padding: 4px 10px; border-radius: 12px; font-size: 0.85em;">
                                    Age: {p.get('age_score', 0):.3f}
                                </span>
                            </div>
                        </div>
        """
    
    html_content += """
                    </div>
                    <div>
                        <h4 style="color: #374151; margin-bottom: 15px; font-size: 1.1em;">📋 Ground Truth (Expected Top Matches)</h4>
    """
    
    for gt in gt_candidates[:15]:
        is_found = gt['user_id'] in pred_ids
        status = "✓ Found" if is_found else "○ Missed"
        color = "color: #059669; font-weight: bold;" if is_found else "color: #9ca3af;"
        
        html_content += f"""
                        <div style="border: 2px solid #e5e7eb; padding: 15px; margin-bottom: 12px; border-radius: 8px; background: white;">
                            <div style="float: right; {color}">{status}</div>
                            <strong>{gt['name']} (ID: {gt['user_id']})</strong> - Relevance: {gt['relevance_score']}<br>
                            {gt['job_title']}, {gt['education']}<br>
                            {gt['location']}, Age {gt['age']}
                        </div>
        """
    
    html_content += """
                    </div>
                </div>
            </div>
    """

html_content += """
        </div>
        
        <!-- Key Insights -->
        <div class="insights-box">
            <h3>🔑 Key Insights</h3>
            <ul>
                <li><strong>Two-Stage Processing:</strong> Gaussian scoring narrows 50 candidates to 20, then cross-encoder refines to top 10 for efficiency.</li>
                <li><strong>Balanced Weights:</strong> 87.46% semantic + 12.54% age maintained in BOTH stages for consistency.</li>
                <li><strong>Soft Filters Only:</strong> Only Gender and Religion are hard filters - Age is soft (scored, not filtered). Location is semantic.</li>
                <li><strong>Age Preserved:</strong> Unlike old approach, age scoring is combined with CE in final stage, not dropped.</li>
                <li><strong>Symmetric Decay:</strong> Nearly equal decay rates (10.62% vs 9.09%) treat older/younger candidates fairly.</li>
            </ul>
        </div>
    </div>
    
    <script>
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
                        borderColor: 'rgba(139, 92, 246, 1)',
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

with open("approach_4_report.html", "w", encoding='utf-8') as f:
    f.write(html_content)

print("\n✅ Detailed HTML Report saved to: approach_4_report.html")

import webbrowser
webbrowser.open('approach_4_report.html')

cur.close()
conn.close()
