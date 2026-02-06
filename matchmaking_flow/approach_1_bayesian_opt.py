"""
APPROACH 1 Experiment: Bayesian Optimization for Hyperparameter Tuning
--------------------------------------------------------------------
Goal: optimize the weights and age-decay parameters to maximize accuracy against Ground Truth.
Based on Approach 1 logic (Soft Filters + Gaussian Age).

Hyperparameters to Tune:
a - Semantic Weight (0-100) -> Normalized to 0-1
b - Age Weight (Derived as 100-a) -> Normalized to 0-1
c - Older Age Decay % (0-50) -> Base = 1 - (c/100)
d - Younger Age Decay % (0-50) -> Base = 1 - (d/100)

Uses: bayesian-optimization library
"""

import psycopg2
import json
import time
import pandas as pd
from bayes_opt import BayesianOptimization
from google import genai
from google.genai import types
from google.oauth2 import service_account
import os
from pathlib import Path
from dotenv import load_dotenv

# ========================
# 1. SETUP
# ========================
import sys
import io

# Force UTF-8 for stdout/stderr to handle emojis on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load environment
env_path = Path('../../vivah_api/.env')
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# Google GenAI Setup
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if CREDENTIALS_PATH and not os.path.isabs(CREDENTIALS_PATH):
    CREDENTIALS_PATH = str((Path('../../vivah_api') / CREDENTIALS_PATH).absolute())
    
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "triple-name-473801-t7")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    credentials=service_account.Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=["https://www.googleapis.com/auth/cloud-platform"])
)

# DB Connection
conn = psycopg2.connect(
    host="localhost",
    port="5433",
    database="postgres",
    user="postgres",
    password="matchpass"
)
cur = conn.cursor()

# Load Ground Truth
with open('ground_truth_refined.json', 'r', encoding='utf-8') as f:
    ground_truth = json.load(f)

# NDCG Helper
import math
def calculate_dcg(relevances):
    dcg = 0
    for i, rel in enumerate(relevances):
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


print("="*80)
print("🧪 HYPERPARAMETER OPTIMIZATION (Bayesian)")
print("="*80)

# ========================
# 2. CACHING CANDIDATES
# ========================
# To make optimization fast, we fetch top 50 semantic matches ONCE for each scenario.
# The optimization loop will only re-rank these 50 candidates.

def create_embedding(text: str) -> list:
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=str(text),
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=1152)
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"Embedding Error: {e}")
        return None

CANDIDATE_CACHE = {}

print("\n[INFO] Pre-fetching candidates for all scenarios (this avoids repeated DB calls)...")

for scenario in ground_truth:
    sid = scenario['scenario_id']
    query = scenario['query']
    filters = scenario['filters']
    
    print(f"   Processing Scenario {sid}: {scenario['scenario_name']}...")
    
    # 1. Embedding
    query_vec = create_embedding(query)
    vec_str = '[' + ','.join(map(str, query_vec)) + ']'
    
    # 2. DB Fetch (Fetch top 100 to allow age-boosting to surface hidden gems)
    # Note: We apply the Hard Filters from Approach 1 (Gender + Religion)
    sql = f"""
        SELECT 
            user_id, age, location, education, job_title,
            1 - (user_vector <=> '{vec_str}'::vector) as similarity
        FROM users
        WHERE gender = '{filters['gender']}'
          AND religion = '{filters['religion']}'
        ORDER BY user_vector <=> '{vec_str}'::vector
        LIMIT 100;
    """
    cur.execute(sql)
    candidates = cur.fetchall()
    
    # Store processed format
    candidate_list = []
    for c in candidates:
        candidate_list.append({
            "user_id": c[0],
            "age": c[1],
            "similarity": float(c[5])
        })
    
    target_age = scenario['filters'].get('preferred_age', scenario['searcher']['age'])
    
    # Valid candidates are those with relevance > 0 in refined GT
    gt_candidates = scenario['ground_truth_candidates']
    gt_map = {c['user_id']: c['relevance_score'] for c in gt_candidates if c['relevance_score'] > 0}
    
    CANDIDATE_CACHE[sid] = {
        "target_age": target_age,
        "ground_truth_map": gt_map,
        "ground_truth_candidates": gt_candidates,
        "candidates": candidate_list
    }

print("\n[SUCCESS] Caching Complete. Starting Optimization loop...")

# ========================
# 3. OPTIMIZATION FUNCTION
# ========================

def calculate_age_score(target_age, candidate_age, older_decay_pct, younger_decay_pct):
    diff = candidate_age - target_age
    if diff == 0: return 1.0
    
    if diff > 0: # Older
        base = 1.0 - (older_decay_pct / 100.0)
        # Prevent negative base
        base = max(0.01, base)
        return 1.0 * (base ** diff)
    else: # Younger
        base = 1.0 - (younger_decay_pct / 100.0)
        base = max(0.01, base)
        return 1.0 * (base ** abs(diff))

def evaluate_parameters(semantic_weight_a, older_decay_c, younger_decay_d):
    """
    Objective Function for Bayesian Optimization.
    Returns: Average Accuracy across all scenarios.
    
    Constraints:
    semantic_weight_a: 0-100
    age_weight_b: 100 - semantic_weight_a
    """
    
    total_accuracy = 0
    
    # Normalized weights
    w_sem = semantic_weight_a / 100.0
    w_age = (100.0 - semantic_weight_a) / 100.0
    
    for sid, data in CANDIDATE_CACHE.items():
        candidates = data['candidates']
        target_age = data['target_age']
        # gt_ids was removed, we use ground_truth_map now
        
        # Scoring
        scored_candidates = []
        for cand in candidates:
            # Semantic Score
            s_score = cand['similarity']
            
            # Age Score
            a_score = calculate_age_score(target_age, cand['age'], older_decay_c, younger_decay_d)
            
            # Final Score
            final_score = (s_score * w_sem) + (a_score * w_age)
            
            scored_candidates.append((cand['user_id'], final_score))
        
        # Sort desc
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Top 10
        top_10 = [x[0] for x in scored_candidates[:10]]
        
        # Calculate NDCG
        relevance_map = data['ground_truth_map']
        
        # Get relevances for predicted list (Top 10)
        relevances = [relevance_map.get(pid, 0) for pid in top_10]
        
        # Calculate DCG
        actual_dcg = calculate_dcg(relevances)
        
        # Calculate Ideal DCG (IDCG) - sort known relevant items descending
        # We need to take Top K ideal scores
        ideal_relevances = sorted(list(relevance_map.values()), reverse=True)[:10]
        ideal_dcg = calculate_dcg(ideal_relevances)
        
        if ideal_dcg == 0: 
            ndcg = 0.0 
        else:
            ndcg = actual_dcg / ideal_dcg
            
        total_accuracy += ndcg # We are maximizing Sum/Avg of NDCG
    
    return total_accuracy / len(CANDIDATE_CACHE)

# ========================
# 4. RUN BAYESIAN OPTIMIZATION
# ========================

# Bounded region of parameter space
pbounds = {
    'semantic_weight_a': (50, 95),      # Assuming semantic is still very important
    'older_decay_c': (0, 50),           # % decay per year
    'younger_decay_d': (0, 50)          # % decay per year
}

optimizer = BayesianOptimization(
    f=evaluate_parameters,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

print("\n[INFO] Starting Optimization (15 Iterations)...")
start_time = time.time()

optimizer.maximize(
    init_points=5,
    n_iter=15,
)

# ========================
# 5. REPORTING
# ========================
print("\n" + "="*80)
print("OPTIMIZATION RESULTS")
print("="*80)

best = optimizer.max
best_params = best['params']
best_accuracy = best['target']

a = best_params['semantic_weight_a']
b = 100 - a
c = best_params['older_decay_c']
d = best_params['younger_decay_d']

print(f"Best Accuracy: {best_accuracy:.2f}%")
print("\nOptimal Parameters:")
print(f"   a (Semantic Weight):  {a:.2f}")
print(f"   b (Age Weight):       {b:.2f}")
print(f"   c (Older Decay %):    {c:.2f}% (Base: {1-(c/100):.3f})")
print(f"   d (Younger Decay %):  {d:.2f}% (Base: {1-(d/100):.3f})")

print("\nComparison with Baseline (Approach 1):")
print(f"   Baseline Params: a=80, b=20, c=20, d=40")
# Run baseline to compare
baseline_acc = evaluate_parameters(80, 20, 40)
print(f"   Baseline Accuracy: {baseline_acc:.2f}%")

print("\nOptimization took: {:.2f} seconds".format(time.time() - start_time))

# Save Experiment Results
experiment_results = {
    "best_accuracy": best_accuracy,
    "best_params": {
        "semantic_weight": a,
        "age_weight": b,
        "older_decay_pct": c,
        "younger_decay_pct": d
    },
    "all_iterations": []
}

for i, res in enumerate(optimizer.res):
    experiment_results["all_iterations"].append({
        "iteration": i+1,
        "params": res['params'],
        "accuracy": res['target']
    })

with open('bayes_opt_results.json', 'w') as f:
    json.dump(experiment_results, f, indent=4)
print("\n[SUCCESS] Results saved to 'bayes_opt_results.json'")

# Generate HTML Report
print("\nGenerating Detailed HTML Report...")

# Prepare data for charts
iteration_numbers = [i+1 for i in range(len(optimizer.res))]
iteration_scores = [res['target'] for res in optimizer.res]
semantic_weights = [res['params']['semantic_weight_a'] for res in optimizer.res]
older_decays = [res['params']['older_decay_c'] for res in optimizer.res]
younger_decays = [res['params']['younger_decay_d'] for res in optimizer.res]

# Sort iterations by accuracy
sorted_iterations = sorted(optimizer.res, key=lambda x: x['target'], reverse=True)

# Calculate statistics
avg_accuracy = sum(iteration_scores) / len(iteration_scores)
best_iter_idx = iteration_scores.index(max(iteration_scores)) + 1
worst_iter_idx = iteration_scores.index(min(iteration_scores)) + 1
improvement = max(iteration_scores) - min(iteration_scores)

html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Bayesian Optimization - Detailed Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
            border-left: 4px solid #667eea;
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
        
        .stat-card .subvalue {{
            display: block;
            font-size: 0.9em;
            color: #a0aec0;
            margin-top: 5px;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #2d3748;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .params-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 30px;
            margin-bottom: 40px;
        }}
        
        .param-card {{
            background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
            padding: 30px;
            border-radius: 12px;
            border: 2px solid #e2e8f0;
        }}
        
        .param-card h3 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        
        .param-card .param-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2d3748;
            margin-bottom: 10px;
        }}
        
        .param-card .param-desc {{
            color: #718096;
            font-size: 0.95em;
            line-height: 1.6;
        }}
        
        .param-card .param-formula {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #4a5568;
        }}
        
        .chart-container {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        
        .chart-container h3 {{
            color: #2d3748;
            margin-bottom: 20px;
            font-size: 1.3em;
        }}
        
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        .comparison-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        .comparison-table td {{
            padding: 15px;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        .comparison-table tr:hover {{
            background-color: #f7fafc;
        }}
        
        .comparison-table .winner {{
            background-color: #f0fff4;
            border-left: 4px solid #48bb78;
        }}
        
        .iteration-list {{
            list-style: none;
            padding: 0;
        }}
        
        .iteration-item {{
            background: white;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 8px;
            border-left: 4px solid #cbd5e0;
            display: grid;
            grid-template-columns: 60px 1fr 120px;
            gap: 20px;
            align-items: center;
            transition: all 0.2s;
        }}
        
        .iteration-item:hover {{
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        .iteration-item.top-3 {{
            background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 50%);
            border-left-color: #48bb78;
        }}
        
        .iteration-rank {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .iteration-params {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        
        .iteration-params .param-line {{
            font-size: 0.9em;
            color: #4a5568;
        }}
        
        .iteration-score {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2d3748;
            text-align: right;
        }}
        
        .insights-box {{
            background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
            border-left: 4px solid #fc8181;
            padding: 25px;
            border-radius: 8px;
            margin-top: 30px;
        }}
        
        .insights-box h3 {{
            color: #c53030;
            margin-bottom: 15px;
        }}
        
        .insights-box ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .insights-box li {{
            padding: 10px 0;
            color: #742a2a;
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
            background-color: #c6f6d5;
            color: #22543d;
        }}
        
        .badge-info {{
            background-color: #bee3f8;
            color: #2c5282;
        }}
        
        .badge-warning {{
            background-color: #feebc8;
            color: #744210;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Bayesian Optimization Report</h1>
        <div class="subtitle">Hyperparameter Tuning for Matchmaking Algorithm</div>
        
        <!-- Hero Section -->
        <div class="hero-section">
            <h2>{best_accuracy:.4f}</h2>
            <p>Best NDCG Score Achieved (out of 1.0)</p>
            <p style="font-size: 0.9em; margin-top: 10px; opacity: 0.8;">
                Optimization completed in {time.time() - start_time:.1f} seconds | {len(optimizer.res)} iterations
            </p>
        </div>
        
        <!-- Statistics Grid -->
        <div class="stats-grid">
            <div class="stat-card">
                <span class="label">Average NDCG</span>
                <span class="value">{avg_accuracy:.4f}</span>
                <span class="subvalue">Across all iterations</span>
            </div>
            <div class="stat-card">
                <span class="label">Best Iteration</span>
                <span class="value">#{best_iter_idx}</span>
                <span class="subvalue">Peak performance</span>
            </div>
            <div class="stat-card">
                <span class="label">Improvement</span>
                <span class="value">{improvement:.4f}</span>
                <span class="subvalue">From worst to best</span>
            </div>
            <div class="stat-card">
                <span class="label">vs Baseline</span>
                <span class="value">+{(best_accuracy - baseline_acc):.4f}</span>
                <span class="subvalue">Better than default</span>
            </div>
        </div>
        
        <!-- Optimal Parameters Section -->
        <div class="section">
            <h2 class="section-title">🏆 Optimal Parameters</h2>
            <div class="params-grid">
                <div class="param-card">
                    <h3>Semantic Weight (a)</h3>
                    <div class="param-value">{a:.2f}%</div>
                    <div class="param-desc">
                        Weight given to semantic similarity (job, education, location match). 
                        Higher value means profile content matching is more important than age.
                    </div>
                    <div class="param-formula">
                        Semantic Component = similarity × {a/100:.3f}
                    </div>
                </div>
                
                <div class="param-card">
                    <h3>Age Weight (b)</h3>
                    <div class="param-value">{b:.2f}%</div>
                    <div class="param-desc">
                        Weight given to age proximity scoring. Automatically calculated as (100 - a).
                        Lower value indicates age difference has less impact on final score.
                    </div>
                    <div class="param-formula">
                        Age Component = age_score × {b/100:.3f}
                    </div>
                </div>
                
                <div class="param-card">
                    <h3>Older Age Decay (c)</h3>
                    <div class="param-value">{c:.2f}%</div>
                    <div class="param-desc">
                        Penalty per year for candidates older than preferred age. 
                        Formula uses exponential decay: base^years_difference
                    </div>
                    <div class="param-formula">
                        Decay Base = {1-(c/100):.3f}<br>
                        Example: 3 years older → {(1-(c/100))**3:.3f} multiplier
                    </div>
                </div>
                
                <div class="param-card">
                    <h3>Younger Age Decay (d)</h3>
                    <div class="param-value">{d:.2f}%</div>
                    <div class="param-desc">
                        Penalty per year for candidates younger than preferred age.
                        Typically has stronger penalty as younger partners are less preferred.
                    </div>
                    <div class="param-formula">
                        Decay Base = {1-(d/100):.3f}<br>
                        Example: 3 years younger → {(1-(d/100))**3:.3f} multiplier
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Charts Section -->
        <div class="section">
            <h2 class="section-title">📊 Optimization Progress</h2>
            
            <div class="chart-container">
                <h3>NDCG Score Evolution</h3>
                <canvas id="accuracyChart" height="100"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>Parameter Exploration</h3>
                <canvas id="parametersChart" height="100"></canvas>
            </div>
        </div>
        
        <!-- Comparison Table -->
        <div class="section">
            <h2 class="section-title">⚖️ Baseline Comparison</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Configuration</th>
                        <th>Semantic Weight</th>
                        <th>Age Weight</th>
                        <th>Older Decay</th>
                        <th>Younger Decay</th>
                        <th>NDCG Score</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="winner">
                        <td><strong>Optimized (Bayesian)</strong> <span class="badge badge-success">WINNER</span></td>
                        <td>{a:.2f}%</td>
                        <td>{b:.2f}%</td>
                        <td>{c:.2f}%</td>
                        <td>{d:.2f}%</td>
                        <td><strong>{best_accuracy:.4f}</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Baseline (Default)</strong></td>
                        <td>80.00%</td>
                        <td>20.00%</td>
                        <td>20.00%</td>
                        <td>40.00%</td>
                        <td>{baseline_acc:.4f}</td>
                    </tr>
                    <tr style="background-color: #f0fff4;">
                        <td><strong>Improvement</strong></td>
                        <td>{a-80:+.2f}%</td>
                        <td>{b-20:+.2f}%</td>
                        <td>{c-20:+.2f}%</td>
                        <td>{d-40:+.2f}%</td>
                        <td><strong style="color: #48bb78;">{best_accuracy - baseline_acc:+.4f}</strong></td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <!-- All Iterations -->
        <div class="section">
            <h2 class="section-title">🔬 All Iterations (Ranked)</h2>
            <ul class="iteration-list">
"""

for i, res in enumerate(sorted_iterations, 1):
    p = res['params']
    css_class = "top-3" if i <= 3 else ""
    badge = f'<span class="badge badge-success">TOP {i}</span>' if i <= 3 else ""
    
    html_content += f"""
                <li class="iteration-item {css_class}">
                    <div class="iteration-rank">#{i}</div>
                    <div class="iteration-params">
                        <div class="param-line"><strong>Semantic:</strong> {p['semantic_weight_a']:.2f}% | <strong>Age:</strong> {100-p['semantic_weight_a']:.2f}%</div>
                        <div class="param-line"><strong>Older Decay:</strong> {p['older_decay_c']:.2f}% | <strong>Younger Decay:</strong> {p['younger_decay_d']:.2f}%</div>
                    </div>
                    <div class="iteration-score">{res['target']:.4f} {badge}</div>
                </li>
    """

html_content += f"""
            </ul>
        </div>
        
        <!-- Key Insights -->
        <div class="insights-box">
            <h3>🔑 Key Insights</h3>
            <ul>
                <li><strong>Semantic Dominance:</strong> Optimal semantic weight is {a:.1f}%, meaning job/education/location match is MUCH more important than age difference.</li>
                <li><strong>Age Flexibility:</strong> Age weight is only {b:.1f}%, indicating users are more flexible about age if other criteria match well.</li>
                <li><strong>Asymmetric Age Preference:</strong> Younger candidates face {d:.1f}% decay vs {c:.1f}% for older ones, showing {'stronger' if d > c else 'weaker'} penalty for being younger.</li>
                <li><strong>Performance Gain:</strong> Bayesian optimization achieved {(best_accuracy - baseline_acc)*100:.2f}% improvement over baseline configuration.</li>
                <li><strong>Convergence:</strong> Best result found at iteration #{best_iter_idx} out of {len(optimizer.res)}, showing efficient optimization.</li>
            </ul>
        </div>
    </div>
    
    <script>
        // Accuracy Chart
        const ctxAccuracy = document.getElementById('accuracyChart').getContext('2d');
        new Chart(ctxAccuracy, {{
            type: 'line',
            data: {{
                labels: {iteration_numbers},
                datasets: [{{
                    label: 'NDCG Score',
                    data: {iteration_scores},
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }},
                {{
                    label: 'Baseline',
                    data: Array({len(optimizer.res)}).fill({baseline_acc}),
                    borderColor: '#fc8181',
                    borderDash: [5, 5],
                    borderWidth: 2,
                    pointRadius: 0
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: true, position: 'top' }},
                    title: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: false,
                        min: {min(iteration_scores) - 0.05},
                        max: {max(iteration_scores) + 0.02},
                        title: {{ display: true, text: 'NDCG Score' }}
                    }},
                    x: {{
                        title: {{ display: true, text: 'Iteration' }}
                    }}
                }}
            }}
        }});
        
        // Parameters Chart
        const ctxParams = document.getElementById('parametersChart').getContext('2d');
        new Chart(ctxParams, {{
            type: 'scatter',
            data: {{
                datasets: [
                    {{
                        label: 'Semantic Weight',
                        data: {[{'x': i+1, 'y': semantic_weights[i], 'r': 8} for i in range(len(semantic_weights))]},
                        backgroundColor: 'rgba(102, 126, 234, 0.6)'
                    }},
                    {{
                        label: 'Older Decay',
                        data: {[{'x': i+1, 'y': older_decays[i], 'r': 8} for i in range(len(older_decays))]},
                        backgroundColor: 'rgba(72, 187, 120, 0.6)'
                    }},
                    {{
                        label: 'Younger Decay',
                        data: {[{'x': i+1, 'y': younger_decays[i], 'r': 8} for i in range(len(younger_decays))]},
                        backgroundColor: 'rgba(237, 137, 54, 0.6)'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: true, position: 'top' }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        title: {{ display: true, text: 'Parameter Value (%)' }}
                    }},
                    x: {{
                        title: {{ display: true, text: 'Iteration' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

with open('approach_1_bayes_report.html', 'w', encoding='utf-8') as f:
    f.write(html_content)
    
print("[SUCCESS] Report saved to 'approach_1_bayes_report.html'")
import webbrowser
webbrowser.open('approach_1_bayes_report.html')

cur.close()
conn.close()
