
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
import json

warnings.filterwarnings('ignore')

print("✅ Imports successful")

# ========================
# 1. CONFIGURATION & SETUP
# ========================

# Load environment variables
import os
env_path = Path('../../vivah_api/.env')
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env from: {env_path.absolute()}")
else:
    load_dotenv()
    print("⚠️  Using default .env")

# Get credentials
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "triple-name-473801-t7")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if CREDENTIALS_PATH and not os.path.isabs(CREDENTIALS_PATH):
    CREDENTIALS_PATH = str((Path('../../vivah_api') / CREDENTIALS_PATH).absolute())

# Initialize Client
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=SCOPES)
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION, credentials=credentials)

print("✅ Google GenAI client initialized!")

# Models to Test
CONFIGS_TO_TEST = [
    {'name': 'Text-Embed-004 (768)',  'id': 'text-embedding-004', 'dim': 768},
    {'name': 'Text-Embed-005 (768)',  'id': 'text-embedding-005', 'dim': 768},
    {'name': 'Gemini-001 (768)',   'id': 'gemini-embedding-001', 'dim': 768},
    {'name': 'Gemini-001 (1152)',  'id': 'gemini-embedding-001', 'dim': 1152},
    {'name': 'Gemini-001 (1536)',  'id': 'gemini-embedding-001', 'dim': 1536},
    {'name': 'Gemini-001 (3072)',  'id': 'gemini-embedding-001', 'dim': 3072},
]

# ========================
# 2. LOAD CORPUS FROM CSV
# ========================
csv_path = Path(__file__).parent / "matchmaking_fixed.csv"
print(f"\n📂 Loading Corpus from: {csv_path.name}")

try:
    df_corpus = pd.read_csv(csv_path)
    if 'User_ID' not in df_corpus.columns:
        df_corpus['User_ID'] = df_corpus.index + 1

    df_corpus['emb_text'] = (
        "Works as " + df_corpus['Job_Title'].fillna('') + ". " +
        "Has " + df_corpus['Education'].fillna('') + " degree. " +
        "Located in " + df_corpus['Location'].fillna('') + "."
    )
    
    df_corpus['display_text'] = (
        "<strong>#" + df_corpus['User_ID'].astype(str) + " " + df_corpus['Name'] + "</strong><br>" +
        "<small>" + df_corpus['Job_Title'] + " • " + df_corpus['Education'] + " • " + df_corpus['Location'] + "</small>"
    )
    
    # Store raw columns for analysis
    PROFILES_DATA = df_corpus[['Job_Title', 'Education', 'Location']].to_dict('records')
    CORPUS_TEXTS = df_corpus['emb_text'].tolist()
    DISPLAY_TEXTS = df_corpus['display_text'].tolist()
    IDS = df_corpus['User_ID'].tolist()
    
    print(f"✅ Loaded {len(CORPUS_TEXTS)} Profiles")
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# ========================
# 3. TEST SCENARIOS (Broad & Combinations)
# ========================

# ========================
# 3. DYNAMIC SCENARIO GENERATION (Templates & Common Indian Queries)
# ========================

def generate_query_variations(intent):
    """
    Generates varied queries based on intent (Formal, Template, Indian Common).
    """
    job = intent['job'][0] if intent['job'] else "Someone"
    loc = intent['loc'][0] if intent['loc'] else ""
    edu = intent['edu'][0] if intent['edu'] else ""
    
    variations = []
    
    # 1. Fixed Template (Strict)
    parts = [f"Looking for a {job}"]
    if edu: parts.append(f"who has done {edu}")
    if loc: parts.append(f"and lives in {loc}")
    variations.append(" ".join(parts) + ".")
    
    # 2. Common Indian/Simple Style
    if loc and job:
        variations.append(f"{job} in {loc} urgent requirement")
    elif job:
        variations.append(f"Need {job} profile")
        
    return variations[0] # For now, we pick the Template one as primary for consistent benchmarking

# Define Base Scenarios (Intent Only)
# Define Base Scenarios (Intent Only)
# Define Base Scenarios (Intent Only) with HARDCODED Verified Tiers
BASE_SCENARIOS = [
    {
        "intent_name": "SDE (Prof + Edu)",
        "intent": {"job": ["SDE", "Software Engineer"], "loc": [], "edu": ["B.Tech", "B.E."]},
        "gt_tiers": {'tier_1': [7, 60, 154], 'tier_2': [1, 19, 57, 71, 75]}
    },
    {
        "intent_name": "Radiologist (Prof + Loc)",
        "intent": {"job": ["Radiologist"], "loc": ["Ahmedabad"], "edu": []},
        "gt_tiers": {'tier_1': [9, 49], 'tier_2': [4, 5, 7, 8, 15]}
    },
    {
        "intent_name": "CA (Edu + Loc)",
        "intent": {"job": [], "loc": ["Delhi"], "edu": ["CA", "Chartered Accountant"]},
        "gt_tiers": {'tier_1': [11, 119], 'tier_2': [17, 21, 30, 35, 36]}
    },
    {
        "intent_name": "Business Analyst (All 3)",
        "intent": {"job": ["Business Analyst"], "loc": ["Noida"], "edu": ["MBA"]},
        "gt_tiers": {'tier_1': [6, 160], 'tier_2': [2, 8, 16, 22, 24]}
    }
]

# Build Final Scenarios with generated queries
FINAL_SCENARIOS = []
for base in BASE_SCENARIOS:
    # Generate query using the new Template Logic
    query_text = generate_query_variations(base['intent'])
    
    # Direct use of verified Ground Truth
    gt_tiers = base['gt_tiers']
    # Ensure all keys exist
    if 'tier_3' not in gt_tiers: gt_tiers['tier_3'] = []
    
    intent = base['intent']
    FINAL_SCENARIOS.append({
        "query": query_text,
        "ground_truth": gt_tiers,
        "intent": intent,
        "desc": base['intent_name']
    })

print(f"✅ Generated {len(FINAL_SCENARIOS)} Templated Scenarios")

# ========================
# 4. ANALYSIS & METRICS (Bias Calculation)
# ========================
TIER_WEIGHTS = {'tier_1': 3.0, 'tier_2': 2.0, 'tier_3': 1.0}

def get_tier_badge(uid, ground_truth):
    if uid in ground_truth['tier_1']: return "<span class='badge tier1'>👑 Tier 1</span>"
    if uid in ground_truth['tier_2']: return "<span class='badge tier2'>🥈 Tier 2</span>"
    return ""

def calculate_model_bias(model_results):
    """
    Aggregates match stats to determine model focus (Job vs Loc vs Edu).
    """
    stats = {'job_hits': 0, 'loc_hits': 0, 'edu_hits': 0, 'total': 0}
    
    for res in model_results:
        tags = res['tags'] # HTML string containing matched tags
        if "✅ Job" in tags: stats['job_hits'] += 1
        if "✅ Loc" in tags: stats['loc_hits'] += 1
        if "✅ Edu" in tags: stats['edu_hits'] += 1
        stats['total'] += 1
        
    # Convert to percentages
    total = max(stats['total'], 1)
    return {
        'Job Focus': (stats['job_hits'] / total) * 100,
        'Loc Focus': (stats['loc_hits'] / total) * 100,
        'Edu Focus': (stats['edu_hits'] / total) * 100
    }

def analyze_match_tags_list(profile_idx, intent):
    """Returns list of plain tags for calculation"""
    tags = []
    prof = PROFILES_DATA[profile_idx]
    if intent.get('job'):
        if any(kw.lower() in str(prof['Job_Title']).lower() for kw in intent['job']): tags.append("✅ Job")
    if intent.get('loc'):
        if any(kw.lower() in str(prof['Location']).lower() for kw in intent['loc']): tags.append("✅ Loc")
    if intent.get('edu'):
        if any(kw.lower() in str(prof['Education']).lower() for kw in intent['edu']): tags.append("✅ Edu")
    return tags

def analyze_match_html(tags_list, intent):
    """Converts tags list to HTML"""
    html_tags = []
    # Job
    if intent.get('job'):
        if "✅ Job" in tags_list: html_tags.append("<span class='tag match'>✅ Job</span>")
        else: html_tags.append("<span class='tag mismatch'>❌ Job</span>")
    # Loc
    if intent.get('loc'):
        if "✅ Loc" in tags_list: html_tags.append("<span class='tag match'>✅ Loc</span>")
        else: html_tags.append("<span class='tag mismatch'>❌ Loc</span>")
    # Edu
    if intent.get('edu'):
        if "✅ Edu" in tags_list: html_tags.append("<span class='tag match'>✅ Edu</span>")
        else: html_tags.append("<span class='tag mismatch'>❌ Edu</span>")
    return " ".join(html_tags)

def calculate_metrics(retrieved_ids, ground_truth_tiered, k=5):
    retrieved_k = retrieved_ids[:k]
    item_weights = {}
    for tier, items in ground_truth_tiered.items():
        weight = TIER_WEIGHTS.get(tier, 0.0)
        for item in items: item_weights[item] = weight     
    all_weights_sorted = sorted(item_weights.values(), reverse=True)
    dcg = sum((item_weights.get(item, 0.0) / np.log2(i + 2)) for i, item in enumerate(retrieved_k))
    idcg = sum((w / np.log2(i + 2)) for i, w in enumerate(all_weights_sorted[:k]))
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg

def generate_html_report(results, query_details, bias_summary, timestamp):
    df = pd.DataFrame(results)
    winner = df.sort_values("Avg nDCG", ascending=False).iloc[0] if not df.empty else {'Model': 'None', 'Dim':0, 'Avg nDCG':0}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Matrimony Search Benchmark</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #cbd5e1; padding: 2rem; }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        .card {{ background: #1e293b; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border: 1px solid #334155; }}
        h1 {{ background: linear-gradient(to right, #8b5cf6, #ec4899); -webkit-background-clip: text; color: transparent; margin-bottom: 0.5rem; }}
        table {{ width: 100%; border-collapse: separate; border-spacing: 0; margin-top: 1rem; }}
        th {{ background: #334155; padding: 1rem; text-align: left; font-size: 0.9em; color: #fff; border-radius: 8px 8px 0 0; }}
        td {{ padding: 0.5rem; background: #1e293b; border-bottom: 1px solid #334155; vertical-align: top; }}
        .tag {{ font-size: 0.7em; padding: 2px 6px; border-radius: 4px; margin-right: 4px; display: inline-block; margin-top: 4px; }}
        .match {{ background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }}
        .mismatch {{ background: rgba(248, 113, 113, 0.1); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }}
        .result-card {{ padding: 12px; border: 1px solid #334155; border-radius: 8px; margin-bottom: 8px; background: #0f172a; position: relative; }}
        .score-pill {{ position: absolute; top: 10px; right: 10px; font-size: 0.8em; font-family: monospace; background: #334155; padding: 2px 6px; border-radius: 4px; }}
        .badge {{ font-size: 0.75em; padding: 2px 6px; border-radius: 4px; font-weight: bold; margin-bottom: 4px; display: inline-block; }}
        .tier1 {{ background: linear-gradient(135deg, #fbbf24, #d97706); color: #fff; text-shadow: 0 1px 2px rgba(0,0,0,0.2); }}
        .tier2 {{ background: #94a3b8; color: #fff; }}
        .model-header {{ min-width: 220px; border-right: 1px solid #334155; }}
        .bias-bar {{ height: 6px; border-radius: 3px; background: #334155; margin-top: 4px; overflow: hidden; }}
        .bias-fill {{ height: 100%; background: #a855f7; }}
    </style>
</head>
<body>
    <div class="container">
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1>💒 Matrimony Search Benchmark</h1>
            <p>Evaluating Embedding Models on Dynamic User Queries</p>
        </div>
        
        <div class="card" style="border: 1px solid #8b5cf6; background: linear-gradient(to right, rgba(139, 92, 246, 0.1), rgba(236, 72, 153, 0.1));">
            <h2 style="color: #fff; margin-top: 0;">🏆 Champion: {winner['Model']} ({winner['Dim']}d)</h2>
            <div style="display: flex; gap: 2rem;">
                <div><span style="color: #94a3b8;">Average nDCG</span><br><strong style="font-size: 1.5em; color: #4ade80;">{winner['Avg nDCG']:.4f}</strong></div>
                <div><span style="color: #94a3b8;">Avg Search Time</span><br><strong style="font-size: 1.5em;">{winner['Search Time (ms)']:.1f}ms</strong></div>
            </div>
        </div>

        <div class="card">
            <h3>🧠 Model Behavior & Bias Analysis</h3>
            <p style="color:#94a3b8; font-size:0.9em;">Analysis of what each model prioritizes (Job, Location, Education) across all queries.</p>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Job Sensitivity</th>
                    <th>Loc Sensitivity</th>
                    <th>Edu Sensitivity</th>
                </tr>
    """
    
    for m in bias_summary:
        j = m['stats']['Job Focus']
        l = m['stats']['Loc Focus']
        e = m['stats']['Edu Focus']
        html += f"""
        <tr>
            <td><strong>{m['name']}</strong></td>
            <td>{j:.1f}%<div class="bias-bar"><div class="bias-fill" style="width:{j}%; background:#22c55e;"></div></div></td>
            <td>{l:.1f}%<div class="bias-bar"><div class="bias-fill" style="width:{l}%; background:#3b82f6;"></div></div></td>
            <td>{e:.1f}%<div class="bias-bar"><div class="bias-fill" style="width:{e}%; background:#f59e0b;"></div></div></td>
        </tr>
        """
        
    html += """
            </table>
        </div>

        <div class="card">
            <h3>📊 Performance Summary</h3>
            <table>
                <tr><th>Model</th><th>Dim</th><th>nDCG</th><th>Latency</th></tr>
    """
    if not df.empty:
        for _, row in df.sort_values("Avg nDCG", ascending=False).iterrows():
            html += f"<tr><td><strong>{row['Model']}</strong></td><td>{row['Dim']}</td><td>{row['Avg nDCG']:.4f}</td><td>{row['Embed Time (ms)']:.1f}ms / {row['Search Time (ms)']:.1f}ms</td></tr>"
    
    html += "</table></div>"
    
    for q, det in query_details.items():
        desc = det.get('desc', 'Query')
        
        # Find Target Matches for this query's scenario
        target_names = []
        for scen in FINAL_SCENARIOS:
            if scen['query'] == q:
                # We look at the Ground Truth (Tier 1) IDs and fetch their Names/Jobs
                tier1_ids = scen['ground_truth'].get('tier_1', [])
                for uid in tier1_ids:
                    # Find details in PROFILES_DATA (we need to match UID to index)
                    # UID is 1-based, list is 0-based index if sequential. 
                    # Let's map UID -> Profile Data safer
                    try:
                        idx = IDS.index(uid)
                        prof = PROFILES_DATA[idx]
                        name = df_corpus.iloc[idx]['Name']
                        target_names.append(f"<strong>{name}</strong> <small>({prof['Job_Title']}, {prof['Location']})</small>")
                    except: continue
                break
        
        if target_names:
            target_html = "<div style='display:flex; flex-wrap:wrap; gap:10px;'>" + \
                          "".join([f"<div style='background:rgba(139, 92, 246, 0.2); border:1px solid #8b5cf6; padding:4px 8px; border-radius:6px; color:#ddd;'>{t}</div>" for t in target_names]) + \
                          "</div>"
        else:
            target_html = "<span style='color:#64748b'>No specific Tier 1 targets defined</span>"

        html += f"""
        <div class="card">
            <h3 style="margin-bottom:0.5rem;">🔍 <span style="font-weight: normal; color: #94a3b8;">{desc}:</span> <span style="color:#fff">"{q}"</span></h3>
            
            <div style="margin-bottom: 1.5rem; background: #2e1065; padding: 1rem; border-radius: 8px; border-left: 5px solid #a855f7;">
                <h4 style="margin-top:0; color:#a855f7; margin-bottom:0.5rem;">🎯 Expected Ideal Matches (Tier 1):</h4>
                {target_html}
            </div>

            <div style="overflow-x: auto;">
            <table><thead><tr>
        """
        for m in det['models']:
            html += f"<th class='model-header'>{m['model_name']} ({m['dim']}d)<br><span style='font-weight:normal; opacity:0.8'>nDCG: {m['ndcg']:.3f}</span></th>"
        html += "</tr></thead><tbody><tr>"
        for m in det['models']:
            html += "<td style='padding:0.5rem; border-right: 1px solid #334155;'>"
            for res in m['top_5']:
                tier_badge = res['tier_badge']
                html += f"""
                <div class="result-card">
                    {tier_badge}<br>
                    {res['display_html']}
                    <div style="margin-top:6px; border-top: 1px dashed #334155; padding-top: 4px;">{res['tags_html']}</div>
                    <span class="score-pill">Sim: {res['score']:.3f}</span>
                </div>
                """
            html += "</td>"
        html += "</tr></tbody></table></div></div>"
    html += "</div></body></html>"
    return html

# ========================
# 6. RUN BENCHMARK
# ========================
def run_benchmark():
    print(f"\n🚀 STARTING MIXED BENCHMARK (With Dynamic Templates & Bias Analysis)")
    print("=" * 60)
    
    final_results = []
    query_details = {} 
    bias_summary = []

    for config in CONFIGS_TO_TEST:
        model_id = config['id']
        dim = config['dim']
        name = config['name']
        
        print(f"\n🔵 Processing: {name} (Dim: {dim})...", end=" ")
        
        try:
            t_start = time.time()
            corpus_vecs = []
            chunk_size = 50
            for i in range(0, len(CORPUS_TEXTS), chunk_size):
                chunk = CORPUS_TEXTS[i:i+chunk_size]
                resp = client.models.embed_content(
                    model=model_id, contents=chunk,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=dim)
                )
                batch_vecs = [e.values for e in resp.embeddings]
                corpus_vecs.extend(batch_vecs)
            
            avg_embed_time = ((time.time() - t_start) * 1000) / len(CORPUS_TEXTS)
            corpus_vecs = np.array(corpus_vecs)
            print(f"Indexed.", end=" ")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        search_times = []
        ndcg_scores = []
        model_result_stats = [] # Store all top results for bias calc
        
        for scenario in FINAL_SCENARIOS:
            query = scenario['query']
            gt = scenario['ground_truth']
            intent = scenario['intent']
            desc = scenario.get('desc', 'Query')
            
            if query not in query_details: query_details[query] = {'models': [], 'desc': desc}
            
            s_start = time.time()
            q_resp = client.models.embed_content(
                model=model_id, contents=query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=dim)
            )
            q_vec = np.array(q_resp.embeddings[0].values).reshape(1, -1)
            
            scores = cosine_similarity(q_vec, corpus_vecs)[0]
            top_idx = np.argsort(scores)[::-1]
            search_times.append((time.time() - s_start) * 1000)
            
            retrieved_ids = [IDS[i] for i in top_idx]
            ndcg = calculate_metrics(retrieved_ids, gt, k=5)
            ndcg_scores.append(ndcg)
            
            top_5_details = []
            for idx in top_idx[:5]:
                tags_list = analyze_match_tags_list(idx, intent)
                tags_html = analyze_match_html(tags_list, intent) 
                
                uid = IDS[idx]
                tier_badge = get_tier_badge(uid, gt)
                score = scores[idx]
                
                res_obj = {
                    'display_html': DISPLAY_TEXTS[idx],
                    'tags_html': tags_html,
                    'tags': tags_list, # For bias calc
                    'tier_badge': tier_badge,
                    'score': score
                }
                top_5_details.append(res_obj)
                model_result_stats.append(res_obj)
            
            query_details[query]['models'].append({
                'model_name': name, 'dim': dim, 'ndcg': ndcg, 'top_5': top_5_details
            })
        
        # Calculate Bias for this model
        bias_stats = calculate_model_bias(model_result_stats)
        bias_summary.append({'name': f"{name} ({dim}d)", 'stats': bias_stats})
        
        print(f"| Avg Search Latency: {np.mean(search_times):.1f}ms")
        
        final_results.append({
            'Model': name, 'Dim': dim, 'Avg nDCG': np.mean(ndcg_scores),
            'Embed Time (ms)': avg_embed_time, 'Search Time (ms)': np.mean(search_times),
        })

    html = generate_html_report(final_results, query_details, bias_summary, datetime.now())
    html_path = Path(__file__).parent / "mix_benchmark_report.html"
    with open(html_path, 'w', encoding="utf-8") as f: f.write(html)
    
    print(f"\n✅ Reports Saved: {html_path.name}")
    try: webbrowser.open(f'file:///{html_path.absolute()}')
    except: pass

if __name__ == "__main__":
    run_benchmark()
