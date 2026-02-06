
import time
import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.oauth2 import service_account
from sklearn.metrics.pairwise import cosine_similarity
import webbrowser
from datetime import datetime
import concurrent.futures

warnings.filterwarnings('ignore')

print("✅ Imports successful")

# ========================
# 1. SETUP
# ========================
env_path = Path('../../vivah_api/.env')
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "triple-name-473801-t7")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if CREDENTIALS_PATH and not os.path.isabs(CREDENTIALS_PATH):
    CREDENTIALS_PATH = str((Path('../../vivah_api') / CREDENTIALS_PATH).absolute())

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=SCOPES)
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION, credentials=credentials)

print("✅ Client Initialized")

# MODEL TO USE (Baseline)
MODEL_ID = 'gemini-embedding-001'
DIM = 768

# ========================
# 2. DATA LOADING
# ========================
csv_path = Path(__file__).parent / "matchmaking_fixed.csv"
try:
    df_corpus = pd.read_csv(csv_path)
    if 'User_ID' not in df_corpus.columns:
        df_corpus['User_ID'] = df_corpus.index + 1

    # SINGLE VECTOR TEXT
    df_corpus['single_text'] = (
        "Works as " + df_corpus['Job_Title'].fillna('') + ". " +
        "Has " + df_corpus['Education'].fillna('') + " degree. " +
        "Located in " + df_corpus['Location'].fillna('') + "."
    )

    # RAW FIELDS FOR MULTI VECTOR
    CORPUS_JOBS = df_corpus['Job_Title'].fillna('Unknown').tolist()
    CORPUS_EDUS = df_corpus['Education'].fillna('Unknown').tolist()
    CORPUS_LOCS = df_corpus['Location'].fillna('Unknown').tolist()
    
    # METADATA
    IDS = df_corpus['User_ID'].tolist()
    DISPLAY_TEXTS = (
        "<strong>#" + df_corpus['User_ID'].astype(str) + " " + df_corpus['Name'] + "</strong><br>" +
        "<small>" + df_corpus['Job_Title'] + " • " + df_corpus['Education'] + " • " + df_corpus['Location'] + "</small>"
    ).tolist()
    
    PROFILES_DATA = df_corpus[['Job_Title', 'Education', 'Location']].to_dict('records')

    print(f"✅ Loaded {len(IDS)} Profiles")

except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# ========================
# 3. HELPER FUNCTIONS
# ========================
def get_embeddings(texts, task_type="RETRIEVAL_DOCUMENT", dim=768):
    vecs = []
    chunk_size = 50
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        # Handle empty strings? - API might error on empty, replace with " "
        chunk = [t if t.strip() else " " for t in chunk]
        
        resp = client.models.embed_content(
            model=MODEL_ID, contents=chunk,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=dim)
        )
        batch_vecs = [e.values for e in resp.embeddings]
        vecs.extend(batch_vecs)
    return np.array(vecs)

def generate_query_variations(intent):
    """
    Generates varied queries based on intent (Formal, Template, Indian Common).
    Matches logic in mix_test.py for consistency.
    """
    # intent uses lists, mix_test logic expects list access
    job = intent['job'][0] if intent.get('job') else "Someone"
    loc = intent['loc'][0] if intent.get('loc') else ""
    edu = intent['edu'][0] if intent.get('edu') else ""
    
    variations = []
    
    # 1. Fixed Template (Strict)
    parts = [f"Looking for a {job}"]
    if edu: parts.append(f"who has done {edu}")
    if loc: parts.append(f"and lives in {loc}")
    variations.append(" ".join(parts) + ".")
    
    # 2. Common Indian/Simple Style (Unused but kept for parity)
    if loc and job:
        variations.append(f"{job} in {loc} urgent requirement")
    elif job:
        variations.append(f"Need {job} profile")
        
    return variations[0]

# Weights
W_JOB = 0.6
W_EDU = 0.3
W_LOC = 0.1

# ========================
# 4. SCENARIOS (WEIGHTAGE TEST)
# ========================
FULL_SCENARIOS = [
    {
        "name": "SDE (Prof + Edu)",
        "single_query": "", # Dynamically generated
        "multi_pref": {"job": ["SDE", "Software Engineer"], "edu": ["B.Tech", "B.E."], "loc": []},
        "ground_truth": {'tier_1': [7, 60, 154], 'tier_2': [1, 19, 57, 71, 75]}
    },
    {
        "name": "Radiologist (Prof + Loc)",
        "single_query": "",
        "multi_pref": {"job": ["Radiologist"], "edu": [], "loc": ["Ahmedabad"]},
        "ground_truth": {'tier_1': [9, 49], 'tier_2': [4, 5, 7, 8, 15]}
    },
    {
        "name": "CA (Edu + Loc)",
        "single_query": "",
        "multi_pref": {"job": [], "edu": ["CA", "Chartered Accountant"], "loc": ["Delhi"]},
        "ground_truth": {'tier_1': [11, 119], 'tier_2': [17, 21, 30, 35, 36]}
    },
    {
        "name": "Business Analyst (All 3)",
        "single_query": "",
        "multi_pref": {"job": ["Business Analyst"], "edu": ["MBA"], "loc": ["Noida"]},
        "ground_truth": {'tier_1': [6, 160], 'tier_2': [2, 8, 16, 22, 24]}
    }
]

print(f"✅ Generated {len(FULL_SCENARIOS)} Consistent Scenarios")

# ========================
# 4. EXECUTION
# ========================

def run_experiment():
    results = []
    
    # Run Benchmark for each Dimension
    DIMS_TO_TEST = [768, 1152, 1536]
    
    for current_dim in DIMS_TO_TEST:
        print(f"\n==========================================")
        print(f"� Running Benchmark for DIMENSION: {current_dim}")
        print(f"==========================================")
        
        print("\n�📦 [Indexing Phase]")
        
        # --- METHOD A: SINGLE VECTOR ---
        t0 = time.time()
        # Pass current_dim to get_embeddings
        vecs_single = get_embeddings(df_corpus['single_text'].tolist(), "RETRIEVAL_DOCUMENT", current_dim)
        idx_time_single = (time.time() - t0) * 1000
        print(f"🔹 Single-Vector Indexing: {idx_time_single:.1f}ms (Total)")

        # --- METHOD B: MULTI VECTOR ---
        t0 = time.time()
        print("🔸 Multi-Vector Indexing (Parallel)...", end=" ")
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Pass current_dim to concurrent calls
            future_job = executor.submit(get_embeddings, CORPUS_JOBS, "RETRIEVAL_DOCUMENT", current_dim)
            future_edu = executor.submit(get_embeddings, CORPUS_EDUS, "RETRIEVAL_DOCUMENT", current_dim)
            future_loc = executor.submit(get_embeddings, CORPUS_LOCS, "RETRIEVAL_DOCUMENT", current_dim)
            
            vecs_job = future_job.result()
            vecs_edu = future_edu.result()
            vecs_loc = future_loc.result()
            
        idx_time_multi = (time.time() - t0) * 1000
        print(f"Done in {idx_time_multi:.1f}ms (Total)")
        
        
        print("\n🔍 [Search Phase & Comparison]")
        
        # Helper Inner Loop Logic functions need access to vecs which change per dim
        # Bias Calc
        def calculate_focus(retrieved_ids, intent_job, intent_edu, intent_loc):
            hits_job = 0
            hits_edu = 0
            hits_loc = 0
            total = len(retrieved_ids)
            if total == 0: return "N/A"
            for uid in retrieved_ids:
                idx = IDS.index(uid)
                row = df_corpus.iloc[idx]
                if any(j.lower() in str(row['Job_Title']).lower() for j in intent_job): hits_job += 1
                if any(e.lower() in str(row['Education']).lower() for e in intent_edu): hits_edu += 1
                if any(l.lower() in str(row['Location']).lower() for l in intent_loc): hits_loc += 1
            return f"Job:{round(hits_job/total*100)}% | Edu:{round(hits_edu/total*100)}% | Loc:{round(hits_loc/total*100)}%"

        # Cascade Logic
        def search_cascade(job_sim, edu_sim, loc_sim, i_job, i_edu, i_loc):
            current_candidates = np.array(range(len(IDS)))
            candidate_scores = np.zeros(len(IDS))
            if i_job:
                scores = job_sim[current_candidates]
                top_k_indices = np.argsort(scores)[-min(50, len(scores)):] 
                current_candidates = current_candidates[top_k_indices]
                candidate_scores[current_candidates] += job_sim[current_candidates] * 2.0
            if i_edu:
                scores = edu_sim[current_candidates]
                top_k_indices = np.argsort(scores)[-min(20, len(scores)):]
                current_candidates = current_candidates[top_k_indices]
                candidate_scores[current_candidates] += edu_sim[current_candidates] * 1.0
            if i_loc:
                scores = loc_sim[current_candidates]
                top_k_indices = np.argsort(scores)[-min(10, len(scores)):]
                current_candidates = current_candidates[top_k_indices]
                candidate_scores[current_candidates] += loc_sim[current_candidates] * 0.5
            final_subset_scores = candidate_scores[current_candidates]
            sorted_local_indices = np.argsort(final_subset_scores)[::-1]
            final_ranked_indices = current_candidates[sorted_local_indices]
            final_scores = np.zeros(len(IDS))
            for rank, idx in enumerate(final_ranked_indices):
                final_scores[idx] = 1.0 / (rank + 1)
            return final_scores

        for scen in FULL_SCENARIOS:
            print(f"\n--- Scenario: {scen['name']} ({current_dim}d) ---")
            gt = scen['ground_truth']
            
            # Intent Lists
            i_job = scen['multi_pref']['job']
            i_loc = scen['multi_pref']['loc']
            i_edu = scen['multi_pref']['edu']
            
            # --- DYNAMIC QUERY GENERATION ---
            scen['single_query'] = generate_query_variations(scen['multi_pref'])
            
            # --- EXECUTE A: SINGLE ---
            t_start = time.time()
            q_vec = get_embeddings([scen['single_query']], "RETRIEVAL_QUERY", current_dim)[0]
            scores_a = cosine_similarity([q_vec], vecs_single)[0]
            time_a = (time.time() - t_start) * 1000
            
            # --- EXECUTE B/C: MULTI PREP (Parallel) ---
            t_start_par = time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                f1 = executor.submit(get_embeddings, [", ".join(i_job) if i_job else " "], "RETRIEVAL_QUERY", current_dim)
                f2 = executor.submit(get_embeddings, [", ".join(i_edu) if i_edu else " "], "RETRIEVAL_QUERY", current_dim)
                f3 = executor.submit(get_embeddings, [", ".join(i_loc) if i_loc else " "], "RETRIEVAL_QUERY", current_dim)
                q_j = f1.result()[0]
                q_e = f2.result()[0]
                q_l = f3.result()[0]
            
            sim_j = cosine_similarity([q_j], vecs_job)[0]
            sim_e = cosine_similarity([q_e], vecs_edu)[0]
            sim_l = cosine_similarity([q_l], vecs_loc)[0]
            
            # PARALLEL
            has_job, has_edu, has_loc = bool(i_job), bool(i_edu), bool(i_loc)
            raw_w = [W_JOB if has_job else 0, W_EDU if has_edu else 0, W_LOC if has_loc else 0]
            total_w = sum(raw_w)
            if total_w == 0: weights = [0, 0, 0]
            else: weights = [x/total_w for x in raw_w]
            scores_parallel = (sim_j * weights[0]) + (sim_e * weights[1]) + (sim_l * weights[2])
            time_par = (time.time() - t_start_par) * 1000
            
            # CASCADE
            t_start_seq = time.time()
            scores_cascade = search_cascade(sim_j, sim_e, sim_l, i_job, i_edu, i_loc)
            time_seq_logic = (time.time() - t_start_seq) * 1000
            
            num_filters = max(1, sum([bool(i_job), bool(i_edu), bool(i_loc)]))
            time_seq_total = time_seq_logic + (time_a * num_filters)
            
            # Evaluations
            def calc_metrics(scores):
                top_idx = np.argsort(scores)[::-1][:5]
                retrieved = [IDS[i] for i in top_idx]
                dcg = 0
                for i, uid in enumerate(retrieved):
                    rel = 3 if uid in gt['tier_1'] else (2 if uid in gt['tier_2'] else 0)
                    dcg += rel / np.log2(i + 2)
                
                ideal_rels = ([3] * len(gt['tier_1'])) + ([2] * len(gt['tier_2']))
                ideal_rels.sort(reverse=True)
                num_ideal = min(len(ideal_rels), 5)
                idcg = 0
                for i in range(num_ideal): idcg += ideal_rels[i] / np.log2(i + 2)
                
                ndcg = 0 if idcg == 0 else (dcg / idcg)
                focus = calculate_focus(retrieved, i_job, i_edu, i_loc)
                return ndcg, focus, top_idx

            dcg_a, focus_a, top_a = calc_metrics(scores_a)
            dcg_p, focus_p, top_p = calc_metrics(scores_parallel)
            dcg_c, focus_c, top_c = calc_metrics(scores_cascade)
            
            # Print to Terminal
            print(f"   ⏱️  Single: {round(time_a)}ms | Par: {round(time_par)}ms | Seq: ~{round(time_seq_total)}ms")
            print(f"   🎯  nDCG -> Single: {round(dcg_a,3)} | Par: {round(dcg_p,3)} | Seq: {round(dcg_c,3)}\n")

            # Capture Detailed Results
            def get_profile_cards(indices):
                cards = []
                for i in indices:
                    uid = IDS[i]
                    is_tier1 = uid in gt['tier_1']
                    is_tier2 = uid in gt['tier_2']
                    badge = ""
                    border_color = "#334155" 
                    if is_tier1: 
                        badge = "<span style='background:#22c55e; color:black; padding:2px 6px; border-radius:4px; font-size:10px; font-weight:bold;'>👑 Perfect</span>"
                        border_color = "#22c55e"
                    elif is_tier2:
                        badge = "<span style='background:#f59e0b; color:black; padding:2px 6px; border-radius:4px; font-size:10px; font-weight:bold;'>🥈 Good</span>"
                        border_color = "#f59e0b"
                    
                    prof = PROFILES_DATA[i]
                    card = f"""
                    <div style="background: #1e293b; border-left: 4px solid {border_color}; padding: 10px; margin-bottom: 8px; border-radius: 4px;">
                        <div style="font-weight: bold; color: #f8fafc; font-size: 14px;">
                            {df_corpus.iloc[i]['Name']} {badge}
                        </div>
                        <div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">
                            {prof['Job_Title']} | {prof['Location']}
                        </div>
                    </div>
                    """
                    cards.append(card)
                return "".join(cards)

            results.append({
                "Scenario": f"{scen['name']} ({current_dim}d)",
                "Intent": f"Job: {i_job}<br>Edu: {i_edu}<br>Loc: {i_loc}",
                
                "Single_Metric": f"⏱️ {round(time_a)}ms<br>🎯 {round(dcg_a,2)}<br>🧭 {focus_a}",
                "MultiPar_Metric": f"⏱️ {round(time_par)}ms<br>🎯 {round(dcg_p,2)}<br>🧭 {focus_p}",
                "MultiCas_Metric": f"⏱️ ~{round(time_seq_total)}ms<br>🎯 {round(dcg_c,2)}<br>🧭 {focus_c}", 
                
                "Single_Prof": get_profile_cards(top_a[:3]),
                "Multi_Prof": get_profile_cards(top_p[:3]),
                "Cascade_Prof": get_profile_cards(top_c[:3]),
                
                "Winner": "Cascade 🏆" if dcg_c > dcg_p else ("Parallel 🏆" if dcg_p > dcg_a else "Single")
            })

    # HTML Report Generation
    html_rows = ""
    for res in results:
        row = f"""
        <div class="scenario-box">
             <div class="header">
                <h3>{res['Scenario']}</h3>
                <span class="winner-badge" style="background:#5D5DFF">{res['Winner']} wins</span>
            </div>
            <p style="color:#aaa; font-size:12px;">{res['Intent']}</p>
            
            <div class="comparison-grid" style="grid-template-columns: 1fr 1fr 1fr;">
                <!-- SINGLE -->
                <div class="col">
                    <div class="col-title">Single Vector</div>
                    <div class="stats">{res['Single_Metric']}</div>
                    <div class="profiles">{res['Single_Prof']}</div>
                </div>
                <!-- MULTI PARALLEL -->
                <div class="col">
                    <div class="col-title" style="color:#22c55e">Multi (Parallel)</div>
                    <div class="stats">{res['MultiPar_Metric']}</div>
                    <div class="profiles">{res['Multi_Prof']}</div>
                </div>
                <!-- MULTI SEQUENTIAL -->
                <div class="col">
                    <div class="col-title" style="color:#f59e0b">Multi (Sequential)</div>
                    <div class="stats">{res['MultiCas_Metric']}</div>
                    <div class="profiles">{res['Cascade_Prof']}</div>
                </div>
            </div>
        </div>
        """
        html_rows += row

    full_html = f"""
    <html><head><style>
        body {{ font-family: sans-serif; background: #111; color: #fff; padding: 20px; }}
        .scenario-box {{ background: #222; margin-bottom: 20px; padding: 20px; border-radius: 8px; border: 1px solid #444; }}
        .comparison-grid {{ display: grid; gap: 20px; }}
        .stats {{ background: #333; padding: 10px; border-radius: 4px; margin-bottom: 10px; font-family: monospace; line-height: 1.6; }}
        .header {{ display: flex; justify-content: space-between; }}
        .winner-badge {{ background: #444; padding: 5px 10px; border-radius: 4px; }}
        .col-title {{ margin-bottom: 10px; font-weight: bold; color: #888; border-bottom: 2px solid #444; }}
    </style></head><body>
    <h1>🚀 Search Strategy Benchmark: Single vs Multi (Dimensions: 768, 1536)</h1>
    {html_rows}
    </body></html>
    """
    
    with open("multi_vec_report.html", "w", encoding="utf-8") as f: f.write(full_html)
    print("✅ Rich HTML Report Generated!")
    webbrowser.open(f'file:///{os.path.abspath("multi_vec_report.html")}')

if __name__ == "__main__":
    run_experiment()
