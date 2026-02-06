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

import warnings
import numpy as np
warnings.filterwarnings('ignore')

print("✅ Imports successful")
# ========================
# 1. CONFIGURATION & SETUP
# ========================

# Load Credentials

# Load environment variables
import os  # <-- Add this line
env_path = Path('../vivah_api/.env')
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env from: {env_path.absolute()}")
else:
    load_dotenv()
    print("⚠️  Using default .env")
# Get API Key
API_KEY = os.getenv("GEMINI_API_KEY")
# Get credentials path
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if CREDENTIALS_PATH and not os.path.isabs(CREDENTIALS_PATH):
    CREDENTIALS_PATH = str((Path('../vivah_api') / CREDENTIALS_PATH).absolute())

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "triple-name-473801-t7")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

print(f"\n📋 Configuration:")
print(f"   Project: {PROJECT_ID}")
print(f"   Location: {LOCATION}")
print(f"   Credentials: {CREDENTIALS_PATH}")
print(f"   Exists: {os.path.exists(CREDENTIALS_PATH) if CREDENTIALS_PATH else False}")

# Initialize Google GenAI Client

# Create credentials
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
credentials = service_account.Credentials.from_service_account_file(
    CREDENTIALS_PATH, scopes=SCOPES
)

# Initialize GenAI client
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    credentials=credentials,
)

print("✅ Google GenAI client initialized successfully!")


CONFIGS_TO_TEST = [
    # Baseline Models (Standard 768)
    {'name': 'Text-Embed-004 (768)',  'id': 'models/text-embedding-004', 'dim': 768},
    {'name': 'Text-Embed-005 (768)',  'id': 'models/text-embedding-005', 'dim': 768},
    
    # Gemini 001 Variable Dimensions Test
    {'name': 'Gemini-001 (768)',   'id': 'models/gemini-embedding-001', 'dim': 768},
    {'name': 'Gemini-001 (1152)',  'id': 'models/gemini-embedding-001', 'dim': 1152},
    {'name': 'Gemini-001 (1536)',  'id': 'models/gemini-embedding-001', 'dim': 1536},
    {'name': 'Gemini-001 (3072)',  'id': 'models/gemini-embedding-001', 'dim': 3072},
]


# ========================
# 2. CORPUS (Professions Database)
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
# 3. DIRECT SHORT QUERIES (Updated)
# ========================
TEST_SCENARIOS = [
    {
        "query": "Software Developer",
        "ground_truth_top_5": ["Software Engineer", "Backend Developer", "Full Stack Developer", "Frontend Developer", "Senior Software Developer"]
    },
    {
        "query": "Doctor",
        "ground_truth_top_5": ["Doctor", "General Physician", "Surgeon", "Cardiologist", "Medical Researcher"]
    },
    {
        "query": "Designer",
        "ground_truth_top_5": ["Graphic Designer", "UI/UX Designer", "Art Director", "Fashion Designer", "Interior Designer"]
    },
    {
        "query": "Lawyer",
        "ground_truth_top_5": ["Corporate Lawyer", "Advocate", "Legal Consultant", "Judge", "Tax Consultant"]
    },
    {
        "query": "Professor",
        "ground_truth_top_5": ["Professor", "Research Scholar", "Lecturer", "School Teacher", "Principal"]
    }
]

# ========================
# 4. METRICS CALCULATION
# ========================
def calculate_metrics(retrieved_jobs, expected_top_5, k=5):
    retrieved_k = retrieved_jobs[:k]
    matches = [job for job in retrieved_k if job in expected_top_5]
    
    precision = len(matches) / k
    recall = len(matches) / len(expected_top_5) if len(expected_top_5) > 0 else 0
    
    mrr = 0
    for i, job in enumerate(retrieved_jobs):
        if job in expected_top_5:
            mrr = 1 / (i + 1)
            break
            
    dcg = 0
    idcg = 0
    for i in range(k):
        if i < len(retrieved_k) and retrieved_k[i] in expected_top_5:
            dcg += 1 / np.log2(i + 2)
        if i < len(expected_top_5):
            idcg += 1 / np.log2(i + 2)
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return precision, recall, mrr, ndcg

# ========================
# 5. BENCHMARK ENGINE
# ========================
def run_benchmark():
    print(f"\n🚀 STARTING SEMANTIC BENCHMARK (Short Queries)")
    print("=" * 100)
    
    final_results = []

    for config in CONFIGS_TO_TEST:
        model_name = config['name']
        model_id = config['id']
        dim = config['dim']
        
        print(f"\n🔵 Processing: {model_name} (Dim: {dim}) ...", end=" ")
        
        # A. Embed Corpus
        try:
            doc_resp = client.models.embed_content(
                model=model_id, contents=CORPUS_JOBS,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=dim)
            )
            corpus_vecs = np.array([e.values for e in doc_resp.embeddings])
        except Exception as e:
            print(f"❌ Error embedding corpus: {e}")
            continue
        print("Done.")

        # Storage for this model's performance across all queries
        # Structure: {'cosine': {'p':[], 'r':[], ...}, 'l2': ...}
        model_stats = {
            'cosine': {'P': [], 'R': [], 'MRR': [], 'nDCG': []},
            'l2': {'P': [], 'R': [], 'MRR': [], 'nDCG': []},
            'l1': {'P': [], 'R': [], 'MRR': [], 'nDCG': []},
            'inner_product': {'P': [], 'R': [], 'MRR': [], 'nDCG': []}
        }

        # B. Loop Scenarios
        for scenario in TEST_SCENARIOS:
            query = scenario['query']
            gt_top_5 = scenario['ground_truth_top_5']
            
            try:
                # Embed Query
                q_resp = client.models.embed_content(
                    model=model_id, contents=query,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=dim)
                )
                q_vec = np.array(q_resp.embeddings[0].values).reshape(1, -1)
                
                # C. Calculate All Distances
                distances = {
                    'cosine': cosine_similarity(q_vec, corpus_vecs)[0],
                    'l2': -euclidean_distances(q_vec, corpus_vecs)[0], # Negative for sorting (closet=max)
                    'l1': -manhattan_distances(q_vec, corpus_vecs)[0], # Negative for sorting
                    'inner_product': np.dot(corpus_vecs, q_vec.T).flatten()
                }

                # D. Compute Metrics for each Distance
                for dist_name, scores in distances.items():
                    top_indices = np.argsort(scores)[::-1]
                    retrieved_jobs = [CORPUS_JOBS[i] for i in top_indices]
                    
                    p, r, mrr, ndcg = calculate_metrics(retrieved_jobs, gt_top_5, k=5)
                    
                    model_stats[dist_name]['P'].append(p)
                    model_stats[dist_name]['R'].append(r)
                    model_stats[dist_name]['MRR'].append(mrr)
                    model_stats[dist_name]['nDCG'].append(ndcg)
                    
            except Exception as e:
                print(f"   ❌ Query Error ({query}): {e}")

        # E. Aggregate Results for this Config
        for dist_name, metrics in model_stats.items():
            final_results.append({
                'Model': f"{model_name}",
                'Dim': dim,
                'Distance': dist_name,
                'Avg Precision@5': np.mean(metrics['P']),
                'Avg Recall@5': np.mean(metrics['R']),
                'Avg MRR': np.mean(metrics['MRR']),
                'Avg nDCG': np.mean(metrics['nDCG'])
            })

    # ========================
    # 6. FINAL REPORT
    # ========================
    print("\n\n" + "=" * 110)
    print(f"{'🏆 FINAL BENCHMARK LEADERBOARD':^110}")
    print("=" * 110)
    
    df_res = pd.DataFrame(final_results)
    if not df_res.empty:
        # Sort by nDCG to see best combination at the top
        df_res = df_res.sort_values(by=["Avg nDCG", "Avg MRR"], ascending=False)
        
        # Format for clean display
        print(df_res.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    else:
        print("No results generated.")
    print("=" * 110)

if __name__ == "__main__":
    run_benchmark()