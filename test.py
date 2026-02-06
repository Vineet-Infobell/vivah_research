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

# Model ID Mapping (Display Name -> Actual API ID)
# Update these IDs based on your specific GCP Model availability
MODEL_MAP = {
    'Gemini Embedding 001': 'gemini-embedding-001',     
    'Text Embedding 005': 'text-embedding-005',        
    'Text Embedding 004': 'text-embedding-004',        
    'Multilingual 002': 'text-multilingual-embedding-002' 
}

# ========================
# 2. DATA & GROUND TRUTH
# ========================

# Load Data
try:
    df = pd.read_csv('F:\Vivahai\Vivahai_backend_v2\matchmaking_search_lab\matchmaking_structured_preferences.csv').head(100)
    print(f"✅ Loaded {len(df)} profiles")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    exit()

# Construct Embedding Text (Strict: Job + Education + Location)
df['embed_text'] = df.apply(
    lambda x: f" Works as a {x['Job_Title']}, holds a degree in {x['Education']}, and is currently located in {x['Location']}.", 
    axis=1
)

# New Queries List with Ground Truth
QUERIES = [
    {
        'id': 1,
        'text': "I am looking for a Software Developer with Computer Science degree.",
        'gt_ids': [60, 7, 75, 12, 57]
    },
    {
        'id': 2,
        'text': "Searching for a Healthcare Professional specializing in Surgical Operations with a Postgraduate Medical degree, currently in Mumbai.",
        'gt_ids': [86, 90, 27, 61, 33]
    },
    {
        'id': 3,
        'text': "I am looking for a Digital Media Professional based in Kolkata who specializes in visual arts, media production, or text composition.",
        'gt_ids': [14, 18, 25, 61, 33]
    },
    {
        'id': 4,
        'text': "Seeking an individual based in Delhi who holds a Master's degree or any other postgraduate qualification, regardless of their current profession.",
        'gt_ids': [21, 76, 97, 67, 53]
    },
    {
        'id': 5,
        'text': "Searching for an Academic Researcher or scholar who studies Legal Systems and is based in Kolkata.",
        'gt_ids': [3, 62, 52, 44, 81]
    }
]

# ========================
# 3. HELPER FUNCTIONS
# ========================

def calculate_metrics(recommended_ids, gt_ids, k=5):
    """Calculates Precision@k, Recall@k, MRR, nDCG"""
    relevant = [idx for idx in recommended_ids[:k] if idx in gt_ids]
    
    # Precision & Recall
    precision = len(relevant) / k
    recall = len(relevant) / len(gt_ids) if len(gt_ids) > 0 else 0
    
    # MRR
    mrr = 0
    for i, idx in enumerate(recommended_ids):
        if idx in gt_ids:
            mrr = 1 / (i + 1)
            break
            
    # nDCG
    dcg = sum([1 / np.log2(i + 2) for i, idx in enumerate(recommended_ids[:k]) if idx in gt_ids])
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(gt_ids), k))])
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return precision, recall, mrr, ndcg

# ========================
# 4. BENCHMARKING ENGINE
# ========================

def run_model(display_name, dimension):
    # Map display name to actual model ID
    model_id = MODEL_MAP.get(display_name, display_name)
    
    print(f"\n🚀 STARTING BENCHMARK: {display_name} -> {model_id} (Dim: {dimension})")
    
    # --- Step 1: Embed Documents (Batch) ---
    print("   Creating Document Embeddings...", end=" ")
    start_embed = time.time()
    
    try:
        doc_response = client.models.embed_content(
            model=model_id,
            contents=df['embed_text'].tolist(),
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=dimension
            )
        )
        # Extract embeddings
        doc_embeddings = np.array([e.values for e in doc_response.embeddings])
        print(f"Done! ({len(doc_embeddings)} docs)")
    except Exception as e:
        print(f"\n❌ Error embedding docs: {e}")
        return

    embed_latency_ms = (time.time() - start_embed) * 1000
    print(f"   ⏱️  Doc Embedding Latency: {embed_latency_ms:.2f} ms")

    results = []
    doc_ids = df['User_ID'].values
    
    # --- Step 2: Process Queries ---
    print("   Processing Queries...", end=" ")
    
    for q_item in QUERIES:
        q_id = q_item['id']
        q_text = q_item['text']
        gt_ids = q_item['gt_ids']
        
        # Embed Query
        try:
            q_resp = client.models.embed_content(
                model=model_id,
                contents=q_text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=dimension
                )
            )
            q_vec = np.array(q_resp.embeddings[0].values).reshape(1, -1)
        except Exception as e:
            print(f"❌ Query Error (ID {q_id}): {e}")
            continue

        # --- Step 3: Calculate Distances & Metrics ---
        for dist_type in ['cosine', 'l1', 'l2', 'inner_product']:
            start_search = time.time()
            
            if dist_type == 'cosine':
                scores = cosine_similarity(q_vec, doc_embeddings)[0]
            elif dist_type == 'l2':
                scores = -euclidean_distances(q_vec, doc_embeddings)[0] 
            elif dist_type == 'l1':
                scores = -manhattan_distances(q_vec, doc_embeddings)[0]
            elif dist_type == 'inner_product':
                scores = np.dot(doc_embeddings, q_vec.T).flatten()

            # Get Top 5 IDs
            top_indices = np.argsort(scores)[::-1][:5]
            recommended_ids = doc_ids[top_indices]

            # Print top 5 profiles for this query/model/distance
            print(f"\n🔝 Top 5 profiles for Query {q_id} [{dist_type}] ({display_name}):")
            for rank, idx in enumerate(recommended_ids, 1):
                profile = df[df['User_ID'] == idx].iloc[0]
                print(f"  {rank}. User_ID: {idx} | Profession: {profile['Job_Title']} | Education: {profile['Education']} | Location: {profile['Location']}")

            search_latency = (time.time() - start_search) * 1000

            # Compute Ranking Metrics
            p, r, mrr, ndcg = calculate_metrics(recommended_ids, gt_ids)

            # Print metrics immediately after top 5
            print(f"    Precision@5: {p:.2f} | Recall@5: {r:.2f} | MRR: {mrr:.2f} | nDCG: {ndcg:.2f} | Latency: {search_latency:.2f} ms")

            results.append({
                'Model': display_name,
                'Dim': dimension,
                'Query_ID': q_id,
                'Distance': dist_type,
                'Precision@5': p,
                'Recall@5': r,
                'MRR': mrr,
                'nDCG': ndcg,
                'Latency(ms)': search_latency
            })
    
    print("Done!")

    # --- Step 4: Summary Report ---
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        summary = res_df.groupby('Distance')[['Precision@5', 'Recall@5', 'nDCG', 'Latency(ms)']].mean()
        print(f"\n📊 SUMMARY RESULTS ({display_name} - {dimension}):")
        print(summary)
        
        # Save detailed results
        filename = f"benchmark_{display_name.replace(' ', '_')}_{dimension}.csv"
        res_df.to_csv(filename, index=False)
        print(f"   💾 Saved detailed results to {filename}")
    else:
        print("⚠️ No results to summarize.")
    print("-" * 60)

# ========================
# 5. SPECIFIC MODEL RUNNERS
# ========================

def run_gemini_001_768():
    run_model('Gemini Embedding 001', 768)

def run_gemini_001_1152():
    run_model('Gemini Embedding 001', 1152)

def run_gemini_001_1536():
    run_model('Gemini Embedding 001', 1536)

def run_gemini_001_3072():
    run_model('Gemini Embedding 001', 3072)

def run_text_embedding_005_768():
    run_model('Text Embedding 005', 768)

def run_text_embedding_004_768():
    run_model('Text Embedding 004', 768)

def run_multilingual_002_768():
    run_model('Multilingual 002', 768)

# ========================
# 6. MAIN EXECUTION
# ========================

if __name__ == "__main__":
    print("🚦 STARTING ALL BENCHMARKS...\n")
    
    # Comment/Uncomment the ones you want to run
    
    # --- Text Embedding 005 ---
    run_text_embedding_005_768()
    
    # --- Text Embedding 004 ---
    run_text_embedding_004_768()
    
    # --- Multilingual ---
    run_multilingual_002_768()
    
    # --- Gemini 001 (Various Dimensions) ---
    # Note: Ensure your model supports these dimensions, otherwise API will error
    run_gemini_001_768()
    run_gemini_001_1152()
    run_gemini_001_1536()
    run_gemini_001_3072()
    
    print("\n✅ All Benchmarks Completed.")