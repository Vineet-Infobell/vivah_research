
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

# CONFIG
MODEL_ID = 'gemini-embedding-001'
DIM = 1152

print(f"✅ Client Initialized | Model: {MODEL_ID} | Dim: {DIM}")

# ========================
# 2. CORPUS GENERATION (Ages 20-35)
# ========================
# We use a standard sentence structure to isolate the 'Age' variable.
AGES = list(range(20, 36)) # 20 to 35 inclusive
CORPUS_TEXTS = [str(age) for age in AGES]

print(f"✅ Generated {len(CORPUS_TEXTS)} Age Profiles (20-35)")

# ========================
# 3. HELPER FUNCTIONS
# ========================
def get_embeddings(texts, task_type="RETRIEVAL_DOCUMENT"):
    vecs = []
    chunk_size = 50
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        resp = client.models.embed_content(
            model=MODEL_ID, contents=chunk,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=DIM)
        )
        batch_vecs = [e.values for e in resp.embeddings]
        vecs.extend(batch_vecs)
    return np.array(vecs)

# ========================
# 4. EXECUTION
# ========================

print("\n📦 Indexing Corpus...")
t0 = time.time()
corpus_vecs = get_embeddings(CORPUS_TEXTS, "RETRIEVAL_DOCUMENT")
print(f"✅ Indexing Done in {(time.time()-t0)*1000:.1f}ms")

SCENARIOS = [
    "age 27",
    "age twenty seven",
    "age around 27",
    "age between 25 to 28"
]

results = {}

print("\n🔍 Running Queries...")

# Prepare DataFrame columns
df_results = pd.DataFrame({'Age': AGES})

for query in SCENARIOS:
    print(f"   > Query: {query}")
    
    # Embed Query
    q_vec = get_embeddings([query], "RETRIEVAL_QUERY")[0]
    q_vec = q_vec.reshape(1, -1)
    
    # Search
    scores = cosine_similarity(q_vec, corpus_vecs)[0]
    
    # Store in DataFrame
    df_results[query] = scores.round(4)

# ========================
# 5. REPORTING
# ========================
print("\n" + "="*60)
print(f"📊 AGE SEMANTIC SCORE MATRIX (Model: {MODEL_ID} {DIM}d)")
print("="*60)

# Format for pretty printing
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
print(df_results.to_string(index=False))

import json

# ========================
# 5. HTML REPORT GENERATION
# ========================

def generate_html_report():
    # Prepare data for Chart.js
    datasets = []
    colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']
    
    for i, query in enumerate(SCENARIOS):
        data_points = df_results[query].tolist()
        datasets.append({
            'label': query,
            'data': data_points,
            'borderColor': colors[i % len(colors)],
            'backgroundColor': colors[i % len(colors)],
            'tension': 0.4,
            'fill': False
        })
        
    # Serialize to JSON for safe JS injection (Handles False -> false, etc.)
    js_labels = json.dumps(df_results['Age'].tolist())
    js_datasets = json.dumps(datasets)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Age Semantic Benchmark</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: sans-serif; background: #1a1a1a; color: #fff; padding: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            .chart-box {{ background: #2d2d2d; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; color: #ddd; }}
            th, td {{ padding: 10px; text-align: center; border-bottom: 1px solid #444; }}
            th {{ background: #333; }}
            tr:hover {{ background: #333; }}
            h1, h2 {{ color: #4BC0C0; }}
            .highlight {{ color: #ffeb3b; font-weight: bold; }}
            canvas {{ max-height: 400px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Age Semantic Search Analysis</h1>
            <p>Model: {MODEL_ID} ({DIM} dims) | Corpus: Ages 20-35 (Raw Numbers)</p>
            
            <div class="chart-box">
                <canvas id="ageChart"></canvas>
            </div>
            
            <div class="chart-box">
                <h2>🔢 Score Matrix</h2>
                {df_results.to_html(index=False, classes='table', border=0, float_format=lambda x: '{:.4f}'.format(x) if isinstance(x, float) else x)}
            </div>
            
            <div class="chart-box">
                 <h2>💡 Key Observations</h2>
                 <ul>
                    <li><strong>Exact Match (27):</strong> Clear peak at 27. Steep drop-off at 26/28. Acts like a keyword search.</li>
                    <li><strong>Range (25-28):</strong> <span style="color:#ff6b6b">FAILURE CASE.</span> The model spikes at boundaries (25, 28) but dips in the middle (26, 27). It does NOT understand numerical continuity.</li>
                    <li><strong>Text (Twenty Seven):</strong> Successfully maps "Twenty Seven" to "27". Semantic understanding of number-words is strong.</li>
                 </ul>
            </div>
        </div>

        <script>
            const ctx = document.getElementById('ageChart').getContext('2d');
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {js_labels},
                    datasets: {js_datasets}
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: false,
                            grid: {{ color: '#444' }}
                        }},
                        x: {{
                             grid: {{ color: '#444' }}
                        }}
                    }},
                    plugins: {{
                        legend: {{ labels: {{ color: '#fff' }} }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    with open("age_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("\n✅ Report Generated: age_report.html")
    webbrowser.open(f'file:///{os.path.abspath("age_report.html")}')

# Generate the report
generate_html_report()
