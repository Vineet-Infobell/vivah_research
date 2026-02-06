import pandas as pd
import psycopg2
from tqdm import tqdm
import time
import os
from pathlib import Path
from dotenv import load_dotenv

# Google GenAI Setup
from google import genai
from google.genai import types
from google.oauth2 import service_account

print("="*60)
print("🚀 MATCHMAKING DATA LOADER - PostgreSQL + Embeddings")
print("="*60)

# ========================
# 1. ENVIRONMENT SETUP
# ========================
print("\n🔧 Loading environment variables...")
env_path = Path('../../vivah_api/.env')
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded: {env_path.absolute()}")
else:
    load_dotenv()
    print("⚠️ Using current directory .env")

# ========================
# 2. GOOGLE GENAI CLIENT
# ========================
print("\n🔑 Initializing Google GenAI...")

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

print(f"✅ GenAI Client Ready")
print(f"   Model: gemini-embedding-001")
print(f"   Dimension: 1152")

# ========================
# 3. DATABASE CONNECTION
# ========================
print("\n🔌 Connecting to PostgreSQL...")
try:
    conn = psycopg2.connect(
        host="localhost",
        port="5433",
        database="postgres",
        user="postgres",
        password="matchpass"
    )
    conn.autocommit = True
    cur = conn.cursor()
    print("✅ Database connected!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit(1)

# ========================
# 4. TABLE SETUP
# ========================
print("\n🔄 Setting up table...")
try:
    cur.execute("DROP TABLE IF EXISTS users;")
    cur.execute("""
        CREATE TABLE users (
            user_id INT PRIMARY KEY,
            name TEXT,
            gender TEXT,
            age INT,
            religion TEXT,
            location TEXT,
            education TEXT,
            job_title TEXT,
            user_vector vector(1152),
            preferences TEXT
        );
    """)
    print("✅ Table 'users' created successfully!")
except Exception as e:
    print(f"❌ Table creation failed: {e}")
    cur.close()
    conn.close()
    exit(1)

# ========================
# 5. EMBEDDING FUNCTION
# ========================
def create_embedding(text: str) -> list:
    """Generate 1152-dim embedding using gemini-embedding-001"""
    if not text or pd.isna(text):
        return None
    
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=str(text),
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=1152
            )
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"\n⚠️ Embedding Error: {e}")
        return None

# ========================
# 6. LOAD CSV
# ========================
print("\n📊 Loading dataset...")
try:
    df = pd.read_csv('matchmaking_1000_clean.csv')
    print(f"✅ Loaded {len(df)} profiles")
    print(f"   Columns: {list(df.columns)}")
except Exception as e:
    print(f"❌ CSV loading failed: {e}")
    cur.close()
    conn.close()
    exit(1)

# ========================
# 7. PROCESS & INSERT
# ========================
print("\n🚀 Generating embeddings and inserting...")
print("⏱️ Estimated time: ~10-15 minutes for 1000 profiles\n")

success_count = 0
error_count = 0
embedding_times = []  # Track individual embedding times

start_time = time.time()  # Total process start

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
    # Generate NLP text (Option B format)
    user_info_text = f"I am a {row['Job_Title']}, completed {row['Education']}, and live in {row['Location']}."
    
    # Generate embedding (with timing)
    emb_start = time.time()
    user_vec = create_embedding(user_info_text)
    emb_time = (time.time() - emb_start) * 1000  # Convert to ms
    
    if user_vec:
        embedding_times.append(emb_time)
        
        try:
            # Convert to PostgreSQL vector format
            vec_str = '[' + ','.join(map(str, user_vec)) + ']'
            
            # Insert into database
            cur.execute("""
                INSERT INTO users (
                    user_id, name, gender, age, religion, location,
                    education, job_title, user_vector, preferences
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
            """, (
                int(row['User_ID']),
                str(row['Name']),
                str(row['Gender']),
                int(row['Age']),
                str(row['Religion']),
                str(row['Location']),
                str(row['Education']),
                str(row['Job_Title']),
                vec_str,
                str(row['preferences'])
            ))
            success_count += 1
            
        except Exception as e:
            error_count += 1
            print(f"\n❌ DB Error at User {row['User_ID']}: {e}")
    else:
        error_count += 1
        print(f"\n⚠️ Embedding failed for User {row['User_ID']}")
    
    # Rate limiting (API safety)
    time.sleep(0.15)

# ========================
# 8. SUMMARY
# ========================
total_time = time.time() - start_time

print("\n" + "="*60)
print("📊 PROCESS COMPLETE!")
print("="*60)
print(f"✅ Successfully inserted: {success_count}/{len(df)}")
print(f"❌ Errors: {error_count}")
print(f"📈 Success rate: {(success_count/len(df)*100):.1f}%")

# Timing Statistics
print("\n" + "="*60)
print("⏱️  TIMING METRICS")
print("="*60)
print(f"🕐 Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

if embedding_times:
    avg_emb_time = sum(embedding_times) / len(embedding_times)
    min_emb_time = min(embedding_times)
    max_emb_time = max(embedding_times)
    
    print(f"📊 Embedding Generation:")
    print(f"   Average: {avg_emb_time:.2f} ms/profile")
    print(f"   Min:     {min_emb_time:.2f} ms")
    print(f"   Max:     {max_emb_time:.2f} ms")
    print(f"   Total:   {len(embedding_times)} embeddings generated")
    
    # Calculate throughput
    throughput = success_count / total_time if total_time > 0 else 0
    print(f"🚀 Throughput: {throughput:.2f} profiles/second")


# Verify
cur.execute("SELECT COUNT(*) FROM users;")
db_count = cur.fetchone()[0]
print(f"\n🔍 Database verification: {db_count} records")

# Sample data
print("\n📋 Sample records:")
cur.execute("SELECT user_id, name, job_title, location FROM users LIMIT 5;")
for record in cur.fetchall():
    print(f"   User {record[0]}: {record[1]} - {record[2]} ({record[3]})")

# Test vector dimension (Commented out - pgvector doesn't support ::float[] cast)
# print("\n🔬 Testing vector dimensions:")
# Alternative: Just verify data exists
print("\n✅ Data verification complete!")


cur.close()
conn.close()
print("\n✅ Database connection closed.")
print("="*60)
