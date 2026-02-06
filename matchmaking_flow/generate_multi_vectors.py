"""
Generate Multi-Vector Embeddings for All Users
-----------------------------------------------
This script creates 3 separate embeddings for each user:
1. Profession Embedding (from job_title)
2. Education Embedding (from education)
3. Location Embedding (from location)

Stores them in new database columns for true multi-vector matching.
"""

import psycopg2
from google import genai
from google.genai import types
from google.oauth2 import service_account
import os
from pathlib import Path
from dotenv import load_dotenv
import time

# ========================
# 1. SETUP
# ========================

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

# Database connection
conn = psycopg2.connect(
    host="localhost",
    port="5433",
    database="postgres",
    user="postgres",
    password="matchpass"
)
cur = conn.cursor()

def create_embedding(text: str) -> list:
    """Generate 1152-dim embedding using gemini-embedding-001"""
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
        print(f"Embedding Error for '{text}': {e}")
        return None

# ========================
# 2. ADD NEW COLUMNS
# ========================

print("\n" + "="*80)
print("📊 STEP 1: Adding New Columns to Database")
print("="*80)

try:
    # Add profession_vector column
    cur.execute("""
        ALTER TABLE users 
        ADD COLUMN IF NOT EXISTS profession_vector vector(1152);
    """)
    print("✅ Added profession_vector column")
    
    # Add education_vector column
    cur.execute("""
        ALTER TABLE users 
        ADD COLUMN IF NOT EXISTS education_vector vector(1152);
    """)
    print("✅ Added education_vector column")
    
    # Add location_vector column
    cur.execute("""
        ALTER TABLE users 
        ADD COLUMN IF NOT EXISTS location_vector vector(1152);
    """)
    print("✅ Added location_vector column")
    
    conn.commit()
    print("\n✅ All columns added successfully!")
    
except Exception as e:
    print(f"❌ Error adding columns: {e}")
    conn.rollback()

# ========================
# 3. GENERATE EMBEDDINGS
# ========================

print("\n" + "="*80)
print("🚀 STEP 2: Generating Multi-Vector Embeddings for All Users")
print("="*80)

# Get all users
cur.execute("""
    SELECT user_id, job_title, education, location 
    FROM users 
    ORDER BY user_id;
""")

users = cur.fetchall()
total_users = len(users)
print(f"\n📊 Found {total_users} users to process")

success_count = 0
error_count = 0

for i, (user_id, job_title, education, location) in enumerate(users, 1):
    try:
        print(f"\n[{i}/{total_users}] Processing User {user_id}...")
        print(f"  Job: {job_title}")
        print(f"  Edu: {education}")
        print(f"  Loc: {location}")
        
        # Generate 3 separate embeddings
        profession_vec = create_embedding(job_title)
        education_vec = create_embedding(education)
        location_vec = create_embedding(location)
        
        if profession_vec and education_vec and location_vec:
            # Convert to string format for PostgreSQL
            prof_vec_str = '[' + ','.join(map(str, profession_vec)) + ']'
            edu_vec_str = '[' + ','.join(map(str, education_vec)) + ']'
            loc_vec_str = '[' + ','.join(map(str, location_vec)) + ']'
            
            # Update database
            cur.execute("""
                UPDATE users 
                SET 
                    profession_vector = %s::vector,
                    education_vector = %s::vector,
                    location_vector = %s::vector
                WHERE user_id = %s;
            """, (prof_vec_str, edu_vec_str, loc_vec_str, user_id))
            
            conn.commit()
            success_count += 1
            print(f"  ✅ Vectors stored successfully")
        else:
            error_count += 1
            print(f"  ❌ Failed to generate one or more embeddings")
        
        # Rate limiting - small delay
        time.sleep(0.1)
        
    except Exception as e:
        error_count += 1
        print(f"  ❌ Error: {e}")
        conn.rollback()

# ========================
# 4. SUMMARY
# ========================

print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)
print(f"Total Users:     {total_users}")
print(f"✅ Success:      {success_count}")
print(f"❌ Errors:       {error_count}")
print(f"Success Rate:    {(success_count/total_users)*100:.1f}%")

# Verify columns
cur.execute("""
    SELECT COUNT(*) 
    FROM users 
    WHERE profession_vector IS NOT NULL 
      AND education_vector IS NOT NULL 
      AND location_vector IS NOT NULL;
""")
complete_count = cur.fetchone()[0]
print(f"\n✅ Users with all 3 vectors: {complete_count}/{total_users}")

print("\n" + "="*80)
print("✅ Multi-Vector Generation Complete!")
print("="*80)

conn.close()
