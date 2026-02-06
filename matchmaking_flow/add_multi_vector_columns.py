"""
Add Multi-Vector Columns to Users Table
----------------------------------------
Adds 3 new vector columns for separate attribute embeddings:
- profession_vector (from job_title)
- education_vector (from education)
- location_vector (from location)

Then generates and populates embeddings for all existing users.
"""

import psycopg2
import time
from google import genai
from google.genai import types
from google.oauth2 import service_account
import os
from pathlib import Path
from dotenv import load_dotenv

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
        print(f"   ❌ Embedding Error: {e}")
        return None

print("\n" + "="*80)
print("STEP 1: Adding New Vector Columns to Users Table")
print("="*80)

# Check if columns already exist
cur.execute("""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name='users' 
    AND column_name IN ('profession_vector', 'education_vector', 'location_vector');
""")
existing_cols = [row[0] for row in cur.fetchall()]

if len(existing_cols) == 3:
    print("✅ All 3 vector columns already exist")
else:
    # Add columns that don't exist
    columns_to_add = []
    if 'profession_vector' not in existing_cols:
        columns_to_add.append('profession_vector')
    if 'education_vector' not in existing_cols:
        columns_to_add.append('education_vector')
    if 'location_vector' not in existing_cols:
        columns_to_add.append('location_vector')
    
    for col in columns_to_add:
        print(f"Adding column: {col}...")
        cur.execute(f"""
            ALTER TABLE users 
            ADD COLUMN IF NOT EXISTS {col} vector(1152);
        """)
        conn.commit()
        print(f"✅ Added {col}")

print("\n" + "="*80)
print("STEP 2: Generating Multi-Vector Embeddings for All Users")
print("="*80)

# Get all users
cur.execute("SELECT user_id, job_title, education, location FROM users ORDER BY user_id;")
users = cur.fetchall()

print(f"Found {len(users)} users to process\n")

success_count = 0
error_count = 0

for i, (user_id, job_title, education, location) in enumerate(users, 1):
    try:
        print(f"[{i}/{len(users)}] Processing User ID: {user_id}")
        print(f"   Job: {job_title}, Edu: {education}, Loc: {location}")
        
        # Generate 3 separate embeddings
        profession_vec = create_embedding(job_title)
        education_vec = create_embedding(education)
        location_vec = create_embedding(location)
        
        if not profession_vec or not education_vec or not location_vec:
            print(f"   ❌ Failed to generate embeddings")
            error_count += 1
            continue
        
        # Convert to PostgreSQL vector format
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
        print(f"   ✅ Updated successfully")
        
        # Rate limiting - small delay
        if i % 10 == 0:
            print(f"   ⏸️  Processed {i} users, taking a short break...")
            time.sleep(1)
        
    except Exception as e:
        print(f"   ❌ Error processing user {user_id}: {e}")
        error_count += 1
        conn.rollback()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✅ Successfully processed: {success_count} users")
print(f"❌ Errors: {error_count} users")
print(f"📊 Success rate: {(success_count/len(users)*100):.1f}%")

# Verify the data
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

cur.execute("""
    SELECT 
        COUNT(*) as total,
        COUNT(profession_vector) as has_prof,
        COUNT(education_vector) as has_edu,
        COUNT(location_vector) as has_loc
    FROM users;
""")
total, has_prof, has_edu, has_loc = cur.fetchone()

print(f"Total users: {total}")
print(f"Users with profession_vector: {has_prof}")
print(f"Users with education_vector: {has_edu}")
print(f"Users with location_vector: {has_loc}")

if has_prof == total and has_edu == total and has_loc == total:
    print("\n🎉 All users have complete multi-vector embeddings!")
else:
    print(f"\n⚠️  {total - min(has_prof, has_edu, has_loc)} users missing some vectors")

conn.close()
print("\n✅ Database connection closed")
