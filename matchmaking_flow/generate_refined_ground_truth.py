"""
Generate REFINED Ground Truth JSON (Max 20 Profiles with Tiered Relevance)
-------------------------------------------------------------------------
Tier 1 (Score 3): Perfect Match (Right Job, Loc, Age <= 3y gap)
Tier 2 (Score 2): Good Match (Right Job, Loc, Age <= 5y gap)
Tier 3 (Score 1): Fair Match (Related Job OR Wrong Loc if flexible)
Tier 0 (Score 0): Irrelevant
"""

import pandas as pd
import json

# Load dataset
df = pd.read_csv('matchmaking_1000_clean.csv')

# Load ID mapping
with open('ground_truth_id_mapping.json', 'r') as f:
    id_mapping = json.load(f)

# Define queries with ID Maps
queries_config = [
    {
        "scenario_id": 1,
        "scenario_name": "Software/IT Professional in Bangalore",
        "query": "Looking for Software Developer or SDE with B.Tech and lives in Bangalore",
        "searcher": {"gender": "Female", "age": 26, "religion": "Hindu"},
        "filters": {"gender": "Male", "religion": "Hindu", "preferred_age": 29},
        "criteria": {
            "jobs": ["SDE", "Software Developer", "Backend Developer", "Software Engineer"],
            "exact_jobs": ["SDE", "Software Developer"],
            "educations": ["B.Tech", "B.E.", "MCA", "M.Tech"],
            "location": "Bangalore",
            "location_flexible": False
        },
        "id_key": "query_1_ids"
    },
    {
        "scenario_id": 2,
        "scenario_name": "Medical Professional (Doctor/Surgeon)",
        "query": "Looking for Doctor or Surgeon with MBBS",
        "searcher": {"gender": "Female", "age": 28, "religion": "Hindu"},
        "filters": {"gender": "Male", "religion": "Hindu", "preferred_age": 29},
        "criteria": {
            "jobs": ["Doctor", "Surgeon", "Physician", "Medical Officer"],
            "exact_jobs": ["Doctor", "Surgeon"],
            "educations": ["MBBS", "MD"],
            "location": None,
            "location_flexible": True
        },
        "id_key": "query_2_ids"
    },
    {
        "scenario_id": 3,
        "scenario_name": "Finance Professional in Mumbai",
        "query": "Looking for Finance Professional (CA / Finance Manager / Analyst / Auditor) and lives in Mumbai",
        "searcher": {"gender": "Female", "age": 24, "religion": "Hindu"},
        "filters": {"gender": "Male", "religion": "Hindu", "preferred_age": 27},
        "criteria": {
            "jobs": ["Chartered Accountant", "Finance Manager", "Financial Analyst", "Auditor"],
            "exact_jobs": ["Chartered Accountant", "Finance Manager"],
            "educations": ["CA", "MBA", "M.Com", "CFA"],
            "location": "Mumbai",
            "location_flexible": False
        },
        "id_key": "query_3_ids"
    },
    {
        "scenario_id": 4,
        "scenario_name": "Designer/PM in Bangalore",
        "query": "Looking for UI/UX Designer or Product Manager with B.Tech and lives in Bangalore",
        "searcher": {"gender": "Male", "age": 30, "religion": "Hindu"},
        "filters": {"gender": "Female", "religion": "Hindu", "preferred_age": 29},
        "criteria": {
            "jobs": ["UI/UX Designer", "Product Manager", "Product Designer"],
            "exact_jobs": ["UI/UX Designer", "Product Manager"],
            "educations": ["B.Tech", "B.E.", "Bachelor of Design"],
            "location": "Bangalore",
            "location_flexible": False
        },
        "id_key": "query_4_ids"
    },
    {
        "scenario_id": 5,
        "scenario_name": "Data Scientist in Hyderabad",
        "query": "Looking for Data Scientist with M.Tech and lives in Hyderabad",
        "searcher": {"gender": "Male", "age": 27, "religion": "Hindu"},
        "filters": {"gender": "Female", "religion": "Hindu", "preferred_age": 25},
        "criteria": {
            "jobs": ["Data Scientist", "ML Engineer", "AI Engineer"],
            "exact_jobs": ["Data Scientist", "ML Engineer"],
            "educations": ["M.Tech", "MS in CS"],
            "location": "Hyderabad",
            "location_flexible": False
        },
        "id_key": "query_5_ids"
    }
]

def calculate_relevance_tier(row, criteria, preferred_age):
    score = 0
    age_diff = abs(row['Age'] - preferred_age)
    
    # 1. Location Check (Strict if not flexible)
    if not criteria['location_flexible']:
        if row['Location'] != criteria['location']:
            return 0 # Wrong location -> Irrelevant immediately
            
    # 2. Job Check
    is_exact_job = row['Job_Title'] in criteria['exact_jobs']
    is_related_job = row['Job_Title'] in criteria['jobs'] # Note: exact_jobs are also in jobs list usually
    
    if not is_related_job:
        return 0 # Not even related -> Irrelevant
        
    # 3. Tier Assignment
    
    # Tier 1: Perfect Match (Score 3)
    # - Exact Job Role (e.g. Surgeon for Surgeon)
    # - Perfect Age (Gap <= 3)
    # - Location is already checked above
    if is_exact_job and age_diff <= 3:
        return 3
        
    # Tier 2: Good Match (Score 2)
    # Case A: Exact Job, but Age Gap slightly larger (upto 5)
    if is_exact_job and age_diff <= 5:
        return 2
        
    # Case B: Related Job (e.g. Physician for Surgeon), but Perfect Age (Gap <= 3)
    # (Only if it's a related job found in 'jobs' list)
    if is_related_job and age_diff <= 3:
        return 2
        
    # Tier 3: Fair Match (Score 1)
    # - Related Job
    # - Age Gap upto 5
    if is_related_job and age_diff <= 5:
        return 1
        
    return 0
    

ground_truth = []

print("="*80)
print("GENERATING REFINED GROUND TRUTH (Top 20 with Tiers)")
print("="*80)

for query in queries_config:
    print(f"\n📋 Query {query['scenario_id']}: {query['scenario_name']}")
    
    # Get IDs for this query from the generated pools
    # FIX: Don't restrict to id_mapping subset. Search the WHOLE dataset.
    # Because valid matches might exist outside the targeted pool.
    
    # Filter by Hard Constraints first (Gender + Religion)
    # This mimics what the "Search Engine" would see as the base pool
    query_profiles = df[
        (df['Gender'] == query['filters']['gender']) & 
        (df['Religion'] == query['filters']['religion'])
    ].copy()
    
    # Score each profile
    scored_profiles = []
    
    for idx, row in query_profiles.iterrows():
        relevance = calculate_relevance_tier(row, query['criteria'], query['filters']['preferred_age'])
        
        # Only keep Relevant profiles (Score > 0)
        if relevance > 0:
            scored_profiles.append({
                "relevance_score": relevance,
                "data": row
            })
            
    # Sort by Relevance (3 -> 2 -> 1), then by Age Gap ascending
    scored_profiles.sort(key=lambda x: (x['relevance_score'], -abs(x['data']['Age'] - query['filters']['preferred_age'])), reverse=True)
    
    # Keep Top 20
    top_20 = scored_profiles[:20]
    
    print(f"   Found {len(top_20)} relevant profiles for Ground Truth")
    
    top_matches = []
    for i, item in enumerate(top_20, 1):
        row = item['data']
        match = {
            "rank": i,
            "relevance_score": item['relevance_score'], # 3=Perfect, 2=Good, 1=Fair
            "user_id": int(row['User_ID']),
            "name": row['Name'],
            "gender": row['Gender'],
            "age": int(row['Age']),
            "religion": row['Religion'],
            "location": row['Location'],
            "education": row['Education'],
            "job_title": row['Job_Title']
        }
        top_matches.append(match)
        
    # Build scenario
    scenario = {
        "scenario_id": query['scenario_id'],
        "scenario_name": query['scenario_name'],
        "query": query['query'],
        "searcher": query['searcher'],
        "filters": query['filters'],
        "criteria": query['criteria'],
        "ground_truth_candidates": top_matches # Renamed from top_matches to be clear these are ALL candidates
    }
    
    ground_truth.append(scenario)

# Save
output_file = 'ground_truth_refined.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(ground_truth, f, indent=2, ensure_ascii=False)

print("\n" + "="*80)
print(f"✅ Refined Ground Truth saved: {output_file}")
print("="*80)
