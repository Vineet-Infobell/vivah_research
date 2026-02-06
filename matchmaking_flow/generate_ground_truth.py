"""
Generate Ground Truth JSON from Clean Dataset
Uses ground_truth_id_mapping.json to create proper ground truth
"""

import pandas as pd
import json

# Load dataset
df = pd.read_csv('matchmaking_1000_clean.csv')

# Load ID mapping
with open('ground_truth_id_mapping.json', 'r') as f:
    id_mapping = json.load(f)

# Define queries with metadata
queries_config = [
    {
        "scenario_id": 1,
        "scenario_name": "Software/IT Professional in Bangalore",
        "query": "Looking for Software Developer or SDE with B.Tech and lives in Bangalore",
        "searcher": {"gender": "Female", "age": 26, "religion": "Hindu"},
        "filters": {"gender": "Male", "religion": "Hindu", "age_filter": "flexible", "preferred_age": 27},
        "criteria": {
            "jobs": ["SDE", "Software Developer", "Backend Developer", "Software Engineer"],
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
        "filters": {"gender": "Male", "religion": "Hindu", "age_filter": "range", "age_min": 28, "age_max": 32},
        "criteria": {
            "jobs": ["Doctor", "Surgeon", "Physician", "Medical Officer"],
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
        "filters": {"gender": "Male", "religion": "Hindu", "age_filter": "range", "age_min": 25, "age_max": 32},
        "criteria": {
            "jobs": ["Chartered Accountant", "Finance Manager", "Financial Analyst", "Auditor"],
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
        "filters": {"gender": "Female", "religion": "Hindu", "age_filter": "range", "age_min": 26, "age_max": 30},
        "criteria": {
            "jobs": ["UI/UX Designer", "Product Manager", "Product Designer"],
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
        "filters": {"gender": "Female", "religion": "Hindu", "age_filter": "range", "age_min": 24, "age_max": 28},
        "criteria": {
            "jobs": ["Data Scientist", "ML Engineer", "AI Engineer"],
            "educations": ["M.Tech", "MS in CS"],
            "location": "Hyderabad",
            "location_flexible": False
        },
        "id_key": "query_5_ids"
    }
]

ground_truth = []

print("="*80)
print("GENERATING GROUND TRUTH JSON")
print("="*80)

for query in queries_config:
    print(f"\n📋 Query {query['scenario_id']}: {query['scenario_name']}")
    
    # Get IDs for this query
    profile_ids = id_mapping[query['id_key']]
    
    # Get profiles from dataset
    query_profiles = df[df['User_ID'].isin(profile_ids)].copy()
    
    # Take top 10 (should be all 20, but we want best 10)
    query_profiles = query_profiles.head(10)
    
    print(f"   Found {len(query_profiles)} profiles")
    
    # Build top_matches
    top_matches = []
    for i, (idx, row) in enumerate(query_profiles.iterrows(), 1):
        match = {
            "rank": i,
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
        "top_matches": top_matches
    }
    
    ground_truth.append(scenario)
    
    print(f"   ✅ Top {len(top_matches)} profiles selected")
    print(f"      IDs: {[m['user_id'] for m in top_matches]}")
    
    # Validate gender
    genders = set(m['gender'] for m in top_matches)
    expected_gender = query['filters']['gender']
    if len(genders) == 1 and list(genders)[0] == expected_gender:
        print(f"      ✅ Gender: All {expected_gender}")
    else:
        print(f"      ❌ Gender mismatch! Expected: {expected_gender}, Got: {genders}")

# Save
output_file = 'ground_truth.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(ground_truth, f, indent=2, ensure_ascii=False)

print("\n" + "="*80)
print(f"✅ Ground Truth saved: {output_file}")
print("="*80)

print(f"\n📊 Summary:")
print(f"   Total scenarios: {len(ground_truth)}")
print(f"   Profiles per scenario: 10")
print(f"   Total ground truth profiles: {sum(len(s['top_matches']) for s in ground_truth)}")

print(f"\n✅ Ground Truth Generation Complete!")
