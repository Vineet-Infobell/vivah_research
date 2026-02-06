"""
Reorder Ground Truth Profiles - Best to Worst
For better testing and analysis
"""

import json

with open('ground_truth.json', 'r') as f:
    gt = json.load(f)

print("="*80)
print("REORDERING GROUND TRUTH PROFILES (BEST → WORST)")
print("="*80)

for scenario in gt:
    sid = scenario['scenario_id']
    print(f"\n📋 Query {sid}: {scenario['scenario_name']}")
    
    criteria = scenario['criteria']
    filters = scenario['filters']
    
    # Define scoring function
    def score_profile(profile):
        score = 0
        
        # 1. Job Match (Most Important)
        job_title = profile['job_title']
        target_jobs = criteria['jobs']
        
        # Exact match
        if job_title in target_jobs:
            score += 1000
        # Partial/fuzzy match
        elif any(tj.lower() in job_title.lower() or job_title.lower() in tj.lower() for tj in target_jobs):
            score += 500
        
        # 2. Education Match
        education = profile['education']
        target_edu = criteria['educations']
        
        if education in target_edu:
            score += 100
        
        # 3. Location Match (if specified)
        if criteria.get('location') and not criteria.get('location_flexible'):
            if profile['location'] == criteria['location']:
                score += 50
            else:
                score -= 50  # Penalty for wrong location
        
        # 4. Age Proximity (if age filter specified)
        if 'age_min' in filters and 'age_max' in filters:
            age = profile['age']
            age_min = filters['age_min']
            age_max = filters['age_max']
            age_mid = (age_min + age_max) / 2
            
            # Prefer middle of range
            age_diff = abs(age - age_mid)
            score += (10 - age_diff) if age_diff < 10 else 0
        elif 'preferred_age' in filters:
            age = profile['age']
            pref_age = filters['preferred_age']
            age_diff = abs(age - pref_age)
            score += (10 - age_diff) if age_diff < 10 else 0
        
        return score
    
    # Sort profiles by score
    profiles = scenario['top_matches']
    scored = [(p, score_profile(p)) for p in profiles]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Update with sorted profiles and new ranks
    sorted_profiles = []
    for rank, (profile, score) in enumerate(scored, 1):
        profile['rank'] = rank
        sorted_profiles.append(profile)
        
        # Show reasoning
        symbol = "🥇" if rank <= 3 else "✅" if rank <= 7 else "⚠️"
        print(f"   {symbol} Rank {rank}: ID {profile['user_id']} - {profile['name']} ({profile['job_title']}, {profile['education']}, {profile['location']}, Age {profile['age']}) [Score: {score}]")
    
    scenario['top_matches'] = sorted_profiles

# Save reordered ground truth
output_file = 'ground_truth.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(gt, f, indent=2, ensure_ascii=False)

print("\n" + "="*80)
print(f"✅ Ground Truth reordered and saved: {output_file}")
print("="*80)

print("""
📊 Ordering Logic:
   1️⃣  Job Match (exact > partial) - Most Important
   2️⃣  Education Match
   3️⃣  Location Match (for location-specific queries)
   4️⃣  Age Proximity to preferred/range
   
   🥇 Rank 1-3: Best matches
   ✅ Rank 4-7: Good matches
   ⚠️  Rank 8-10: Acceptable matches
""")
