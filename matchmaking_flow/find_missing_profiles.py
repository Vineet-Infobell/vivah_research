"""
Find Missing Profiles - Check if CSV has better profiles than Ground Truth
Applies filters and criteria to find ALL matching profiles, then compares with GT
"""

import pandas as pd
import json

# Load CSV
df = pd.read_csv('matchmaking_1000_clean.csv')

# Load Ground Truth
with open('ground_truth_refined.json', 'r', encoding='utf-8') as f:
    ground_truth = json.load(f)

print("="*120)
print("🔍 FINDING MISSING PROFILES - Are there better matches in CSV?")
print("="*120)

def score_profile(profile, criteria, filters, preferred_age):
    """Score a profile based on how well it matches criteria"""
    score = 0
    details = []
    
    # Job Match (Most Important) - 50 points
    job = profile['Job_Title']
    exact_jobs = criteria.get('exact_jobs', criteria.get('jobs', []))
    all_jobs = criteria.get('jobs', [])
    
    if job in exact_jobs:
        score += 50
        details.append(f"✅ Exact Job Match ({job})")
    elif job in all_jobs:
        score += 35
        details.append(f"✓ Job Match ({job})")
    else:
        details.append(f"❌ Job Mismatch ({job})")
        return 0, details  # If job doesn't match, score = 0
    
    # Education Match - 30 points
    edu = profile['Education']
    if edu in criteria.get('educations', []):
        score += 30
        details.append(f"✅ Edu Match ({edu})")
    else:
        details.append(f"⚠️  Edu Mismatch ({edu})")
    
    # Location Match - 15 points (if location specified and not flexible)
    if criteria.get('location') and not criteria.get('location_flexible', False):
        if profile['Location'] == criteria['location']:
            score += 15
            details.append(f"✅ Location Match ({profile['Location']})")
        else:
            details.append(f"❌ Location Mismatch ({profile['Location']} vs {criteria['location']})")
            return 0, details  # If location required but doesn't match, score = 0
    
    # Age Proximity - 5 points (closer to preferred age = higher score)
    age_diff = abs(profile['Age'] - preferred_age)
    if age_diff == 0:
        age_score = 5
    elif age_diff <= 2:
        age_score = 4
    elif age_diff <= 4:
        age_score = 3
    elif age_diff <= 6:
        age_score = 2
    else:
        age_score = 1
    
    score += age_score
    details.append(f"Age: {profile['Age']} (diff: {age_diff}, score: {age_score})")
    
    return score, details

# Analyze each scenario
for scenario in ground_truth:
    print(f"\n{'='*120}")
    print(f"📋 SCENARIO {scenario['scenario_id']}: {scenario['scenario_name']}")
    print('='*120)
    
    criteria = scenario['criteria']
    filters = scenario['filters']
    preferred_age = filters.get('preferred_age', scenario['searcher']['age'])
    
    print(f"\n🎯 FILTERS & CRITERIA:")
    print(f"   Gender: {filters['gender']}, Religion: {filters['religion']}")
    print(f"   Preferred Age: {preferred_age}")
    print(f"   Jobs (Priority): {criteria.get('exact_jobs', criteria.get('jobs'))}")
    print(f"   Education: {criteria.get('educations')}")
    print(f"   Location: {criteria.get('location')} (Flexible: {criteria.get('location_flexible')})")
    
    # Apply hard filters to CSV
    filtered = df[
        (df['Gender'] == filters['gender']) &
        (df['Religion'] == filters['religion'])
    ].copy()
    
    print(f"\n📊 After Gender+Religion filters: {len(filtered)} profiles")
    
    # Score all filtered profiles
    scored_profiles = []
    for idx, row in filtered.iterrows():
        score, details = score_profile(row, criteria, filters, preferred_age)
        if score > 0:  # Only include profiles with non-zero score
            scored_profiles.append({
                'user_id': row['User_ID'],
                'name': row['Name'],
                'age': row['Age'],
                'job': row['Job_Title'],
                'edu': row['Education'],
                'loc': row['Location'],
                'score': score,
                'details': details
            })
    
    # Sort by score (descending), then by age proximity
    scored_profiles.sort(key=lambda x: (x['score'], -abs(x['age'] - preferred_age)), reverse=True)
    
    print(f"📊 After criteria matching: {len(scored_profiles)} valid profiles")
    
    # Get ground truth IDs
    gt_ids = [c['user_id'] for c in scenario['ground_truth_candidates'][:20]]
    gt_ids_set = set(gt_ids)
    
    # Compare top 20 from our scoring with ground truth
    our_top_20 = scored_profiles[:20]
    our_top_20_ids = set([p['user_id'] for p in our_top_20])
    
    # Find missing profiles (in our top 20 but not in GT)
    missing_in_gt = [p for p in our_top_20 if p['user_id'] not in gt_ids_set]
    
    # Find profiles in GT top 20 but not in our top 20
    extra_in_gt = [uid for uid in gt_ids if uid not in our_top_20_ids]
    
    print(f"\n🔍 COMPARISON:")
    print(f"   Ground Truth Top 20: {len(gt_ids)} profiles")
    print(f"   Our Scoring Top 20: {len(our_top_20)} profiles")
    print(f"   Overlap: {len(gt_ids_set.intersection(our_top_20_ids))} profiles")
    
    if len(missing_in_gt) > 0:
        print(f"\n⚠️  MISSING IN GROUND TRUTH (Should be in top 20):")
        print(f"{'Rank':<6} {'ID':<8} {'Score':<7} {'Name':<15} {'Age':<5} {'Job':<30} {'Edu':<15} {'Location'}")
        print("-"*120)
        for i, p in enumerate(missing_in_gt[:10], 1):  # Show top 10 missing
            rank = scored_profiles.index(p) + 1
            print(f"{rank:<6} {p['user_id']:<8} {p['score']:<7} {p['name']:<15} {p['age']:<5} {p['job']:<30} {p['edu']:<15} {p['loc']}")
    else:
        print(f"\n✅ No missing profiles - Ground Truth top 20 matches our scoring!")
    
    if len(extra_in_gt) > 0:
        print(f"\n⚠️  EXTRA IN GROUND TRUTH (Lower score, shouldn't be in top 20):")
        print(f"{'GT Rank':<9} {'ID':<8} {'Score':<7} {'Our Rank':<10} {'Name':<15} {'Age':<5} {'Job':<30}")
        print("-"*120)
        for uid in extra_in_gt[:10]:  # Show top 10 extras
            # Find this profile in our scoring
            profile_in_our = next((p for p in scored_profiles if p['user_id'] == uid), None)
            gt_rank = gt_ids.index(uid) + 1
            if profile_in_our:
                our_rank = scored_profiles.index(profile_in_our) + 1
                print(f"{gt_rank:<9} {uid:<8} {profile_in_our['score']:<7} {our_rank:<10} {profile_in_our['name']:<15} {profile_in_our['age']:<5} {profile_in_our['job']:<30}")
            else:
                print(f"{gt_rank:<9} {uid:<8} {'N/A':<7} {'Not found':<10}")
    
    # Show top 30 with scores for reference
    print(f"\n📊 TOP 30 PROFILES BY SCORE:")
    print(f"{'Rank':<6} {'ID':<8} {'Score':<7} {'GT?':<5} {'Name':<15} {'Age':<5} {'Job':<30} {'Edu':<15} {'Loc'}")
    print("-"*120)
    for i, p in enumerate(scored_profiles[:30], 1):
        in_gt = "✅" if p['user_id'] in gt_ids_set else "❌"
        print(f"{i:<6} {p['user_id']:<8} {p['score']:<7} {in_gt:<5} {p['name']:<15} {p['age']:<5} {p['job']:<30} {p['edu']:<15} {p['loc']}")

print(f"\n{'='*120}")
print(f"🎯 ANALYSIS COMPLETE")
print(f"{'='*120}")
