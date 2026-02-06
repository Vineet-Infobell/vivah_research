"""
Validate Ground Truth Against CSV Data
Checks if the profiles marked in ground truth actually match the criteria
"""

import pandas as pd
import json

# Load CSV
df = pd.read_csv('matchmaking_1000_clean.csv')
df.set_index('User_ID', inplace=True)

# Load Ground Truth
with open('ground_truth_refined.json', 'r', encoding='utf-8') as f:
    ground_truth = json.load(f)

print("="*100)
print("🔍 GROUND TRUTH VALIDATION - Checking if profiles match criteria")
print("="*100)

total_errors = 0
total_profiles_checked = 0

for scenario in ground_truth:
    print(f"\n{'='*100}")
    print(f"📋 SCENARIO {scenario['scenario_id']}: {scenario['scenario_name']}")
    print('='*100)
    
    criteria = scenario['criteria']
    filters = scenario['filters']
    
    print(f"\n🎯 CRITERIA:")
    print(f"   Jobs (Exact): {criteria.get('exact_jobs', criteria.get('jobs'))}")
    print(f"   Jobs (Any): {criteria.get('jobs')}")
    print(f"   Education: {criteria.get('educations')}")
    print(f"   Location: {criteria.get('location')} (Flexible: {criteria.get('location_flexible')})")
    print(f"   Gender: {filters['gender']}")
    print(f"   Religion: {filters['religion']}")
    print(f"   Preferred Age: {filters.get('preferred_age', 'N/A')}")
    
    print(f"\n🔍 VALIDATING TOP 20 PROFILES:")
    print(f"{'Rank':<6} {'ID':<8} {'Name':<15} {'Age':<5} {'Job':<30} {'Edu':<15} {'Loc':<15} {'Status'}")
    print("-"*100)
    
    errors = []
    
    for candidate in scenario['ground_truth_candidates'][:20]:
        total_profiles_checked += 1
        user_id = candidate['user_id']
        rank = candidate['rank']
        rel_score = candidate['relevance_score']
        
        # Check if ID exists in CSV
        if user_id not in df.index:
            error_msg = f"❌ ID {user_id} NOT FOUND IN CSV!"
            errors.append(error_msg)
            print(f"{rank:<6} {user_id:<8} {'N/A':<15} {'N/A':<5} {'NOT IN CSV':<30} {'N/A':<15} {'N/A':<15} ❌ MISSING")
            total_errors += 1
            continue
        
        # Get actual profile from CSV
        csv_profile = df.loc[user_id]
        
        # Compare each field
        issues = []
        
        # Gender check
        if csv_profile['Gender'] != candidate['gender']:
            issues.append(f"Gender: CSV={csv_profile['Gender']} vs GT={candidate['gender']}")
        
        # Religion check
        if csv_profile['Religion'] != candidate['religion']:
            issues.append(f"Religion: CSV={csv_profile['Religion']} vs GT={candidate['religion']}")
        
        # Age check
        if int(csv_profile['Age']) != candidate['age']:
            issues.append(f"Age: CSV={csv_profile['Age']} vs GT={candidate['age']}")
        
        # Location check
        if csv_profile['Location'] != candidate['location']:
            issues.append(f"Loc: CSV={csv_profile['Location']} vs GT={candidate['location']}")
        
        # Education check
        if csv_profile['Education'] != candidate['education']:
            issues.append(f"Edu: CSV={csv_profile['Education']} vs GT={candidate['education']}")
        
        # Job Title check
        if csv_profile['Job_Title'] != candidate['job_title']:
            issues.append(f"Job: CSV={csv_profile['Job_Title']} vs GT={candidate['job_title']}")
        
        # Check against filters
        if csv_profile['Gender'] != filters['gender']:
            issues.append(f"Wrong Gender for filter!")
        
        if csv_profile['Religion'] != filters['religion']:
            issues.append(f"Wrong Religion for filter!")
        
        # Check against criteria
        if criteria.get('location') and not criteria.get('location_flexible', False):
            if csv_profile['Location'] != criteria['location']:
                issues.append(f"Location mismatch! Need {criteria['location']}")
        
        if criteria.get('jobs'):
            if csv_profile['Job_Title'] not in criteria['jobs']:
                issues.append(f"Job not in criteria list!")
        
        if criteria.get('educations'):
            if csv_profile['Education'] not in criteria['educations']:
                issues.append(f"Education not in criteria list!")
        
        # Print result
        status = "✅" if len(issues) == 0 else "⚠️"
        if len(issues) > 0:
            total_errors += len(issues)
        
        print(f"{rank:<6} {user_id:<8} {csv_profile['Name']:<15} {csv_profile['Age']:<5} {csv_profile['Job_Title']:<30} {csv_profile['Education']:<15} {csv_profile['Location']:<15} {status}")
        
        if issues:
            for issue in issues:
                print(f"       → {issue}")
                errors.append(f"Rank {rank} (ID {user_id}): {issue}")
    
    # Summary for this scenario
    print(f"\n📊 SCENARIO SUMMARY:")
    if len(errors) == 0:
        print(f"   ✅ All 20 profiles are CORRECT!")
    else:
        print(f"   ⚠️  Found {len(errors)} issues")
        

print(f"\n{'='*100}")
print(f"🎯 OVERALL VALIDATION SUMMARY")
print(f"{'='*100}")
print(f"Total Profiles Checked: {total_profiles_checked}")
print(f"Total Issues Found: {total_errors}")

if total_errors == 0:
    print(f"\n✅ ✅ ✅ GROUND TRUTH IS 100% CORRECT! ✅ ✅ ✅")
else:
    print(f"\n⚠️  Ground Truth has {total_errors} issues that need fixing")

print("\n" + "="*100)
