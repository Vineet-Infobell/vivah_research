import json

with open('approach_1_results.json', 'r') as f:
    results = json.load(f)

with open('ground_truth.json', 'r') as f:
    gt = json.load(f)

print("="*80)
print("PROBLEM ANALYSIS - WHY LOW ACCURACY?")
print("="*80)

for scenario in results['scenarios']:
    sid = scenario['scenario_id']
    print(f"\n{'='*80}")
    print(f"QUERY {sid}: {scenario['scenario_name']}")
    print('='*80)
    
    gt_ids = set(scenario['ground_truth_top_10'])
    pred_ids = set(scenario['predicted_top_10'])
    
    matched = gt_ids.intersection(pred_ids)
    missed = gt_ids - pred_ids
    wrong = pred_ids - gt_ids
    
    print(f"\n📊 Results:")
    print(f"   Accuracy: {scenario['performance']['accuracy']}%")
    print(f"   Matched: {len(matched)}/10")
    print(f"   Missed: {len(missed)}")
    print(f"   Wrong picks: {len(wrong)}")
    
    print(f"\n✅ MATCHED IDs: {sorted(matched)}")
    print(f"❌ MISSED IDs: {sorted(missed)}")
    
    # Analyze missed profiles
    print(f"\n🔍 WHY MISSED? (Ground Truth profiles not in top 10)")
    gt_scenario = gt[sid-1]
    for m in gt_scenario['top_matches']:
        if m['user_id'] in missed:
            print(f"   ID {m['user_id']}: {m['name']} - {m['job_title']}, {m['education']}, {m['location']}, Age {m['age']}")
    
    # Analyze wrong picks
    print(f"\n⚠️  WRONG PICKS: (Predicted but not in ground truth)")
    for profile in scenario['predicted_profiles']:
        if profile['user_id'] in wrong:
            print(f"   ID {profile['user_id']}: {profile['name']} - {profile['job_title']}, {profile['education']}, {profile['location']}, Age {profile['age']}")
            print(f"      Semantic: {profile['semantic_score']:.3f}, Age: {profile['age_score']:.2f}, Final: {profile['final_score']:.3f}")
    
    # Key issue
    print(f"\n💡 KEY ISSUE:")
    criteria = gt_scenario['criteria']
    if criteria['location'] and not criteria['location_flexible']:
        print(f"   ⚠️  Query requires location: {criteria['location']}")
        print(f"   ❌ But semantic search doesn't enforce location!")
        
        # Check if wrong picks have wrong location
        wrong_location_count = 0
        for profile in scenario['predicted_profiles']:
            if profile['user_id'] in wrong and profile['location'] != criteria['location']:
                wrong_location_count += 1
        
        if wrong_location_count > 0:
            print(f"   ❌ {wrong_location_count}/{len(wrong)} wrong picks have WRONG LOCATION!")
    
    print()

print("="*80)
print("OVERALL CONCLUSION")
print("="*80)

print("""
🎯 MAIN PROBLEM IDENTIFIED:

1. ❌ LOCATION NOT ENFORCED
   - Query specifies "lives in Bangalore/Mumbai/Hyderabad"
   - But semantic search only checks text similarity
   - Wrong location profiles rank high due to job/education match
   
2. ❌ AGE SCORING TOO WEAK
   - Age score component is low (0.6-1.0)
   - Semantic similarity dominates (0.70-0.78)
   - Age differences not penalized enough

3. ❌ NO HARD FILTERS
   - Gender, Religion, Location should be HARD filters
   - Currently only semantic + age soft scoring

SOLUTIONS:
✅ Add hard filters (gender, religion, age range, location)
✅ Increase age score weight
✅ Add location exact match requirement for location-specific queries
""")
