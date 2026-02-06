"""
Fix Ground Truth - Update Criteria to be Realistic
Making criteria match real-world job-education relationships
"""

import json

# Load current ground truth
with open('ground_truth_refined.json', 'r', encoding='utf-8') as f:
    ground_truth = json.load(f)

print("="*80)
print("🔧 FIXING GROUND TRUTH CRITERIA")
print("="*80)

# Fix each scenario
for scenario in ground_truth:
    sid = scenario['scenario_id']
    
    if sid == 1:
        # Software/IT - Already correct
        print(f"\n✅ Scenario {sid}: Software/IT - No changes needed")
        
    elif sid == 2:
        # Medical Professional - Add MS (Surgery)
        print(f"\n🔧 Scenario {sid}: Medical Professional")
        print(f"   BEFORE: {scenario['criteria']['educations']}")
        scenario['criteria']['educations'] = ['MBBS', 'MD', 'MS (Surgery)']
        print(f"   AFTER:  {scenario['criteria']['educations']}")
        print(f"   ✅ Added 'MS (Surgery)' - Surgeons have specialized degree!")
        
    elif sid == 3:
        # Finance - Already correct
        print(f"\n✅ Scenario {sid}: Finance Professional - No changes needed")
        
    elif sid == 4:
        # Designer/PM - Add MBA and BFA
        print(f"\n🔧 Scenario {sid}: Designer/PM")
        print(f"   BEFORE: {scenario['criteria']['educations']}")
        scenario['criteria']['educations'] = ['B.Tech', 'B.E.', 'Bachelor of Design', 'MBA', 'BFA']
        print(f"   AFTER:  {scenario['criteria']['educations']}")
        print(f"   ✅ Added 'MBA' (for PMs) and 'BFA' (for Designers)")
        
    elif sid == 5:
        # Data Scientist - Add B.Tech (common entry level)
        print(f"\n🔧 Scenario {sid}: Data Scientist")
        print(f"   BEFORE: {scenario['criteria']['educations']}")
        scenario['criteria']['educations'] = ['M.Tech', 'MS in CS', 'B.Tech']
        print(f"   AFTER:  {scenario['criteria']['educations']}")
        print(f"   ✅ Added 'B.Tech' - Many Data Scientists start with B.Tech")

# Save updated ground truth
with open('ground_truth_refined.json', 'w', encoding='utf-8') as f:
    json.dump(ground_truth, f, indent=2, ensure_ascii=False)

print("\n" + "="*80)
print("✅ GROUND TRUTH UPDATED SUCCESSFULLY!")
print("="*80)

print("""
🎯 CHANGES MADE:

1. Medical Professional (Scenario 2):
   - Added MS (Surgery) for Surgeons
   
2. Designer/PM (Scenario 4):
   - Added MBA for Product Managers (industry standard)
   - Added BFA for UI/UX Designers (design background)
   
3. Data Scientist (Scenario 5):
   - Added B.Tech (entry-level Data Scientists)

These changes make the criteria REALISTIC for semantic search!
""")
