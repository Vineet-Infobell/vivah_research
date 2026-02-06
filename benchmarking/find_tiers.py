import pandas as pd

# Load Data
df = pd.read_csv('matchmaking_fixed.csv')

def get_ids(rules):
    # Tier 1: All rules match
    t1_mask = pd.Series([True] * len(df))
    for col, vals in rules.items():
        if not vals: continue
        # Multi-value OR check (e.g. SDE or Software Engineer)
        t1_mask &= df[col].astype(str).apply(lambda x: any(v.lower() in x.lower() for v in vals))
    
    t1_ids = df[t1_mask]['User_ID'].tolist()
    
    # Tier 2: At least 1 rule matches (but not Tier 1)
    # Actually user said "Partial Match". Let's say if 2 criteria, 1 matches. If 3, 2 match.
    t2_ids = []
    
    # Calculate score for every row
    for idx, row in df.iterrows():
        if row['User_ID'] in t1_ids: continue
        
        score = 0
        total_criteria = 0
        for col, vals in rules.items():
            if not vals: continue
            total_criteria += 1
            if any(v.lower() in str(row[col]).lower() for v in vals):
                score += 1
        
        # If partial match (Score >= Total - 1 for Strict, or Score > 0 for Lenient?)
        # User said "2 ya ek match ho".
        # Let's simple say: Score >= 1
        if score > 0:
            t2_ids.append(row['User_ID'])
            
    return t1_ids, t2_ids[:5] # Limit Tier 2 to top 5 to avoid noise

# Scenario 1: SDE (Prof + Edu)
s1_rules = {
    'Job_Title': ['SDE', 'Software Engineer'],
    'Education': ['B.Tech', 'B.E.']
}
t1_s1, t2_s1 = get_ids(s1_rules)
print(f"S1 (SDE): Tier 1: {t1_s1}")
print(f"S1 (SDE): Tier 2: {t2_s1}")

# Scenario 2: Radiologist (Prof + Loc)
s2_rules = {
    'Job_Title': ['Radiologist'],
    'Location': ['Ahmedabad']
}
t1_s2, t2_s2 = get_ids(s2_rules)
print(f"S2 (Rad): Tier 1: {t1_s2}")
print(f"S2 (Rad): Tier 2: {t2_s2}")

# Scenario 3: CA (Edu + Loc)
s3_rules = {
    'Education': ['CA', 'Chartered Accountant'],
    'Location': ['Delhi']
}
t1_s3, t2_s3 = get_ids(s3_rules)
print(f"S3 (CA): Tier 1: {t1_s3}")
print(f"S3 (CA): Tier 2: {t2_s3}")

# Scenario 4: Business Analyst (Job + Edu + Loc)
s4_rules = {
    'Job_Title': ['Business Analyst'],
    'Education': ['MBA'],
    'Location': ['Noida']
}
t1_s4, t2_s4 = get_ids(s4_rules)
print(f"S4 (BA): Tier 1: {t1_s4}")
print(f"S4 (BA): Tier 2: {t2_s4}")
