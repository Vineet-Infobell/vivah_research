"""
Create Clean 1000 Profile Dataset
100 Targeted (for ground truth) + 900 General (realistic mix)
"""

import pandas as pd
import random
import json

# ========================
# DATA POOLS
# ========================

MALE_NAMES = ["Rahul", "Arjun", "Karan", "Rohan", "Vikram", "Aditya", "Amit", "Sanjay", "Rajesh", "Varun",
              "Nikhil", "Vivek", "Harsh", "Mayank", "Yash", "Sahil", "Ankit", "Deepak", "Ravi", "Mohit",
              "Suresh", "Ramesh", "Krishna", "Akash", "Abhishek", "Siddharth", "Pranav", "Kartik", "Ayaan", "Ishaan"]

FEMALE_NAMES = ["Priya", "Anjali", "Neha", "Pooja", "Riya", "Sneha", "Kavya", "Divya", "Simran", "Aarti",
                "Meera", "Tanya", "Shruti", "Ishita", "Sakshi", "Nisha", "Preeti", "Swati", "Komal", "Megha",
                "Ananya", "Diya", "Sara", "Kiara", "Aaradhya", "Vanshika", "Saanvi", "Myra", "Tara", "Pari"]

RELIGIONS = (
    ["Hindu"] * 800 +  # 80%
    ["Muslim"] * 100 +  # 10%
    ["Sikh"] * 50 +     # 5%
    ["Christian"] * 40 + # 4%
    ["Jain"] * 10       # 1%
)

CITIES = ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad", "Gurgaon", "Noida"]

# Realistic Job-Education Pairs
JOB_EDU_PAIRS = {
    "SDE": ["B.Tech", "B.E.", "MCA", "M.Tech"],
    "Software Developer": ["B.Tech", "B.E.", "MCA"],
    "Backend Developer": ["B.Tech", "B.E.", "M.Tech"],
    "Frontend Developer": ["B.Tech", "BCA", "MCA"],
    "Full Stack Developer": ["B.Tech", "B.E.", "M.Tech"],
    "DevOps Engineer": ["B.Tech", "B.E.", "M.Tech"],
    "Cloud Engineer": ["B.Tech", "M.Tech", "MS in CS"],
    "Data Scientist": ["M.Tech", "MS in CS", "B.Tech"],
    "ML Engineer": ["M.Tech", "MS in CS", "PhD CS"],
    "AI Engineer": ["M.Tech", "MS in CS", "PhD CS"],
    
    "Doctor": ["MBBS", "MD"],
    "Surgeon": ["MBBS", "MD", "MS (Surgery)"],
    "Physician": ["MBBS", "MD"],
    "Cardiologist": ["MBBS", "MD"],
    "Pediatrician": ["MBBS", "MD"],
    
    "Chartered Accountant": ["CA"],
    "Finance Manager": ["CA", "MBA", "M.Com"],
    "Financial Analyst": ["MBA", "CA", "CFA"],
    "Auditor": ["CA", "M.Com"],
    "Investment Banker": ["MBA", "CA", "CFA"],
    
    "UI/UX Designer": ["B.Tech", "Bachelor of Design", "BFA"],
    "Product Manager": ["B.Tech", "MBA"],
    "Product Designer": ["Bachelor of Design", "BFA", "B.Tech"],
    "Graphic Designer": ["BFA", "Bachelor of Design"],
    
    "Business Analyst": ["MBA", "B.Com", "BBA"],
    "Marketing Manager": ["MBA", "BBA", "PGDM"],
    "HR Manager": ["MBA", "BBA", "PGDM"],
    "Civil Engineer": ["B.Tech", "B.E."],
    "Mechanical Engineer": ["B.Tech", "B.E."],
    "Lawyer": ["LLB", "LLM"],
    "Teacher": ["B.Ed", "M.Ed", "MA"],
    "Professor": ["PhD", "M.Tech", "MA"]
}

JOBS = list(JOB_EDU_PAIRS.keys())

def generate_profile(uid, gender=None):
    """Generate one realistic profile"""
    if gender is None:
        gender = "Male" if random.random() < 0.6 else "Female"
    
    name = random.choice(MALE_NAMES if gender == "Male" else FEMALE_NAMES)
    age = random.randint(21, 35)
    religion = random.choice(RELIGIONS)
    location = random.choice(CITIES)
    job_title = random.choice(JOBS)
    education = random.choice(JOB_EDU_PAIRS[job_title])
    
    # Make reasonable preference
    pref_gender = "Female" if gender == "Male" else "Male"
    pref_location = location if random.random() > 0.3 else random.choice(CITIES)
    
    preferences = f"Looking for {pref_location}"
    
    return {
        "User_ID": uid,
        "Name": name,
        "Gender": gender,
        "Age": age,
        "Religion": religion,
        "Location": location,
        "Education": education,
        "Job_Title": job_title,
        "preferences": preferences
    }

# ========================
# GENERATE 900 GENERAL PROFILES
# ========================
print("="*80)
print("GENERATING 900 GENERAL PROFILES")
print("="*80)

general_profiles = []
for i in range(1, 901):
    profile = generate_profile(i)
    general_profiles.append(profile)

df_general = pd.DataFrame(general_profiles)

print(f"\n✅ Generated {len(df_general)} general profiles")
print(f"   Gender: Male={len(df_general[df_general['Gender']=='Male'])}, Female={len(df_general[df_general['Gender']=='Female'])}")
print(f"   Religion: {dict(df_general['Religion'].value_counts())}")

# ========================
# GENERATE 100 TARGETED PROFILES
# ========================
print("\n" + "="*80)
print("GENERATING 100 TARGETED PROFILES (20 per query)")
print("="*80)

targeted_profiles = []
ground_truth = []
uid = 901

# Query 1: Software/IT in Bangalore (Female searches Male)
print("\nQuery 1: Software/IT Professional in Bangalore")
q1_jobs = {
    "exact": [("SDE", "B.Tech"), ("SDE", "B.E."), ("Software Developer", "B.Tech"), ("Software Developer", "B.E.")],
    "semantic": [("Backend Developer", "B.Tech"), ("Frontend Developer", "B.E."), ("Full Stack Developer", "M.Tech"), ("Software Developer", "MCA")],
    "related": [("DevOps Engineer", "B.Tech"), ("Cloud Engineer", "M.Tech")]
}

q1_profiles = []
q1_ids = []

# 8 exact
for i in range(4):
    for job, edu in q1_jobs["exact"]:
        profile = {
            "User_ID": uid,
            "Name": random.choice(MALE_NAMES),
            "Gender": "Male",
            "Age": random.randint(25, 32),
            "Religion": "Hindu",
            "Location": "Bangalore",
            "Education": edu,
            "Job_Title": job,
            "preferences": "Looking for Bangalore"
        }
        targeted_profiles.append(profile)
        q1_ids.append(uid)
        uid += 1
        if len(q1_ids) >= 8:
            break
    if len(q1_ids) >= 8:
        break

# 8 semantic
for i in range(4):
    for job, edu in q1_jobs["semantic"]:
        profile = {
            "User_ID": uid,
            "Name": random.choice(MALE_NAMES),
            "Gender": "Male",
            "Age": random.randint(25, 32),
            "Religion": "Hindu",
            "Location": "Bangalore",
            "Education": edu,
            "Job_Title": job,
            "preferences": "Looking for Bangalore"
        }
        targeted_profiles.append(profile)
        q1_ids.append(uid)
        uid += 1
        if len(q1_ids) >= 16:
            break
    if len(q1_ids) >= 16:
        break

# 4 related
for job, edu in q1_jobs["related"]:
    for _ in range(2):
        profile = {
            "User_ID": uid,
            "Name": random.choice(MALE_NAMES),
            "Gender": "Male",
            "Age": random.randint(25, 32),
            "Religion": "Hindu",
            "Location": "Bangalore",
            "Education": edu,
            "Job_Title": job,
            "preferences": "Looking for Bangalore"
        }
        targeted_profiles.append(profile)
        q1_ids.append(uid)
        uid += 1

print(f"✅ Query 1: {len(q1_ids)} profiles created")

# Similarly create for other 4 queries... (continuing in same pattern)
# For brevity, I'll create a compact version

queries_config = [
    {
        "id": 2,
        "name": "Medical Professional",
        "gender": "Male",
        "jobs": {
            "exact": [("Doctor", "MBBS"), ("Surgeon", "MBBS"), ("Doctor", "MD"), ("Surgeon", "MD")],
            "semantic": [("Physician", "MBBS"), ("Cardiologist", "MD"), ("Pediatrician", "MBBS"), ("Surgeon", "MS (Surgery)")],
            "related": [("Dentist", "BDS"), ("Pharmacist", "B.Pharm")]
        },
        "location": None,  # No specific location
        "ages": (26, 34)
    },
    {
        "id": 3,
        "name": "Finance Professional in Mumbai",
        "gender": "Male",
        "jobs": {
            "exact": [("Chartered Accountant", "CA"), ("Finance Manager", "CA"), ("Finance Manager", "MBA"), ("Chartered Accountant", "CA")],
            "semantic": [("Financial Analyst", "MBA"), ("Auditor", "CA"), ("Investment Banker", "CFA"), ("Finance Manager", "M.Com")],
            "related": [("Business Analyst", "MBA"), ("Accountant", "B.Com")]
        },
        "location": "Mumbai",
        "ages": (25, 32)
    },
    {
        "id": 4,
        "name": "Designer/PM in Bangalore",
        "gender": "Female",
        "jobs": {
            "exact": [("UI/UX Designer", "B.Tech"), ("Product Manager", "B.Tech"), ("UI/UX Designer", "Bachelor of Design"), ("Product Manager", "MBA")],
            "semantic": [("Product Designer", "Bachelor of Design"), ("UX Researcher", "B.Tech"), ("Product Manager", "B.Tech"), ("UI/UX Designer", "BFA")],
            "related": [("Graphic Designer", "BFA"), ("Web Designer", "B.Tech")]
        },
        "location": "Bangalore",
        "ages": (26, 30)
    },
    {
        "id": 5,
        "name": "Data Scientist in Hyderabad",
        "gender": "Female",
        "jobs": {
            "exact": [("Data Scientist", "M.Tech"), ("Data Scientist", "MS in CS"), ("Data Scientist", "M.Tech"), ("Data Scientist", "MS in CS")],
            "semantic": [("ML Engineer", "M.Tech"), ("AI Engineer", "MS in CS"), ("Data Scientist", "B.Tech"), ("ML Engineer", "MS in CS")],
            "related": [("Data Analyst", "M.Tech"), ("Research Scientist", "PhD CS")]
        },
        "location": "Hyderabad",
        "ages": (24, 28)
    }
]

all_query_ids = {"1": q1_ids}

for q_config in queries_config:
    qid = q_config["id"]
    print(f"\nQuery {qid}: {q_config['name']}")
    
    q_ids = []
    
    # 8 exact
    for job, edu in q_config["jobs"]["exact"]:
        for _ in range(2):
            age = random.randint(*q_config["ages"])
            loc = q_config["location"] if q_config["location"] else random.choice(CITIES)
            
            profile = {
                "User_ID": uid,
                "Name": random.choice(MALE_NAMES if q_config["gender"] == "Male" else FEMALE_NAMES),
                "Gender": q_config["gender"],
                "Age": age,
                "Religion": "Hindu",
                "Location": loc,
                "Education": edu,
                "Job_Title": job,
                "preferences": f"Looking for {loc}"
            }
            targeted_profiles.append(profile)
            q_ids.append(uid)
            uid += 1
    
    # 8 semantic
    for job, edu in q_config["jobs"]["semantic"]:
        for _ in range(2):
            age = random.randint(*q_config["ages"])
            loc = q_config["location"] if q_config["location"] else random.choice(CITIES)
            
            profile = {
                "User_ID": uid,
                "Name": random.choice(MALE_NAMES if q_config["gender"] == "Male" else FEMALE_NAMES),
                "Gender": q_config["gender"],
                "Age": age,
                "Religion": "Hindu",
                "Location": loc,
                "Education": edu,
                "Job_Title": job,
                "preferences": f"Looking for {loc}"
            }
            targeted_profiles.append(profile)
            q_ids.append(uid)
            uid += 1
    
    # 4 related
    for job, edu in q_config["jobs"]["related"]:
        for _ in range(2):
            age = random.randint(*q_config["ages"])
            loc = q_config["location"] if q_config["location"] else random.choice(CITIES)
            
            profile = {
                "User_ID": uid,
                "Name": random.choice(MALE_NAMES if q_config["gender"] == "Male" else FEMALE_NAMES),
                "Gender": q_config["gender"],
                "Age": age,
                "Religion": "Hindu",
                "Location": loc,
                "Education": edu,
                "Job_Title": job,
                "preferences": f"Looking for {loc}"
            }
            targeted_profiles.append(profile)
            q_ids.append(uid)
            uid += 1
    
    all_query_ids[str(qid)] = q_ids
    print(f"✅ Query {qid}: {len(q_ids)} profiles created")

df_targeted = pd.DataFrame(targeted_profiles)

print(f"\n✅ Generated {len(df_targeted)} targeted profiles")
print(f"   Gender: Male={len(df_targeted[df_targeted['Gender']=='Male'])}, Female={len(df_targeted[df_targeted['Gender']=='Female'])}")

# ========================
# COMBINE & SAVE
# ========================
print("\n" + "="*80)
print("COMBINING & SAVING")
print("="*80)

df_final = pd.concat([df_general, df_targeted], ignore_index=True)
df_final['User_ID'] = range(1, len(df_final) + 1)

output_csv = 'matchmaking_1000_clean.csv'
df_final.to_csv(output_csv, index=False)

print(f"\n✅ Dataset saved: {output_csv}")
print(f"📊 Total profiles: {len(df_final)}")
print(f"   General: 900")
print(f"   Targeted: 100")

print(f"\n👥 Gender Distribution:")
print(df_final['Gender'].value_counts())

print(f"\n🙏 Religion Distribution:")
print(df_final['Religion'].value_counts())

print(f"\n📍 Location Distribution (top 5):")
print(df_final['Location'].value_counts().head())

print(f"\n💼 Job Distribution (top 10):")
print(df_final['Job_Title'].value_counts().head(10))

# ========================
# SAVE GROUND TRUTH MAPPING
# ========================
gt_mapping = {
    "query_1_ids": all_query_ids["1"],
    "query_2_ids": all_query_ids["2"],
    "query_3_ids": all_query_ids["3"],
    "query_4_ids": all_query_ids["4"],
    "query_5_ids": all_query_ids["5"]
}

with open('ground_truth_id_mapping.json', 'w') as f:
    json.dump(gt_mapping, f, indent=2)

print(f"\n✅ Ground truth ID mapping saved!")
print(f"\n🎯 Dataset Generation Complete!")
