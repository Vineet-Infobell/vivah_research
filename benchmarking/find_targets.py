
import pandas as pd

# Load processed data
df = pd.read_csv(r"F:\Vivahai\Vivahai_backend_v2\research\benchmarking\matchmaking_fixed.csv")

print(f"Total Profiles: {len(df)}")

# Helper to print top 10 matches for a filter
def find_and_print(title, job_query, loc_query, edu_query=None):
    print(f"\n--- {title} ---")
    mask = (df['Job_Title'].str.contains(job_query, case=False, na=False) | 
            df['Job_Title'].str.contains('SDE', case=False, na=False) if job_query == 'Software' else False)
    
    mask = mask & df['Location'].str.contains(loc_query, case=False, na=False)
    
    if edu_query:
        mask = mask & df['Education'].str.contains(edu_query, case=False, na=False)
        
    matches = df[mask]
    
    if len(matches) > 0:
        print(matches[['Name', 'Job_Title', 'Education', 'Location']].head(10).to_string())
    else:
        print("No exact matches found. Trying looser constraints...")
        # Fallback: Just Job + Location
        mask = df['Job_Title'].str.contains(job_query, case=False, na=False) & \
               df['Location'].str.contains(loc_query, case=False, na=False)
        print(df[mask][['Name', 'Job_Title', 'Education', 'Location']].head(5).to_string())

# Scenario 1: IT in Ahmedabad
find_and_print("SCENARIO 1: IT in Ahmedabad", "Software|SDE|Developer", "Ahmedabad")

# Scenario 2: Medical in Ahmedabad
find_and_print("SCENARIO 2: Doctor in Ahmedabad", "Doctor|Surgeon|Dentist|Radiologist", "Ahmedabad")

# Scenario 3: Creative in Kolkata
find_and_print("SCENARIO 3: Creative in Kolkata", "Designer|Architect", "Kolkata")

# Scenario 4: Finance in Delhi/Noida
find_and_print("SCENARIO 4: Finance in Delhi/NCR", "Analyst|Manager|CA", "Delhi|Noida|Gurgaon")
