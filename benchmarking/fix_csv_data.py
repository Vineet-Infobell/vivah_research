
import pandas as pd
import random

# Path to original CSV
input_csv = r"F:\Vivahai\Vivahai_backend_v2\matchmaking_search_lab\matchmaking_structured_preferences.csv"
output_csv = r"F:\Vivahai\Vivahai_backend_v2\research\benchmarking\matchmaking_fixed.csv"

# Job -> Realistic Education Mapping
JOB_EDU_MAP = {
    'SDE': ['B.Tech Computer Science', 'M.Tech', 'MCA', 'B.E. CSE'],
    'Software Engineer': ['B.Tech', 'B.E.', 'MCA'],
    'Software Developer': ['B.Tech', 'BCA', 'BS Computer Science'],
    'Cloud Architect': ['B.Tech', 'M.Tech Cloud Computing'],
    'Data Scientist': ['M.S. Data Science', 'M.Tech', 'PhD Stats'],
    'QA Tester': ['B.Tech', 'BCA'],
    'Business Analyst': ['MBA', 'BBA', 'B.Com'],
    
    'Doctor': ['MBBS', 'MD', 'MS'],
    'Surgeon': ['MS (Surgery)', 'DNB General Surgery'],
    'Dentist': ['BDS', 'MDS'],
    'Physiotherapist': ['BPT', 'MPT'],
    'Radiologist': ['MD Radiology', 'DNB Radiology'],
    'Medical Officer': ['MBBS'],
    'Nurse': ['B.Sc Nursing', 'GNM'],
    
    'Architect': ['B.Arch', 'M.Arch'],
    'Interior Designer': ['B.Des Interior', 'Diploma Interior Design'],
    'Graphic Designer': ['B.Des', 'BFA', 'Diploma Graphics'],
    'Fashion Designer': ['B.Des Fashion', 'NIFT Graduate'],
    
    'Lawyer': ['LLB', 'LLM', 'B.A. LLB'],
    'Advocate': ['LLB', 'LLM'],
    'Judge': ['LLM', 'PhD Law'],
    'Legal Consultant': ['LLM', 'LLB'],
    
    'Chartered Accountant': ['CA', 'B.Com'],
    'Finance Manager': ['MBA Finance', 'CA', 'CFA'],
    'HR Specialist': ['MBA HR', 'PGDM HR'],
    'Marketing Manager': ['MBA Marketing', 'BBA'],
    
    'Professor': ['PhD', 'M.Phil', 'M.Ed'],
    'School Teacher': ['B.Ed', 'M.A.', 'M.Sc'],
    'Journalist': ['B.A. Journalism', 'Mass Communication'],
    'Photographer': ['Diploma Photography', 'BFA']
}

def fix_data():
    try:
        print(f"📂 Reading: {input_csv}")
        df = pd.read_csv(input_csv)
        
        fixed_count = 0
        
        for idx, row in df.iterrows():
            job = row['Job_Title']
            
            # Find matching education list based on job (partial match logic)
            educations = []
            for key, val in JOB_EDU_MAP.items():
                if key.lower() in str(job).lower():
                    educations = val
                    break
            
            # If we found realistic educations for this job, pick one randomly
            if educations:
                new_edu = random.choice(educations)
                # Update DataFrame
                df.at[idx, 'Education'] = new_edu
                fixed_count += 1
        
        # Save fixed CSV
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Fixed {fixed_count} profiles!")
        print(f"💾 Saved to: {output_csv}")
        
        # Show comparison sample
        print("\n--- Before vs After (Sample) ---")
        print(df[['Name', 'Job_Title', 'Education', 'Location']].head(10).to_string())
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    fix_data()
