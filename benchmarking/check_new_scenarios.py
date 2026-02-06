
import pandas as pd

df = pd.read_csv(r"F:\Vivahai\Vivahai_backend_v2\research\benchmarking\matchmaking_fixed.csv")

def find_matches(label, query_mask):
    matches = df[query_mask]
    print(f"\n--- {label} (Count: {len(matches)}) ---")
    if len(matches) > 0:
        print(matches[['Name', 'Job_Title', 'Education', 'Location']].head(10).to_string())

# 1. IT in Bangalore
it_mask = df['Job_Title'].str.contains('Software|SDE|Developer|Engineer|Architect', case=False)
bang_mask = df['Location'].str.contains('Bangalore', case=False)
find_matches("IT in Bangalore", it_mask & bang_mask)

# 2. MBA Graduates (Anywhere)
mba_mask = df['Education'].str.contains('MBA', case=False)
find_matches("MBA Graduates", mba_mask)

# 3. Kolkata Profiles (Any Job)
cal_mask = df['Location'].str.contains('Kolkata', case=False)
find_matches("Any Kolkata", cal_mask)
