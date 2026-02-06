import pandas as pd
df = pd.read_csv('matchmaking_fixed.csv')

print("--- Searching for Ritesh ---")
r_matches = df[df['Name'].str.contains("Ritesh", case=False, na=False)]
print(r_matches[['User_ID', 'Name', 'Job_Title', 'Location']].to_string())

print("\n--- Searching for Diya ---")
d_matches = df[df['Name'].str.contains("Diya", case=False, na=False)]
print(d_matches[['User_ID', 'Name', 'Job_Title', 'Location']].to_string())
