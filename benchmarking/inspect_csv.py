
import pandas as pd
from pathlib import Path

# Load CSV
csv_path = r"F:\Vivahai\Vivahai_backend_v2\matchmaking_search_lab\matchmaking_structured_preferences.csv"
try:
    df = pd.read_csv(csv_path)
    
    # Create combined profile text
    df['combined_text'] = df['Job_Title'] + ", " + df['Education'] + ", " + df['Location']
    
    print(f"Total Profiles: {len(df)}")
    print("\nColumns:", df.columns.tolist())
    
    print("\n--- Sample Profiles (for Scenario Creation) ---")
    # Show varied profiles to help pick scenarios (IT, Medical, locations)
    print(df[['Name', 'Job_Title', 'Education', 'Location']].head(15).to_string())

except Exception as e:
    print(f"Error reading CSV: {e}")
