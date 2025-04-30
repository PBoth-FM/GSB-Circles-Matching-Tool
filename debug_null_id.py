import streamlit as st
import pandas as pd

print("\n\n🔍🔍🔍 PHANTOM PARTICIPANT INSPECTION TOOL 🔍🔍🔍")
if 'results' in st.session_state and st.session_state.results is not None:
    results_df = st.session_state.results
    
    if 'proposed_NEW_circles_id' in results_df.columns and 'Encoded ID' in results_df.columns:
        valid_circle_mask = (results_df['proposed_NEW_circles_id'].notna()) & (results_df['proposed_NEW_circles_id'] != 'UNMATCHED')
        null_id_mask = results_df['Encoded ID'].isna()
        
        null_id_matched = results_df[valid_circle_mask & null_id_mask]
        
        if len(null_id_matched) > 0:
            print(f"FOUND {len(null_id_matched)} PHANTOM PARTICIPANTS (rows with valid circle assignment but null Encoded ID)")
            
            for idx, row in null_id_matched.iterrows():
                print(f"\nPHANTOM PARTICIPANT #{idx}:")
                
                # Print critical identifying information
                participant_id = row.get('participant_id', 'Unknown')
                circle_id = row.get('proposed_NEW_circles_id', 'Unknown')
                
                print(f"participant_id: {participant_id}")
                print(f"Circle assignment: {circle_id}")
                print(f"region: {row.get('region', 'Unknown')}")
                
                # Print the score information
                print(f"location_score: {row.get('location_score', 'Unknown')}")
                print(f"time_score: {row.get('time_score', 'Unknown')}")
                print(f"total_score: {row.get('total_score', 'Unknown')}")
                
                # Print all non-null, non-unnamed columns
                print("\nAll attributes with values:")
                for col, val in row.items():
                    if pd.notna(val) and not col.startswith('Unnamed:'):
                        print(f"  {col}: {val}")
        else:
            print("No phantom participants found (all participants with circle assignments have valid Encoded IDs)")
    else:
        print("Required columns not found in results DataFrame")
else:
    print("No results found in session state")

print("🔍🔍🔍 END PHANTOM PARTICIPANT INSPECTION 🔍🔍🔍\n\n")