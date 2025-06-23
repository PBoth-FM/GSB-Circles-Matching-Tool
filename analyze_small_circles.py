#!/usr/bin/env python3
"""
Analyze small circles to see if they're getting new members
"""
import streamlit as st
import pandas as pd

def analyze_small_circles():
    """Analyze if small circles are actually getting new members"""
    
    if 'matched_circles' not in st.session_state:
        print("No matched_circles data found in session state")
        return
    
    if 'results' not in st.session_state:
        print("No results data found in session state")
        return
        
    circles_df = st.session_state['matched_circles']
    results_df = st.session_state['results']
    
    print("\n=== SMALL CIRCLES ANALYSIS ===")
    print(f"Total circles analyzed: {len(circles_df)}")
    
    # Filter for small circles (2-4 current members)
    small_circles = circles_df[
        (circles_df['member_count'] >= 2) & 
        (circles_df['member_count'] <= 4)
    ].copy()
    
    print(f"\nSmall circles (2-4 members): {len(small_circles)}")
    
    if len(small_circles) > 0:
        print("\nBreakdown by size:")
        size_counts = small_circles['member_count'].value_counts().sort_index()
        for size, count in size_counts.items():
            print(f"  {size} members: {count} circles")
        
        print(f"\nSmall circles with new members:")
        small_with_new = small_circles[small_circles['new_members'] > 0]
        print(f"  Count: {len(small_with_new)}")
        
        if len(small_with_new) > 0:
            print(f"\nDetailed breakdown:")
            for _, circle in small_with_new.iterrows():
                print(f"  Circle {circle['circle_id']}: {circle['member_count']} total, {circle['new_members']} new")
        else:
            print("  ❌ NO small circles received new members!")
            
        # Check very small circles specifically (2-3 members)
        very_small = small_circles[small_circles['member_count'] <= 3]
        very_small_with_new = very_small[very_small['new_members'] > 0]
        
        print(f"\nVery small circles (2-3 members): {len(very_small)}")
        print(f"Very small circles with new members: {len(very_small_with_new)}")
        
        if len(very_small_with_new) == 0:
            print("  ❌ NO very small circles (800pt bonus) received new members!")
            
        # Check circles with exactly 4 members
        size_4_circles = small_circles[small_circles['member_count'] == 4]
        size_4_with_new = size_4_circles[size_4_circles['new_members'] > 0]
        
        print(f"\nSize-4 circles: {len(size_4_circles)}")
        print(f"Size-4 circles with new members: {len(size_4_with_new)}")
        
        if len(size_4_with_new) == 0:
            print("  ❌ NO size-4 circles (50pt bonus) received new members!")
    
    # Also check if there are capacity constraints preventing assignments
    print(f"\n=== CAPACITY ANALYSIS ===")
    if 'circle_eligibility_logs' in st.session_state:
        eligibility_logs = st.session_state['circle_eligibility_logs']
        if eligibility_logs:
            # Convert to DataFrame if it's a list of dicts
            if isinstance(eligibility_logs, list):
                logs_df = pd.DataFrame(eligibility_logs)
            else:
                logs_df = eligibility_logs
                
            # Check small circles in eligibility logs
            small_circle_ids = small_circles['circle_id'].tolist()
            small_logs = logs_df[logs_df['circle_id'].isin(small_circle_ids)]
            
            print(f"Small circles with capacity (max_additions > 0): {len(small_logs[small_logs['max_additions'] > 0])}")
            print(f"Small circles with no capacity (max_additions = 0): {len(small_logs[small_logs['max_additions'] == 0])}")
            
            # Show examples
            no_capacity = small_logs[small_logs['max_additions'] == 0]
            if len(no_capacity) > 0:
                print(f"\nExamples of small circles with NO capacity:")
                for _, log in no_capacity.head(5).iterrows():
                    print(f"  {log['circle_id']}: max_additions = {log['max_additions']}")
    
    print(f"\n=== SUMMARY ===")
    total_new_assigned = circles_df['new_members'].sum() if 'new_members' in circles_df.columns else 0
    print(f"Total new members assigned across all circles: {total_new_assigned}")
    
    small_new_assigned = small_circles['new_members'].sum() if len(small_circles) > 0 else 0
    print(f"New members assigned to small circles: {small_new_assigned}")
    
    if total_new_assigned > 0:
        percentage = (small_new_assigned / total_new_assigned) * 100
        print(f"Percentage of new assignments going to small circles: {percentage:.1f}%")
    else:
        print("No new member assignments found in the data")

if __name__ == "__main__":
    analyze_small_circles()