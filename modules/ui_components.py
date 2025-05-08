import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import random

# Function has been moved to a more comprehensive implementation at line ~4029
# This is to avoid duplication and ensure consistent behavior
def render_split_circle_summary(key_prefix=None):
    """
    Render the summary of split circles - redirects to the main implementation
    
    Args:
        key_prefix (str, optional): Prefix for unique Streamlit component keys to avoid duplicates
    """
    # Forward to the main implementation
    from modules.ui_components import render_split_circle_summary as main_render_split_circle_summary
    return main_render_split_circle_summary(key_prefix=key_prefix)

def calculate_total_diversity_score(matched_circles_df, results_df):
    """
    Calculate the total diversity score by summing all individual category diversity scores from the detailed tabs.
    This function gets the scores for each diversity category by using the same calculation
    methods as in the individual category tabs.
    
    The total diversity score is the simple sum of all five category scores:
    - Class Vintage diversity score
    - Employment diversity score
    - Industry diversity score
    - Racial Identity diversity score
    - Children diversity score
    
    Parameters:
    matched_circles_df (DataFrame): DataFrame containing circle data
    results_df (DataFrame): DataFrame containing participant data
    
    Returns:
    int: Total diversity score (sum of all category scores)
    """
    if matched_circles_df is None or results_df is None:
        return 0
    
    # Debug information
    original_circle_count = len(matched_circles_df) if hasattr(matched_circles_df, '__len__') else 0
    print(f"DEBUG - Match page diversity scores starting with {original_circle_count} circles")
    
    # Calculate the individual category scores using the same methods as in the detailed tabs
    vintage_score = calculate_vintage_diversity_score(matched_circles_df, results_df)
    employment_score = calculate_employment_diversity_score(matched_circles_df, results_df)
    industry_score = calculate_industry_diversity_score(matched_circles_df, results_df)  
    ri_score = calculate_racial_identity_diversity_score(matched_circles_df, results_df)
    children_score = calculate_children_diversity_score(matched_circles_df, results_df)
    
    # Log the individual scores for debugging
    total_score = vintage_score + employment_score + industry_score + ri_score + children_score
    
    # Print debug information to console
    print(f"DEBUG - Match page diversity scores: Vintage({vintage_score}) + Employment({employment_score}) + " +
          f"Industry({industry_score}) + RI({ri_score}) + Children({children_score}) = Total({total_score})")
    
    return total_score

def calculate_vintage_diversity_score(matched_circles_df, results_df):
    """Calculate the total diversity score for the Class Vintage category"""
    if matched_circles_df is None or results_df is None:
        return 0
    
    # Dictionary to track diversity scores
    circle_vintage_diversity_scores = {}
    original_circle_count = len(matched_circles_df) if hasattr(matched_circles_df, '__len__') else 0
    circles_processed = 0
    circles_with_no_data = 0
    circles_with_data = 0
    
    # Debug: Print available columns in results_df
    print(f"DEBUG - Results DataFrame columns: {results_df.columns.tolist()}")
    print(f"DEBUG - 'Encoded ID' exists in results_df: {'Encoded ID' in results_df.columns}")
    print(f"DEBUG - 'Class_Vintage' exists in results_df: {'Class_Vintage' in results_df.columns}")
    print(f"DEBUG - 'proposed_NEW_circles_id' exists in results_df: {'proposed_NEW_circles_id' in results_df.columns}")
    
    # Debug: Print a few sample values from results_df
    if 'Encoded ID' in results_df.columns and len(results_df) > 0:
        print(f"DEBUG - Sample 'Encoded ID' values: {results_df['Encoded ID'].head(3).tolist()}")
        print(f"DEBUG - Data types: {results_df['Encoded ID'].dtype}")
    
    # Store debugging for specific circles of interest
    selected_circles = ['IP-ATL-1', 'IP-BOS-01']
    circle_debug_info = {}
    
    # Process each circle to calculate diversity scores
    for _, circle_row in matched_circles_df.iterrows():
        circles_processed += 1
        
        # Skip circles with no members
        if 'member_count' not in circle_row or circle_row['member_count'] == 0:
            continue
        
        circle_id = circle_row['circle_id']
        
        # Initialize debug info for specific circles
        is_selected_circle = circle_id in selected_circles
        if is_selected_circle:
            circle_debug_info[circle_id] = {
                "members_raw": str(circle_row['members'])[:100] if 'members' in circle_row else "None",
                "member_count": circle_row['member_count'] if 'member_count' in circle_row else 0,
                "members_processed": [],
                "found_members": 0,
                "members_with_vintage": 0,
                "vintage_values": []
            }
        
        # IMPROVED APPROACH: Get the list of members for this circle
        # Method 1: Try to get members from the circle_row['members']
        members_from_row = []
        if 'members' in circle_row and circle_row['members'] and not pd.isna(circle_row['members']).all():
            # For list representation
            if isinstance(circle_row['members'], list):
                members_from_row = [m for m in circle_row['members'] if not pd.isna(m)]
                if is_selected_circle:
                    circle_debug_info[circle_id]["members_format"] = "list"
            # For string representation - convert to list
            elif isinstance(circle_row['members'], str):
                try:
                    if circle_row['members'].startswith('['):
                        members_from_row = [m for m in eval(circle_row['members']) if not pd.isna(m)]
                        if is_selected_circle:
                            circle_debug_info[circle_id]["members_format"] = "string_eval"
                    else:
                        members_from_row = [circle_row['members']]
                        if is_selected_circle:
                            circle_debug_info[circle_id]["members_format"] = "string_single"
                except Exception as e:
                    if is_selected_circle:
                        circle_debug_info[circle_id]["members_format"] = f"error: {str(e)}"
            else:
                if is_selected_circle:
                    circle_debug_info[circle_id]["members_format"] = f"unknown: {type(circle_row['members'])}"
        
        # Method 2: Get members by looking up the circle_id in the results dataframe's proposed_NEW_circles_id column
        members_from_lookup = []
        if 'proposed_NEW_circles_id' in results_df.columns:
            # Find all participants assigned to this circle
            circle_members = results_df[results_df['proposed_NEW_circles_id'] == circle_id]
            if not circle_members.empty and 'Encoded ID' in circle_members.columns:
                members_from_lookup = circle_members['Encoded ID'].dropna().tolist()
        
        # Combine both methods, prioritizing non-empty results
        if members_from_lookup:
            members = members_from_lookup
            if is_selected_circle:
                circle_debug_info[circle_id]["members_source"] = "lookup_in_results"
        else:
            members = members_from_row
            if is_selected_circle:
                circle_debug_info[circle_id]["members_source"] = "from_circle_row"
        
        if is_selected_circle:
            circle_debug_info[circle_id]["members_processed"] = members[:10]  # First 10 members for debugging
            circle_debug_info[circle_id]["members_count_method1"] = len(members_from_row)
            circle_debug_info[circle_id]["members_count_method2"] = len(members_from_lookup)
            circle_debug_info[circle_id]["final_members_count"] = len(members)
        
        # Initialize set to track unique categories
        unique_vintages = set()
        
        # For each member, look up their demographic data
        for member_id in members:
            # Skip NaN or invalid member IDs
            if pd.isna(member_id):
                continue
                
            # Debug for specific circles
            member_debug = {"id": str(member_id), "found": False, "has_vintage": False, "vintage_value": None}
            
            # Try exact match first
            member_data = results_df[results_df['Encoded ID'] == member_id]
            
            # If no match, try converting both to strings for comparison
            if member_data.empty:
                # Convert to string and try again
                member_data = results_df[results_df['Encoded ID'].astype(str) == str(member_id)]
                
                # If still no match and member_id has numeric format but might be int vs float
                if member_data.empty and str(member_id).replace('.', '', 1).isdigit():
                    try:
                        # Try as float
                        float_id = float(member_id)
                        member_data = results_df[results_df['Encoded ID'].astype(float) == float_id]
                        
                        # Try as int if it's a whole number
                        if member_data.empty and float_id.is_integer():
                            int_id = int(float_id)
                            member_data = results_df[results_df['Encoded ID'].astype(int) == int_id]
                    except:
                        pass
            
            if not member_data.empty:
                member_debug["found"] = True
                
                # Vintage diversity
                if 'Class_Vintage' in member_data.columns:
                    vintage = member_data['Class_Vintage'].iloc[0]
                    if pd.notna(vintage):
                        unique_vintages.add(vintage)
                        member_debug["has_vintage"] = True
                        member_debug["vintage_value"] = vintage
            
            # Add debug info for selected circles
            if is_selected_circle:
                circle_debug_info[circle_id]["found_members"] += 1 if member_debug["found"] else 0
                circle_debug_info[circle_id]["members_with_vintage"] += 1 if member_debug["has_vintage"] else 0
                if member_debug["has_vintage"]:
                    circle_debug_info[circle_id]["vintage_values"].append(member_debug["vintage_value"])
        
        # Calculate diversity score for this circle - include ALL circles, even those with no data
        if unique_vintages:
            score = len(unique_vintages)
            circle_vintage_diversity_scores[circle_id] = score
            circles_with_data += 1
        else:
            # Still include the circle but with a score of 0
            circle_vintage_diversity_scores[circle_id] = 0
            circles_with_no_data += 1
    
    # Print detailed debug info for selected circles
    for circle_id, debug_info in circle_debug_info.items():
        print(f"\nDETAILED DEBUG FOR {circle_id}:")
        print(f"  Raw members data: {debug_info['members_raw']}")
        print(f"  Member count: {debug_info['member_count']}")
        print(f"  Members format: {debug_info.get('members_format', 'unknown')}")
        print(f"  Members source: {debug_info.get('members_source', 'unknown')}")
        print(f"  Members count (method1): {debug_info.get('members_count_method1', 0)}")
        print(f"  Members count (method2): {debug_info.get('members_count_method2', 0)}")
        print(f"  Final members count: {debug_info.get('final_members_count', 0)}")
        print(f"  Processed members (sample): {debug_info['members_processed']}")
        print(f"  Found {debug_info['found_members']} members in results_df")
        print(f"  {debug_info['members_with_vintage']} members have vintage data")
        print(f"  Vintage values found: {debug_info['vintage_values']}")
        print(f"  Final score: {circle_vintage_diversity_scores.get(circle_id, 0)}")
    
    # Calculate total score across all circles
    total_score = sum(circle_vintage_diversity_scores.values())
    
    # Debug information
    print(f"DIVERSITY DEBUG - Vintage: {circles_processed} circles processed, {circles_with_data} with data, {circles_with_no_data} without data, total score: {total_score}")
    
    return total_score

def calculate_employment_diversity_score(matched_circles_df, results_df):
    """Calculate the total diversity score for the Employment category"""
    if matched_circles_df is None or results_df is None:
        return 0
    
    # Dictionary to track diversity scores
    circle_employment_diversity_scores = {}
    original_circle_count = len(matched_circles_df) if hasattr(matched_circles_df, '__len__') else 0
    circles_processed = 0
    circles_with_no_data = 0
    circles_with_data = 0
    
    # Process each circle to calculate diversity scores
    for _, circle_row in matched_circles_df.iterrows():
        circles_processed += 1
        
        # Skip circles with no members
        if 'member_count' not in circle_row or circle_row['member_count'] == 0:
            continue
        
        circle_id = circle_row['circle_id']
        
        # IMPROVED APPROACH: Get the list of members for this circle
        # Method 1: Try to get members from the circle_row['members']
        members_from_row = []
        if 'members' in circle_row and circle_row['members'] and not pd.isna(circle_row['members']).all():
            # For list representation
            if isinstance(circle_row['members'], list):
                members_from_row = [m for m in circle_row['members'] if not pd.isna(m)]
            # For string representation - convert to list
            elif isinstance(circle_row['members'], str):
                try:
                    if circle_row['members'].startswith('['):
                        members_from_row = [m for m in eval(circle_row['members']) if not pd.isna(m)]
                    else:
                        members_from_row = [circle_row['members']]
                except Exception:
                    pass
        
        # Method 2: Get members by looking up the circle_id in the results dataframe's proposed_NEW_circles_id column
        members_from_lookup = []
        if 'proposed_NEW_circles_id' in results_df.columns:
            # Find all participants assigned to this circle
            circle_members = results_df[results_df['proposed_NEW_circles_id'] == circle_id]
            if not circle_members.empty and 'Encoded ID' in circle_members.columns:
                members_from_lookup = circle_members['Encoded ID'].dropna().tolist()
        
        # Combine both methods, prioritizing non-empty results
        members = members_from_lookup if members_from_lookup else members_from_row
        
        # Initialize set to track unique categories
        unique_employment_categories = set()
        
        # For each member, look up their demographic data
        for member_id in members:
            # Skip NaN or invalid member IDs
            if pd.isna(member_id):
                continue
                
            # Try exact match first
            member_data = results_df[results_df['Encoded ID'] == member_id]
            
            # If no match, try converting both to strings for comparison
            if member_data.empty:
                # Convert to string and try again
                member_data = results_df[results_df['Encoded ID'].astype(str) == str(member_id)]
                
                # If still no match and member_id has numeric format but might be int vs float
                if member_data.empty and str(member_id).replace('.', '', 1).isdigit():
                    try:
                        # Try as float
                        float_id = float(member_id)
                        member_data = results_df[results_df['Encoded ID'].astype(float) == float_id]
                        
                        # Try as int if it's a whole number
                        if member_data.empty and float_id.is_integer():
                            int_id = int(float_id)
                            member_data = results_df[results_df['Encoded ID'].astype(int) == int_id]
                    except:
                        pass
            
            if not member_data.empty:
                # Employment diversity
                if 'Employment_Category' in member_data.columns:
                    employment = member_data['Employment_Category'].iloc[0]
                    if pd.notna(employment):
                        unique_employment_categories.add(employment)
        
        # Calculate diversity score for this circle - include ALL circles, even those with no data
        if unique_employment_categories:
            score = len(unique_employment_categories)
            circle_employment_diversity_scores[circle_id] = score
            circles_with_data += 1
        else:
            # Still include the circle but with a score of 0
            circle_employment_diversity_scores[circle_id] = 0
            circles_with_no_data += 1
    
    # Calculate total score across all circles
    total_score = sum(circle_employment_diversity_scores.values())
    
    # Debug information
    print(f"DIVERSITY DEBUG - Employment: {circles_processed} circles processed, {circles_with_data} with data, {circles_with_no_data} without data, total score: {total_score}")
    
    return total_score

def calculate_industry_diversity_score(matched_circles_df, results_df):
    """Calculate the total diversity score for the Industry category"""
    if matched_circles_df is None or results_df is None:
        return 0
    
    # Dictionary to track diversity scores
    circle_industry_diversity_scores = {}
    original_circle_count = len(matched_circles_df) if hasattr(matched_circles_df, '__len__') else 0
    circles_processed = 0
    circles_with_no_data = 0
    circles_with_data = 0
    
    # Process each circle to calculate diversity scores
    for _, circle_row in matched_circles_df.iterrows():
        circles_processed += 1
        
        # Skip circles with no members
        if 'member_count' not in circle_row or circle_row['member_count'] == 0:
            continue
        
        circle_id = circle_row['circle_id']
        
        # IMPROVED APPROACH: Get the list of members for this circle
        # Method 1: Try to get members from the circle_row['members']
        members_from_row = []
        if 'members' in circle_row and circle_row['members'] and not pd.isna(circle_row['members']).all():
            # For list representation
            if isinstance(circle_row['members'], list):
                members_from_row = [m for m in circle_row['members'] if not pd.isna(m)]
            # For string representation - convert to list
            elif isinstance(circle_row['members'], str):
                try:
                    if circle_row['members'].startswith('['):
                        members_from_row = [m for m in eval(circle_row['members']) if not pd.isna(m)]
                    else:
                        members_from_row = [circle_row['members']]
                except Exception:
                    pass
        
        # Method 2: Get members by looking up the circle_id in the results dataframe's proposed_NEW_circles_id column
        members_from_lookup = []
        if 'proposed_NEW_circles_id' in results_df.columns:
            # Find all participants assigned to this circle
            circle_members = results_df[results_df['proposed_NEW_circles_id'] == circle_id]
            if not circle_members.empty and 'Encoded ID' in circle_members.columns:
                members_from_lookup = circle_members['Encoded ID'].dropna().tolist()
        
        # Combine both methods, prioritizing non-empty results
        members = members_from_lookup if members_from_lookup else members_from_row
        
        # Initialize set to track unique categories
        unique_industry_categories = set()
        
        # For each member, look up their demographic data
        for member_id in members:
            # Skip NaN or invalid member IDs
            if pd.isna(member_id):
                continue
                
            # Try exact match first
            member_data = results_df[results_df['Encoded ID'] == member_id]
            
            # If no match, try converting both to strings for comparison
            if member_data.empty:
                # Convert to string and try again
                member_data = results_df[results_df['Encoded ID'].astype(str) == str(member_id)]
                
                # If still no match and member_id has numeric format but might be int vs float
                if member_data.empty and str(member_id).replace('.', '', 1).isdigit():
                    try:
                        # Try as float
                        float_id = float(member_id)
                        member_data = results_df[results_df['Encoded ID'].astype(float) == float_id]
                        
                        # Try as int if it's a whole number
                        if member_data.empty and float_id.is_integer():
                            int_id = int(float_id)
                            member_data = results_df[results_df['Encoded ID'].astype(int) == int_id]
                    except:
                        pass
            
            if not member_data.empty:
                # Industry diversity
                if 'Industry_Category' in member_data.columns:
                    industry = member_data['Industry_Category'].iloc[0]
                    if pd.notna(industry):
                        unique_industry_categories.add(industry)
        
        # Calculate diversity score for this circle - include ALL circles, even those with no data
        if unique_industry_categories:
            score = len(unique_industry_categories)
            circle_industry_diversity_scores[circle_id] = score
            circles_with_data += 1
        else:
            # Still include the circle but with a score of 0
            circle_industry_diversity_scores[circle_id] = 0
            circles_with_no_data += 1
    
    # Calculate total score across all circles
    total_score = sum(circle_industry_diversity_scores.values())
    
    # Debug information
    print(f"DIVERSITY DEBUG - Industry: {circles_processed} circles processed, {circles_with_data} with data, {circles_with_no_data} without data, total score: {total_score}")
    
    return total_score

def calculate_racial_identity_diversity_score(matched_circles_df, results_df):
    """Calculate the total diversity score for the Racial Identity category"""
    if matched_circles_df is None or results_df is None:
        return 0
    
    # Dictionary to track diversity scores
    circle_ri_diversity_scores = {}
    original_circle_count = len(matched_circles_df) if hasattr(matched_circles_df, '__len__') else 0
    circles_processed = 0
    circles_with_no_data = 0
    circles_with_data = 0
    
    # Process each circle to calculate diversity scores
    for _, circle_row in matched_circles_df.iterrows():
        circles_processed += 1
        
        # Skip circles with no members
        if 'member_count' not in circle_row or circle_row['member_count'] == 0:
            continue
        
        circle_id = circle_row['circle_id']
        
        # IMPROVED APPROACH: Get the list of members for this circle
        # Method 1: Try to get members from the circle_row['members']
        members_from_row = []
        if 'members' in circle_row and circle_row['members'] and not pd.isna(circle_row['members']).all():
            # For list representation
            if isinstance(circle_row['members'], list):
                members_from_row = [m for m in circle_row['members'] if not pd.isna(m)]
            # For string representation - convert to list
            elif isinstance(circle_row['members'], str):
                try:
                    if circle_row['members'].startswith('['):
                        members_from_row = [m for m in eval(circle_row['members']) if not pd.isna(m)]
                    else:
                        members_from_row = [circle_row['members']]
                except Exception:
                    pass
        
        # Method 2: Get members by looking up the circle_id in the results dataframe's proposed_NEW_circles_id column
        members_from_lookup = []
        if 'proposed_NEW_circles_id' in results_df.columns:
            # Find all participants assigned to this circle
            circle_members = results_df[results_df['proposed_NEW_circles_id'] == circle_id]
            if not circle_members.empty and 'Encoded ID' in circle_members.columns:
                members_from_lookup = circle_members['Encoded ID'].dropna().tolist()
        
        # Combine both methods, prioritizing non-empty results
        members = members_from_lookup if members_from_lookup else members_from_row
        
        # Initialize set to track unique categories
        unique_ri_categories = set()
        
        # For each member, look up their demographic data
        for member_id in members:
            # Skip NaN or invalid member IDs
            if pd.isna(member_id):
                continue
                
            # Try exact match first
            member_data = results_df[results_df['Encoded ID'] == member_id]
            
            # If no match, try converting both to strings for comparison
            if member_data.empty:
                # Convert to string and try again
                member_data = results_df[results_df['Encoded ID'].astype(str) == str(member_id)]
                
                # If still no match and member_id has numeric format but might be int vs float
                if member_data.empty and str(member_id).replace('.', '', 1).isdigit():
                    try:
                        # Try as float
                        float_id = float(member_id)
                        member_data = results_df[results_df['Encoded ID'].astype(float) == float_id]
                        
                        # Try as int if it's a whole number
                        if member_data.empty and float_id.is_integer():
                            int_id = int(float_id)
                            member_data = results_df[results_df['Encoded ID'].astype(int) == int_id]
                    except:
                        pass
            
            if not member_data.empty:
                # Racial Identity diversity
                if 'Racial_Identity_Category' in member_data.columns:
                    ri = member_data['Racial_Identity_Category'].iloc[0]
                    if pd.notna(ri):
                        unique_ri_categories.add(ri)
        
        # Calculate diversity score for this circle - include ALL circles, even those with no data
        if unique_ri_categories:
            score = len(unique_ri_categories)
            circle_ri_diversity_scores[circle_id] = score
            circles_with_data += 1
        else:
            # Still include the circle but with a score of 0
            circle_ri_diversity_scores[circle_id] = 0
            circles_with_no_data += 1
    
    # Calculate total score across all circles
    total_score = sum(circle_ri_diversity_scores.values())
    
    # Debug information
    print(f"DIVERSITY DEBUG - Racial Identity: {circles_processed} circles processed, {circles_with_data} with data, {circles_with_no_data} without data, total score: {total_score}")
    
    return total_score

def calculate_children_diversity_score(matched_circles_df, results_df):
    """Calculate the total diversity score for the Children category"""
    if matched_circles_df is None or results_df is None:
        return 0
    
    # Dictionary to track diversity scores
    circle_children_diversity_scores = {}
    original_circle_count = len(matched_circles_df) if hasattr(matched_circles_df, '__len__') else 0
    circles_processed = 0
    circles_with_no_data = 0
    circles_with_data = 0
    
    # Process each circle to calculate diversity scores
    for _, circle_row in matched_circles_df.iterrows():
        circles_processed += 1
        
        # Skip circles with no members
        if 'member_count' not in circle_row or circle_row['member_count'] == 0:
            continue
        
        circle_id = circle_row['circle_id']
        
        # IMPROVED APPROACH: Get the list of members for this circle
        # Method 1: Try to get members from the circle_row['members']
        members_from_row = []
        if 'members' in circle_row and circle_row['members'] and not pd.isna(circle_row['members']).all():
            # For list representation
            if isinstance(circle_row['members'], list):
                members_from_row = [m for m in circle_row['members'] if not pd.isna(m)]
            # For string representation - convert to list
            elif isinstance(circle_row['members'], str):
                try:
                    if circle_row['members'].startswith('['):
                        members_from_row = [m for m in eval(circle_row['members']) if not pd.isna(m)]
                    else:
                        members_from_row = [circle_row['members']]
                except Exception:
                    pass
        
        # Method 2: Get members by looking up the circle_id in the results dataframe's proposed_NEW_circles_id column
        members_from_lookup = []
        if 'proposed_NEW_circles_id' in results_df.columns:
            # Find all participants assigned to this circle
            circle_members = results_df[results_df['proposed_NEW_circles_id'] == circle_id]
            if not circle_members.empty and 'Encoded ID' in circle_members.columns:
                members_from_lookup = circle_members['Encoded ID'].dropna().tolist()
        
        # Combine both methods, prioritizing non-empty results
        members = members_from_lookup if members_from_lookup else members_from_row
        
        # Initialize set to track unique categories
        unique_children_categories = set()
        
        # For each member, look up their demographic data
        for member_id in members:
            # Skip NaN or invalid member IDs
            if pd.isna(member_id):
                continue
                
            # Try exact match first
            member_data = results_df[results_df['Encoded ID'] == member_id]
            
            # If no match, try converting both to strings for comparison
            if member_data.empty:
                # Convert to string and try again
                member_data = results_df[results_df['Encoded ID'].astype(str) == str(member_id)]
                
                # If still no match and member_id has numeric format but might be int vs float
                if member_data.empty and str(member_id).replace('.', '', 1).isdigit():
                    try:
                        # Try as float
                        float_id = float(member_id)
                        member_data = results_df[results_df['Encoded ID'].astype(float) == float_id]
                        
                        # Try as int if it's a whole number
                        if member_data.empty and float_id.is_integer():
                            int_id = int(float_id)
                            member_data = results_df[results_df['Encoded ID'].astype(int) == int_id]
                    except:
                        pass
            
            if not member_data.empty:
                # Children diversity
                if 'Children_Category' in member_data.columns:
                    children = member_data['Children_Category'].iloc[0]
                    if pd.notna(children):
                        unique_children_categories.add(children)
        
        # Calculate diversity score for this circle - include ALL circles, even those with no data
        if unique_children_categories:
            score = len(unique_children_categories)
            circle_children_diversity_scores[circle_id] = score
            circles_with_data += 1
        else:
            # Still include the circle but with a score of 0
            circle_children_diversity_scores[circle_id] = 0
            circles_with_no_data += 1
    
    # Calculate total score across all circles
    total_score = sum(circle_children_diversity_scores.values())
    
    # Debug information
    print(f"DIVERSITY DEBUG - Children: {circles_processed} circles processed, {circles_with_data} with data, {circles_with_no_data} without data, total score: {total_score}")
    
    return total_score

# Function to render different tabs in the UI
def render_match_tab():
    """Render the main matching tab content"""
    # Organizing into columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Participant Data")
        
        # File uploader for participant data
        uploaded_file = st.file_uploader("Upload a CSV file with participant data", type=["csv"])
        
        if uploaded_file is not None:
            # Add a 'Process Data' button for explicit processing
            process_button = st.button("Process Data")
            
            if process_button or 'processed_data' not in st.session_state:
                # If button is clicked or no processed data exists, process the data
                from app import process_uploaded_file
                process_uploaded_file(uploaded_file)
                st.success("Data processed successfully!")
                st.session_state.button_clicked = True
        else:
            # Clear session state if no file is uploaded
            if 'processed_data' in st.session_state:
                st.warning("File removed. Upload a new file to continue.")
                # Keep a backup of the current data for inspection
                if 'backup_data' not in st.session_state:
                    st.session_state.backup_data = st.session_state.processed_data.copy() if st.session_state.processed_data is not None else None
                
                # Clear current working data
                st.session_state.processed_data = None
                st.session_state.button_clicked = False
    
    with col2:
        st.subheader("Matching Algorithm")
        
        # Optimization button with options - only if data has been processed
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            from app import run_optimization
            
            # Create expandable section for advanced options
            with st.expander("Advanced Options"):
                # Always use 'optimize' mode (selector removed as requested)
                # This mode allows NEW participants to join continuing circles while preserving CURRENT-CONTINUING members
                
                st.selectbox("Circle Size Preference", 
                            options=["Balanced", "Larger Circles", "Smaller Circles"],
                            index=0,
                            key="circle_size_pref",
                            help="'Balanced' tries to get everyone in a circle. 'Larger Circles' favors having fewer, larger circles. 'Smaller Circles' favors more intimate groups.")
                
                # Weight for location preferences
                st.slider("Location Match Weight", 
                        min_value=1.0, 
                        max_value=10.0, 
                        value=5.0, 
                        step=0.5,
                        key="location_weight",
                        help="Higher values prioritize participants' location preferences over other factors.")
                
                # Weight for time preferences
                st.slider("Time Match Weight", 
                        min_value=1.0, 
                        max_value=10.0, 
                        value=5.0, 
                        step=0.5,
                        key="time_weight",
                        help="Higher values prioritize participants' time preferences over other factors.")
                
                # Weight for full circles (only relevant for Balanced mode)
                st.slider("Complete Circle Weight", 
                        min_value=1.0, 
                        max_value=10.0, 
                        value=3.0, 
                        step=0.5,
                        key="circle_weight",
                        help="Higher values prioritize having complete circles over partial matches.")
                
                # Tradeoff between honoring existing circles and optimizing for preferences
                st.slider("Existing Circle Preservation", 
                        min_value=0.0, 
                        max_value=10.0, 
                        value=7.0, 
                        step=0.5,
                        key="existing_circle_weight",
                        help="Higher values favor keeping members in their current circles rather than moving them.")
                
                # Maximum iterations for the solver
                st.number_input("Max Solver Iterations", 
                                min_value=1000, 
                                max_value=1000000, 
                                value=100000, 
                                step=10000,
                                key="max_iterations",
                                help="Maximum number of iterations for the optimization solver. Higher values may improve solutions but take longer.")
                
                # Tolerance for solver convergence
                st.number_input("Solver Tolerance", 
                                min_value=0.0001, 
                                max_value=0.1, 
                                value=0.001, 
                                format="%f",
                                step=0.001,
                                key="solver_tolerance",
                                help="Numerical tolerance for solver convergence. Lower values give more precise results but may take longer.")
            
            # Run Optimization button
            if st.button("Run Matching Algorithm"):
                with st.spinner("Running optimization - this may take a moment..."):
                    run_optimization()
                
                # Show results
                if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
                    st.success("Matching completed successfully!")
                    st.session_state.active_tab = 1  # Switch to Results tab
                    
                    # Use this to force a rerun to switch tabs
                    # Replaced st.experimental_rerun() with st.rerun()
                    st.rerun()
        else:
            st.info("Upload and process data to enable matching.")


def render_details_tab():
    """Render the details tab content"""
    if ('matched_circles' not in st.session_state or 
        st.session_state.matched_circles is None or 
        (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty)):
        st.warning("No matching results available. Please run the matching algorithm first.")
        return
    
    # Create tabs for different views, adding Split Circles tab
    detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs(["Overview", "Circles", "Participants", "Split Circles"])
    
    with detail_tab1:
        # Use render_results_overview without showing the split circle summary inside it
        # to avoid duplicate rendering
        st.session_state.skip_split_circle_summary = True
        render_results_overview()
        st.session_state.skip_split_circle_summary = False
    
    with detail_tab2:
        render_circle_details()
    
    with detail_tab3:
        render_participant_details()
        
    with detail_tab4:
        # Use specific tab key prefix
        render_split_circle_summary(key_prefix="details_tab4")


def render_demographics_tab():
    """Render the demographics analysis tab content"""
    st.subheader("Demographics Analysis")
    
    # Add filter by region and match status at the top
    col1, col2 = st.columns(2)
    
    with col1:
        # Get available regions
        available_regions = []
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            if 'Current_Region' in df.columns:
                available_regions = sorted(df['Current_Region'].dropna().unique().tolist())
            elif 'Region' in df.columns:
                available_regions = sorted(df['Region'].dropna().unique().tolist())
            
        # Add "All Regions" option
        available_regions = ["All Regions"] + available_regions
        
        # Region filter
        selected_region = st.selectbox("Filter by Region", options=available_regions, index=0, key="participant_region_filter")
    
    with col2:
        # Match status filter
        match_options = ["All Participants", "Matched", "Unmatched"]
        selected_match = st.selectbox("Filter by Match Status", options=match_options, index=0, key="participant_match_status_filter")
    
    # Get the filtered data based on selections
    filtered_data = None
    if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
        filtered_data = st.session_state.processed_data.copy()
        
        # Apply region filter if not "All Regions"
        if selected_region != "All Regions":
            if 'Current_Region' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Current_Region'] == selected_region]
            elif 'Region' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Region'] == selected_region]
        
        # Apply match status filter
        if selected_match == "Matched" and 'proposed_NEW_circles_id' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['proposed_NEW_circles_id'] != "UNMATCHED"]
        elif selected_match == "Unmatched" and 'proposed_NEW_circles_id' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['proposed_NEW_circles_id'] == "UNMATCHED"]
    
    # Create tabs for different demographic views
    demo_tab1, demo_tab2, demo_tab3, demo_tab4, demo_tab5, demo_tab6 = st.tabs(["Class Vintage", "Employment", "Industry", "Racial Identity", "Children", "Circles Detail"])
    
    with demo_tab1:
        render_class_vintage_analysis(filtered_data)
    
    with demo_tab2:
        render_employment_analysis(filtered_data)
    
    with demo_tab3:
        render_industry_analysis(filtered_data)
    
    with demo_tab4:
        render_racial_identity_analysis(filtered_data)
    
    with demo_tab5:
        render_children_analysis(filtered_data)
    
    with demo_tab6:
        render_circles_detail()

def render_vintage_diversity_histogram():
    """
    Create a histogram showing the distribution of circles based on 
    the number of different class vintages they contain
    """
    # First check if we have the necessary data
    if 'matched_circles' not in st.session_state or st.session_state.matched_circles is None:
        st.warning("No matched circles data available to analyze vintage diversity.")
        return
        
    # Check if the matched_circles dataframe is empty
    if hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty:
        st.warning("No matched circles data available (empty dataframe).")
        return
        
    if 'results' not in st.session_state or st.session_state.results is None:
        st.warning("No results data available to analyze vintage diversity.")
        return
    
    # Get the circle data
    circles_df = st.session_state.matched_circles.copy()
    results_df = st.session_state.results.copy()
    
    # Create score columns if they don't exist
    if 'vintage_score' not in circles_df.columns:
        circles_df['vintage_score'] = 0
    
    # DEBUG: Show total circles at the start
    total_initial_circles = len(circles_df)
    st.caption(f"🔍 **Demographics DEBUG:** Starting with {total_initial_circles} total circles")
    
    # Check for member_count column
    if 'member_count' not in circles_df.columns:
        st.warning("Circles data does not have member_count column - will use members list length instead")
        # Add member_count derived from members list
        circles_df['member_count'] = circles_df.apply(
            lambda row: len(eval(row['members'])) if isinstance(row['members'], str) and row['members'].startswith('[') 
            else (len(row['members']) if isinstance(row['members'], list) else 1), 
            axis=1
        )
    
    # Capture circles with zero member_count but with actual members
    zero_count_with_members = 0
    if 'members' in circles_df.columns:
        for idx, row in circles_df.iterrows():
            if row['member_count'] == 0 and row['members']:
                # For list representation
                if isinstance(row['members'], list) and len(row['members']) > 0:
                    circles_df.at[idx, 'member_count'] = len(row['members'])
                    zero_count_with_members += 1
                # For string representation
                elif isinstance(row['members'], str):
                    try:
                        if row['members'].startswith('['):
                            member_list = eval(row['members'])
                            if len(member_list) > 0:
                                circles_df.at[idx, 'member_count'] = len(member_list)
                                zero_count_with_members += 1
                    except:
                        pass
    
    if zero_count_with_members > 0:
        st.caption(f"🔧 Fixed {zero_count_with_members} circles that had zero member_count but had actual members")
    
    # FILTER STEP 1: Circles with no members (KEEP this filter)
    previous_count = len(circles_df)
    circles_df = circles_df[circles_df['member_count'] > 0]
    filtered_no_members = previous_count - len(circles_df)
    
    if filtered_no_members > 0:
        st.caption(f"⚠️ Excluded {filtered_no_members} circles with no members (member_count=0)")
    
    if len(circles_df) == 0:
        st.warning("No circles with members available for analysis.")
        return
    
    # Dictionary to track unique vintages per circle and debugging info
    circle_vintage_counts = {}
    circle_vintage_diversity_scores = {}
    circles_with_no_vintage_data = []
    circles_with_parsing_errors = []
    circles_included = []
    
    # Track special debug circles
    debug_circles = ['IP-ATL-1', 'IP-BOS-01']
    
    # Get vintage data for each member of each circle
    for idx, circle_row in circles_df.iterrows():
        circle_id = circle_row['circle_id']
        is_debug_circle = circle_id in debug_circles
        
        # Initialize empty set to track unique vintages
        unique_vintages = set()
        has_parsing_error = False
        
        # IMPROVED APPROACH: Get the list of members for this circle
        # Method 1: Try to get members from the circle_row['members']
        members_from_row = []
        if 'members' in circle_row and circle_row['members'] and not pd.isna(circle_row['members']).all():
            # For list representation
            if isinstance(circle_row['members'], list):
                members_from_row = [m for m in circle_row['members'] if not pd.isna(m)]
            # For string representation - convert to list
            elif isinstance(circle_row['members'], str):
                try:
                    if circle_row['members'].startswith('['):
                        members_from_row = [m for m in eval(circle_row['members']) if not pd.isna(m)]
                    else:
                        members_from_row = [circle_row['members']]
                except Exception as e:
                    # Record parsing errors
                    has_parsing_error = True
                    circles_with_parsing_errors.append(circle_id)
            else:
                has_parsing_error = True
                circles_with_parsing_errors.append(circle_id)
        
        # Method 2: Get members by looking up the circle_id in the results dataframe
        members_from_lookup = []
        if 'proposed_NEW_circles_id' in results_df.columns:
            # Find all participants assigned to this circle
            circle_members = results_df[results_df['proposed_NEW_circles_id'] == circle_id]
            if not circle_members.empty and 'Encoded ID' in circle_members.columns:
                members_from_lookup = circle_members['Encoded ID'].dropna().tolist()
        
        # Combine both methods, prioritizing non-empty results
        member_ids = []
        if members_from_lookup:
            member_ids = members_from_lookup
        else:
            member_ids = members_from_row
        
        if is_debug_circle:
            print(f"VINTAGE DEBUG - {circle_id}: Found {len(member_ids)} members using improved extraction")
                
        # For each member, look up their vintage in results_df
        for member_id in member_ids:
            # Skip NaN or invalid member IDs
            if pd.isna(member_id):
                continue
                
            # Try exact match first
            member_data = results_df[results_df['Encoded ID'] == member_id]
            
            # If no match, try converting both to strings for comparison
            if member_data.empty:
                # Convert to string and try again
                member_data = results_df[results_df['Encoded ID'].astype(str) == str(member_id)]
                
                # If still no match and member_id has numeric format
                if member_data.empty and str(member_id).replace('.', '', 1).isdigit():
                    try:
                        # Try as float
                        float_id = float(member_id)
                        member_data = results_df[results_df['Encoded ID'].astype(float) == float_id]
                        
                        # Try as int if it's a whole number
                        if member_data.empty and float_id.is_integer():
                            int_id = int(float_id)
                            member_data = results_df[results_df['Encoded ID'].astype(int) == int_id]
                    except:
                        pass
            
            if not member_data.empty and 'Class_Vintage' in member_data.columns:
                vintage = member_data['Class_Vintage'].iloc[0]
                if pd.notna(vintage):
                    unique_vintages.add(vintage)
        
        # INCLUDE ALL CIRCLES in the analysis, even if they have no vintage data
        # Those without vintage data get a count of 0
        if unique_vintages:  
            # Circle has vintage data
            count = len(unique_vintages)
            circle_vintage_counts[circle_id] = count
            circle_vintage_diversity_scores[circle_id] = count
            circles_included.append(circle_id)
            
            # CRITICAL FIX: Update the dataframe with the vintage score for this circle
            circles_df.at[idx, 'vintage_score'] = count
            
            if is_debug_circle:
                print(f"VINTAGE DEBUG - {circle_id}: Updated score to {count} (found {len(unique_vintages)} unique vintages)")
        else:
            # Circle has no vintage data - still include with 0 count
            circle_vintage_counts[circle_id] = 0
            circle_vintage_diversity_scores[circle_id] = 0
            circles_df.at[idx, 'vintage_score'] = 0
            
            if is_debug_circle:
                print(f"VINTAGE DEBUG - {circle_id}: No vintage data found, set score to 0")
                
            if not has_parsing_error:
                circles_with_no_vintage_data.append(circle_id)
    
    # Show debug information
    if circles_with_parsing_errors:
        st.caption(f"⚠️ {len(circles_with_parsing_errors)} circles had parsing errors but are still included with 0 diversity score")
    
    if circles_with_no_vintage_data:
        st.caption(f"ℹ️ {len(circles_with_no_vintage_data)} circles had no vintage data but are still included with 0 diversity score")
    
    # Process histogram data from the vintage counts
    total_included = len(circle_vintage_counts)
    st.caption(f"✅ Including {total_included} total circles in the analysis (out of {total_initial_circles} initial circles)")
    
    # Count circles by number of unique vintages - INCLUDING zeros this time
    diversity_counts = pd.Series(circle_vintage_counts).value_counts().sort_index()
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Number of Vintages': diversity_counts.index,
        'Number of Circles': diversity_counts.values
    })
    
    st.subheader("Vintage Diversity Within Circles")
    
    # Create histogram using plotly with Stanford cardinal red color
    fig = px.bar(
        plot_df,
        x='Number of Vintages',
        y='Number of Circles',
        title=f'Distribution of Circles by Number of Class Vintages (Total: {total_included} circles)',
        text='Number of Circles',  # Display count values on bars
        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
    )
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis=dict(
            title="Number of Different Class Vintages",
            tickmode='linear',
            dtick=1  # Force integer labels
        ),
        yaxis_title="Number of Circles"
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True, key="plot_331")
    
    # Show a table with the data
    st.caption("Data table:")
    st.dataframe(plot_df, hide_index=True)
    
    # Display the Vintage diversity score
    st.subheader("Vintage Diversity Score")
    st.write("""
    For each circle, the vintage diversity score is calculated as follows:
    - 0 points: No vintage data available
    - 1 point: All members in the same class vintage
    - 2 points: Members from two different class vintages
    - 3 points: Members from three different class vintages
    - And so on, with more points for more diverse circles
    """)
    
    # Calculate average and total diversity scores - now including circles with 0 score
    total_diversity_score = sum(circle_vintage_diversity_scores.values())
    avg_diversity_score = total_diversity_score / len(circle_vintage_diversity_scores) if circle_vintage_diversity_scores else 0
    
    # Store the total score in session state for use in the Match tab
    st.session_state.vintage_diversity_score = total_diversity_score
    
    # Create two columns for the metrics
    col1, col2 = st.columns(2)
    
    # Show average and total scores
    with col1:
        st.metric("Average Vintage Diversity Score", f"{avg_diversity_score:.2f}")
    with col2:
        st.metric("Total Vintage Diversity Score", f"{total_diversity_score}")
    
    # Add a brief explanation
    total_circles = sum(diversity_counts.values)
    circles_with_data = sum(diversity_counts[diversity_counts.index > 0].values)
    diverse_circles = sum(diversity_counts[diversity_counts.index > 1].values)
    
    no_data_circles = 0
    if 0 in diversity_counts.index:
        no_data_circles = diversity_counts[0]
    
    # Calculate percentages
    data_pct = (circles_with_data / total_circles * 100) if total_circles > 0 else 0
    diverse_pct = (diverse_circles / circles_with_data * 100) if circles_with_data > 0 else 0
    
    st.write(f"Out of {total_circles} total circles:")
    
    if no_data_circles > 0:
        st.write(f"- {no_data_circles} circles ({100-data_pct:.1f}%) have no vintage data and received a score of 0")
    
    st.write(f"- {circles_with_data} circles ({data_pct:.1f}%) have vintage data")
    st.write(f"- Among circles with vintage data, {diverse_circles} ({diverse_pct:.1f}%) contain members from multiple vintages")
    
    # CRITICAL FIX: Update the session state with our modified circles_df that now has diversity scores
    st.session_state.matched_circles = circles_df
    print(f"VINTAGE HISTOGRAM UPDATE - Updated session state matched_circles with calculated vintage scores for {len(circles_df)} circles")


def render_employment_diversity_histogram():
    """
    Create a histogram showing the distribution of circles based on 
    the number of different employment categories they contain
    """
    # First check if we have the necessary data
    if 'matched_circles' not in st.session_state or st.session_state.matched_circles is None:
        st.warning("No matched circles data available to analyze employment diversity.")
        return
        
    # Check if the matched_circles dataframe is empty
    if hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty:
        st.warning("No matched circles data available (empty dataframe).")
        return
        
    if 'results' not in st.session_state or st.session_state.results is None:
        st.warning("No results data available to analyze employment diversity.")
        return
    
    # Get the circle data
    circles_df = st.session_state.matched_circles.copy()
    results_df = st.session_state.results.copy()
    
    # DEBUG: Show total circles at the start
    total_initial_circles = len(circles_df)
    st.caption(f"🔍 **Demographics DEBUG:** Starting with {total_initial_circles} total circles")
    
    # Check for member_count column
    if 'member_count' not in circles_df.columns:
        st.warning("Circles data does not have member_count column - will use members list length instead")
        # Add member_count derived from members list
        circles_df['member_count'] = circles_df.apply(
            lambda row: len(eval(row['members'])) if isinstance(row['members'], str) and row['members'].startswith('[') 
            else (len(row['members']) if isinstance(row['members'], list) else 1), 
            axis=1
        )
    
    # Capture circles with zero member_count but with actual members
    zero_count_with_members = 0
    if 'members' in circles_df.columns:
        for idx, row in circles_df.iterrows():
            if row['member_count'] == 0 and row['members']:
                # For list representation
                if isinstance(row['members'], list) and len(row['members']) > 0:
                    circles_df.at[idx, 'member_count'] = len(row['members'])
                    zero_count_with_members += 1
                # For string representation
                elif isinstance(row['members'], str):
                    try:
                        if row['members'].startswith('['):
                            member_list = eval(row['members'])
                            if len(member_list) > 0:
                                circles_df.at[idx, 'member_count'] = len(member_list)
                                zero_count_with_members += 1
                    except:
                        pass
    
    if zero_count_with_members > 0:
        st.caption(f"🔧 Fixed {zero_count_with_members} circles that had zero member_count but had actual members")
    
    # FILTER STEP 1: Circles with no members (KEEP this filter)
    previous_count = len(circles_df)
    circles_df = circles_df[circles_df['member_count'] > 0]
    filtered_no_members = previous_count - len(circles_df)
    
    if filtered_no_members > 0:
        st.caption(f"⚠️ Excluded {filtered_no_members} circles with no members (member_count=0)")
    
    if len(circles_df) == 0:
        st.warning("No circles with members available for analysis.")
        return
    
    # Create score column if it doesn't exist
    if 'employment_score' not in circles_df.columns:
        circles_df['employment_score'] = 0
    
    # Dictionary to track unique employment categories per circle and debugging info
    circle_employment_counts = {}
    circle_employment_diversity_scores = {}
    circles_with_no_employment_data = []
    circles_with_parsing_errors = []
    circles_included = []
    
    # Track special debug circles
    debug_circles = ['IP-ATL-1', 'IP-BOS-01']
    
    # Get employment data for each member of each circle
    for idx, circle_row in circles_df.iterrows():
        circle_id = circle_row['circle_id']
        is_debug_circle = circle_id in debug_circles
        
        # Initialize empty set to track unique employment categories
        unique_employment_categories = set()
        has_parsing_error = False
        
        # IMPROVED APPROACH: Get the list of members for this circle
        # Method 1: Try to get members from the circle_row['members']
        members_from_row = []
        if 'members' in circle_row and circle_row['members'] and not pd.isna(circle_row['members']).all():
            # For list representation
            if isinstance(circle_row['members'], list):
                members_from_row = [m for m in circle_row['members'] if not pd.isna(m)]
            # For string representation - convert to list
            elif isinstance(circle_row['members'], str):
                try:
                    if circle_row['members'].startswith('['):
                        members_from_row = [m for m in eval(circle_row['members']) if not pd.isna(m)]
                    else:
                        members_from_row = [circle_row['members']]
                except Exception as e:
                    # Record parsing errors
                    has_parsing_error = True
                    circles_with_parsing_errors.append(circle_id)
            else:
                has_parsing_error = True
                circles_with_parsing_errors.append(circle_id)
        
        # Method 2: Get members by looking up the circle_id in the results dataframe
        members_from_lookup = []
        if 'proposed_NEW_circles_id' in results_df.columns:
            # Find all participants assigned to this circle
            circle_members = results_df[results_df['proposed_NEW_circles_id'] == circle_id]
            if not circle_members.empty and 'Encoded ID' in circle_members.columns:
                members_from_lookup = circle_members['Encoded ID'].dropna().tolist()
        
        # Combine both methods, prioritizing non-empty results
        member_ids = []
        if members_from_lookup:
            member_ids = members_from_lookup
        else:
            member_ids = members_from_row
        
        if is_debug_circle:
            print(f"EMPLOYMENT DEBUG - {circle_id}: Found {len(member_ids)} members using improved extraction")
                
        # For each member, look up their employment category in results_df
        for member_id in member_ids:
            # Skip NaN or invalid member IDs
            if pd.isna(member_id):
                continue
                
            # Try exact match first
            member_data = results_df[results_df['Encoded ID'] == member_id]
            
            # If no match, try converting both to strings for comparison
            if member_data.empty:
                # Convert to string and try again
                member_data = results_df[results_df['Encoded ID'].astype(str) == str(member_id)]
                
                # If still no match and member_id has numeric format
                if member_data.empty and str(member_id).replace('.', '', 1).isdigit():
                    try:
                        # Try as float
                        float_id = float(member_id)
                        member_data = results_df[results_df['Encoded ID'].astype(float) == float_id]
                        
                        # Try as int if it's a whole number
                        if member_data.empty and float_id.is_integer():
                            int_id = int(float_id)
                            member_data = results_df[results_df['Encoded ID'].astype(int) == int_id]
                    except:
                        pass
            
            if not member_data.empty and 'Employment_Category' in member_data.columns:
                employment_category = member_data['Employment_Category'].iloc[0]
                if pd.notna(employment_category):
                    unique_employment_categories.add(employment_category)
        
        # INCLUDE ALL CIRCLES in the analysis, even if they have no employment data
        # Those without employment data get a count of 0
        if unique_employment_categories:
            # Circle has employment data
            count = len(unique_employment_categories)
            circle_employment_counts[circle_id] = count
            circle_employment_diversity_scores[circle_id] = count
            circles_included.append(circle_id)
            
            # CRITICAL FIX: Update the dataframe with the employment score for this circle
            circles_df.at[idx, 'employment_score'] = count
            
            if is_debug_circle:
                print(f"EMPLOYMENT DEBUG - {circle_id}: Updated score to {count} (found {len(unique_employment_categories)} unique categories)")
        else:
            # Circle has no employment data - still include with 0 count
            circle_employment_counts[circle_id] = 0
            circle_employment_diversity_scores[circle_id] = 0
            circles_df.at[idx, 'employment_score'] = 0
            
            if is_debug_circle:
                print(f"EMPLOYMENT DEBUG - {circle_id}: No employment data found, set score to 0")
                
            if not has_parsing_error:
                circles_with_no_employment_data.append(circle_id)
    
    # Show debug information
    if circles_with_parsing_errors:
        st.caption(f"⚠️ {len(circles_with_parsing_errors)} circles had parsing errors but are still included with 0 diversity score")
    
    if circles_with_no_employment_data:
        st.caption(f"ℹ️ {len(circles_with_no_employment_data)} circles had no employment data but are still included with 0 diversity score")
    
    # Process histogram data from the employment counts
    total_included = len(circle_employment_counts)
    st.caption(f"✅ Including {total_included} total circles in the analysis (out of {total_initial_circles} initial circles)")
    
    # Count circles by number of unique employment categories - INCLUDING zeros this time
    diversity_counts = pd.Series(circle_employment_counts).value_counts().sort_index()
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Number of Employment Categories': diversity_counts.index,
        'Number of Circles': diversity_counts.values
    })
    
    st.subheader("Employment Diversity Within Circles")
    
    # Create histogram using plotly with Stanford cardinal red color
    fig = px.bar(
        plot_df,
        x='Number of Employment Categories',
        y='Number of Circles',
        title=f'Distribution of Circles by Number of Employment Categories (Total: {total_included} circles)',
        text='Number of Circles',  # Display count values on bars
        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
    )
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis=dict(
            title="Number of Different Employment Categories",
            tickmode='linear',
            dtick=1  # Force integer labels
        ),
        yaxis_title="Number of Circles"
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True, key="plot_488")
    
    # Show a table with the data
    st.caption("Data table:")
    st.dataframe(plot_df, hide_index=True)
    
    # Display the Employment diversity score
    st.subheader("Employment Diversity Score")
    st.write("""
    For each circle, the employment diversity score is calculated as follows:
    - 0 points: No employment data available
    - 1 point: All members in the same employment category
    - 2 points: Members from two different employment categories
    - 3 points: Members from three or more employment categories
    """)
    
    # Calculate average and total diversity scores - now including circles with 0 score
    total_diversity_score = sum(circle_employment_diversity_scores.values())
    avg_diversity_score = total_diversity_score / len(circle_employment_diversity_scores) if circle_employment_diversity_scores else 0
    
    # Store the total score in session state for use in the Match tab
    st.session_state.employment_diversity_score = total_diversity_score
    
    # Create two columns for the metrics
    col1, col2 = st.columns(2)
    
    # Show average and total scores
    with col1:
        st.metric("Average Employment Diversity Score", f"{avg_diversity_score:.2f}")
    with col2:
        st.metric("Total Employment Diversity Score", f"{total_diversity_score}")
    
    # Add a brief explanation
    total_circles = sum(diversity_counts.values)
    circles_with_data = sum(diversity_counts[diversity_counts.index > 0].values)
    diverse_circles = sum(diversity_counts[diversity_counts.index > 1].values)
    
    no_data_circles = 0
    if 0 in diversity_counts.index:
        no_data_circles = diversity_counts[0]
    
    # Calculate percentages
    data_pct = (circles_with_data / total_circles * 100) if total_circles > 0 else 0
    diverse_pct = (diverse_circles / circles_with_data * 100) if circles_with_data > 0 else 0
    
    st.write(f"Out of {total_circles} total circles:")
    
    if no_data_circles > 0:
        st.write(f"- {no_data_circles} circles ({100-data_pct:.1f}%) have no employment data and received a score of 0")
    
    st.write(f"- {circles_with_data} circles ({data_pct:.1f}%) have employment data")
    st.write(f"- Among circles with employment data, {diverse_circles} ({diverse_pct:.1f}%) contain members from multiple employment categories")
    
    # CRITICAL FIX: Update the session state with our modified circles_df that now has diversity scores
    st.session_state.matched_circles = circles_df
    print(f"EMPLOYMENT HISTOGRAM UPDATE - Updated session state matched_circles with calculated employment scores for {len(circles_df)} circles")

def render_industry_diversity_histogram():
    """
    Create a histogram showing the distribution of circles based on 
    the number of different industry categories they contain
    """
    # First check if we have the necessary data
    if 'matched_circles' not in st.session_state or st.session_state.matched_circles is None:
        st.warning("No matched circles data available to analyze industry diversity.")
        return
        
    # Check if the matched_circles dataframe is empty
    if hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty:
        st.warning("No matched circles data available (empty dataframe).")
        return
        
    if 'results' not in st.session_state or st.session_state.results is None:
        st.warning("No results data available to analyze industry diversity.")
        return
    
    # Get the circle data
    circles_df = st.session_state.matched_circles.copy()
    results_df = st.session_state.results.copy()
    
    # DEBUG: Show total circles at the start
    total_initial_circles = len(circles_df)
    st.caption(f"🔍 **Demographics DEBUG:** Starting with {total_initial_circles} total circles")
    
    # Check for member_count column
    if 'member_count' not in circles_df.columns:
        st.warning("Circles data does not have member_count column - will use members list length instead")
        # Add member_count derived from members list
        circles_df['member_count'] = circles_df.apply(
            lambda row: len(eval(row['members'])) if isinstance(row['members'], str) and row['members'].startswith('[') 
            else (len(row['members']) if isinstance(row['members'], list) else 1), 
            axis=1
        )
    
    # Capture circles with zero member_count but with actual members
    zero_count_with_members = 0
    if 'members' in circles_df.columns:
        for idx, row in circles_df.iterrows():
            if row['member_count'] == 0 and row['members']:
                # For list representation
                if isinstance(row['members'], list) and len(row['members']) > 0:
                    circles_df.at[idx, 'member_count'] = len(row['members'])
                    zero_count_with_members += 1
                # For string representation
                elif isinstance(row['members'], str):
                    try:
                        if row['members'].startswith('['):
                            member_list = eval(row['members'])
                            if len(member_list) > 0:
                                circles_df.at[idx, 'member_count'] = len(member_list)
                                zero_count_with_members += 1
                    except:
                        pass
    
    if zero_count_with_members > 0:
        st.caption(f"🔧 Fixed {zero_count_with_members} circles that had zero member_count but had actual members")
    
    # FILTER STEP 1: Circles with no members (KEEP this filter)
    previous_count = len(circles_df)
    circles_df = circles_df[circles_df['member_count'] > 0]
    filtered_no_members = previous_count - len(circles_df)
    
    if filtered_no_members > 0:
        st.caption(f"⚠️ Excluded {filtered_no_members} circles with no members (member_count=0)")
    
    if len(circles_df) == 0:
        st.warning("No circles with members available for analysis.")
        return
    
    # Create score column if it doesn't exist
    if 'industry_score' not in circles_df.columns:
        circles_df['industry_score'] = 0
    
    # Dictionary to track unique industry categories per circle and debugging info
    circle_industry_counts = {}
    circle_industry_diversity_scores = {}
    circles_with_no_industry_data = []
    circles_with_parsing_errors = []
    circles_included = []
    
    # Track special debug circles
    debug_circles = ['IP-ATL-1', 'IP-BOS-01']
    
    # Get industry data for each member of each circle
    for idx, circle_row in circles_df.iterrows():
        circle_id = circle_row['circle_id']
        is_debug_circle = circle_id in debug_circles
        
        # Initialize empty set to track unique industry categories
        unique_industry_categories = set()
        has_parsing_error = False
        
        # IMPROVED APPROACH: Get the list of members for this circle
        # Method 1: Try to get members from the circle_row['members']
        members_from_row = []
        if 'members' in circle_row and circle_row['members'] and not pd.isna(circle_row['members']).all():
            # For list representation
            if isinstance(circle_row['members'], list):
                members_from_row = [m for m in circle_row['members'] if not pd.isna(m)]
            # For string representation - convert to list
            elif isinstance(circle_row['members'], str):
                try:
                    if circle_row['members'].startswith('['):
                        members_from_row = [m for m in eval(circle_row['members']) if not pd.isna(m)]
                    else:
                        members_from_row = [circle_row['members']]
                except Exception as e:
                    # Record parsing errors
                    has_parsing_error = True
                    circles_with_parsing_errors.append(circle_id)
            else:
                has_parsing_error = True
                circles_with_parsing_errors.append(circle_id)
        
        # Method 2: Get members by looking up the circle_id in the results dataframe
        members_from_lookup = []
        if 'proposed_NEW_circles_id' in results_df.columns:
            # Find all participants assigned to this circle
            circle_members = results_df[results_df['proposed_NEW_circles_id'] == circle_id]
            if not circle_members.empty and 'Encoded ID' in circle_members.columns:
                members_from_lookup = circle_members['Encoded ID'].dropna().tolist()
        
        # Combine both methods, prioritizing non-empty results
        member_ids = []
        if members_from_lookup:
            member_ids = members_from_lookup
        else:
            member_ids = members_from_row
        
        if is_debug_circle:
            print(f"INDUSTRY DEBUG - {circle_id}: Found {len(member_ids)} members using improved extraction")
                
        # For each member, look up their industry category in results_df
        for member_id in member_ids:
            # Skip NaN or invalid member IDs
            if pd.isna(member_id):
                continue
                
            # Try exact match first
            member_data = results_df[results_df['Encoded ID'] == member_id]
            
            # If no match, try converting both to strings for comparison
            if member_data.empty:
                # Convert to string and try again
                member_data = results_df[results_df['Encoded ID'].astype(str) == str(member_id)]
                
                # If still no match and member_id has numeric format
                if member_data.empty and str(member_id).replace('.', '', 1).isdigit():
                    try:
                        # Try as float
                        float_id = float(member_id)
                        member_data = results_df[results_df['Encoded ID'].astype(float) == float_id]
                        
                        # Try as int if it's a whole number
                        if member_data.empty and float_id.is_integer():
                            int_id = int(float_id)
                            member_data = results_df[results_df['Encoded ID'].astype(int) == int_id]
                    except:
                        pass
            
            if not member_data.empty and 'Industry_Category' in member_data.columns:
                industry_category = member_data['Industry_Category'].iloc[0]
                if pd.notna(industry_category):
                    unique_industry_categories.add(industry_category)
        
        # INCLUDE ALL CIRCLES in the analysis, even if they have no industry data
        # Those without industry data get a count of 0
        if unique_industry_categories:
            # Circle has industry data
            count = len(unique_industry_categories)
            circle_industry_counts[circle_id] = count
            circle_industry_diversity_scores[circle_id] = count
            circles_included.append(circle_id)
            
            # CRITICAL FIX: Update the dataframe with the industry score for this circle
            circles_df.at[idx, 'industry_score'] = count
            
            if is_debug_circle:
                print(f"INDUSTRY DEBUG - {circle_id}: Updated score to {count} (found {len(unique_industry_categories)} unique categories)")
        else:
            # Circle has no industry data - still include with 0 count
            circle_industry_counts[circle_id] = 0
            circle_industry_diversity_scores[circle_id] = 0
            circles_df.at[idx, 'industry_score'] = 0
            
            if is_debug_circle:
                print(f"INDUSTRY DEBUG - {circle_id}: No industry data found, set score to 0")
                
            if not has_parsing_error:
                circles_with_no_industry_data.append(circle_id)
    
    # Show debug information
    if circles_with_parsing_errors:
        st.caption(f"⚠️ {len(circles_with_parsing_errors)} circles had parsing errors but are still included with 0 diversity score")
    
    if circles_with_no_industry_data:
        st.caption(f"ℹ️ {len(circles_with_no_industry_data)} circles had no industry data but are still included with 0 diversity score")
    
    # Process histogram data from the industry counts
    total_included = len(circle_industry_counts)
    st.caption(f"✅ Including {total_included} total circles in the analysis (out of {total_initial_circles} initial circles)")
    
    # Count circles by number of unique industry categories - INCLUDING zeros this time
    diversity_counts = pd.Series(circle_industry_counts).value_counts().sort_index()
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Number of Industry Categories': diversity_counts.index,
        'Number of Circles': diversity_counts.values
    })
    
    st.subheader("Industry Diversity Within Circles")
    
    # Create histogram using plotly with Stanford cardinal red color
    fig = px.bar(
        plot_df,
        x='Number of Industry Categories',
        y='Number of Circles',
        title=f'Distribution of Circles by Number of Industry Categories (Total: {total_included} circles)',
        text='Number of Circles',  # Display count values on bars
        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
    )
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis=dict(
            title="Number of Different Industry Categories",
            tickmode='linear',
            dtick=1  # Force integer labels
        ),
        yaxis_title="Number of Circles"
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True, key="plot_643")
    
    # Show a table with the data
    st.caption("Data table:")
    st.dataframe(plot_df, hide_index=True)
    
    # Display the Industry diversity score
    st.subheader("Industry Diversity Score")
    st.write("""
    For each circle, the industry diversity score is calculated as follows:
    - 0 points: No industry data available
    - 1 point: All members in the same industry category
    - 2 points: Members from two different industry categories
    - 3 points: Members from three different industry categories
    - 4 points: Members from all four industry categories
    """)
    
    # Calculate average and total diversity scores - now including circles with 0 score
    total_diversity_score = sum(circle_industry_diversity_scores.values())
    avg_diversity_score = total_diversity_score / len(circle_industry_diversity_scores) if circle_industry_diversity_scores else 0
    
    # Store the total score in session state for use in the Match tab
    st.session_state.industry_diversity_score = total_diversity_score
    
    # Create two columns for the metrics
    col1, col2 = st.columns(2)
    
    # Show average and total scores
    with col1:
        st.metric("Average Industry Diversity Score", f"{avg_diversity_score:.2f}")
    with col2:
        st.metric("Total Industry Diversity Score", f"{total_diversity_score}")
    
    # Add a brief explanation
    total_circles = sum(diversity_counts.values)
    circles_with_data = sum(diversity_counts[diversity_counts.index > 0].values)
    diverse_circles = sum(diversity_counts[diversity_counts.index > 1].values)
    
    no_data_circles = 0
    if 0 in diversity_counts.index:
        no_data_circles = diversity_counts[0]
    
    # Calculate percentages
    data_pct = (circles_with_data / total_circles * 100) if total_circles > 0 else 0
    diverse_pct = (diverse_circles / circles_with_data * 100) if circles_with_data > 0 else 0
    
    st.write(f"Out of {total_circles} total circles:")
    
    if no_data_circles > 0:
        st.write(f"- {no_data_circles} circles ({100-data_pct:.1f}%) have no industry data and received a score of 0")
    
    st.write(f"- {circles_with_data} circles ({data_pct:.1f}%) have industry data")
    st.write(f"- Among circles with industry data, {diverse_circles} ({diverse_pct:.1f}%) contain members from multiple industry categories")
    
    # CRITICAL FIX: Update the session state with our modified circles_df that now has diversity scores
    st.session_state.matched_circles = circles_df
    print(f"INDUSTRY HISTOGRAM UPDATE - Updated session state matched_circles with calculated industry scores for {len(circles_df)} circles")

def render_employment_analysis(data):
    """Render the Employment analysis visualizations"""
    st.subheader("Employment Analysis")
    
    # Check if data is None
    if data is None:
        st.warning("No data available for analysis. Please upload participant data.")
        return
    
    # Create a copy to work with
    df = data.copy()
    
    # Debug: show column names
    st.caption("Debugging information:")
    with st.expander("Show data columns and sample values"):
        st.write("Available columns:")
        for col in df.columns:
            st.text(f"- {col}")
        
        # Try to find Employment Status column
        employment_status_col = None
        for col in df.columns:
            if "employment status" in col.lower():
                employment_status_col = col
                break
        
        if employment_status_col:
            st.write(f"Found Employment Status column: {employment_status_col}")
            # Show some sample values
            sample_values = df[employment_status_col].dropna().head(5).tolist()
            st.write(f"Sample values: {sample_values}")
        else:
            st.write("No Employment Status column found in the data")
    
    # Check if we need to create Employment Category
    if 'Employment_Category' not in df.columns:
        if employment_status_col:
            st.info(f"Creating Employment Category from {employment_status_col}...")
            
            # Define function to categorize employment status
            def categorize_employment(status):
                if pd.isna(status):
                    return None
                
                # Convert to string in case it's not
                status_str = str(status)
                
                # Apply categorization rules
                if "Employed full-time for wages" in status_str:
                    return "Employed full-time for wages"
                elif ("Self-Employed" in status_str or "Self-employed" in status_str) and "Employed full-time for wages" not in status_str:
                    return "Self-employed"
                else:
                    return "All Else"
            
            # Apply the categorization function
            df['Employment_Category'] = df[employment_status_col].apply(categorize_employment)
            
            # Update session state with the new Employment_Category
            if 'results' in st.session_state and st.session_state.results is not None:
                # Copy the newly created Employment_Category to the results DataFrame
                # First, create a dictionary mapping Encoded ID to Employment_Category
                emp_cat_mapping = dict(zip(df['Encoded ID'], df['Employment_Category']))
                
                # Then apply this mapping to the results DataFrame
                if 'Encoded ID' in st.session_state.results.columns:
                    st.session_state.results['Employment_Category'] = st.session_state.results['Encoded ID'].map(emp_cat_mapping)
                    st.info("Updated results data with Employment Categories")
        else:
            st.warning("Employment Status data is not available. Please ensure Employment Status data was included in the uploaded file.")
            return
    
    # Filter out rows with missing Employment Category
    df = df[df['Employment_Category'].notna()]
    
    if len(df) == 0:
        st.warning("No Employment Category data is available after filtering.")
        return
    
    # Define the proper order for Employment Categories
    employment_order = [
        "Employed full-time for wages", "Self-employed", "All Else"
    ]
    
    # FIRST: Display diversity within circles IF we have matched circles
    if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
        if not (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty):
            render_employment_diversity_histogram()
        else:
            st.info("Run the matching algorithm to see the Employment diversity within circles.")
    else:
        st.info("Run the matching algorithm to see the Employment diversity within circles.")
    
    # SECOND: Display Distribution of Employment
    st.subheader("Distribution of Employment")
    
    # Count by Employment Category
    employment_counts = df['Employment_Category'].value_counts().reindex(employment_order).fillna(0).astype(int)
    
    # Create a DataFrame for plotting
    employment_df = pd.DataFrame({
        'Employment Category': employment_counts.index,
        'Count': employment_counts.values
    })
    
    # Create histogram using plotly with Stanford cardinal red color
    fig = px.bar(
        employment_df,
        x='Employment Category',
        y='Count',
        title='Distribution of Employment Categories',
        text='Count',  # Display count values on bars
        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
    )
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': employment_order},
        xaxis_title="Employment Category",
        yaxis_title="Count of Participants"
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True, key="plot_803")
    
    # Create a breakdown by Status if Status column exists
    if 'Status' in df.columns:
        st.subheader("Employment by Status")
        
        # Create a crosstab of Employment Category vs Status
        status_employment = pd.crosstab(
            df['Employment_Category'], 
            df['Status'],
            rownames=['Employment Category'],
            colnames=['Status']
        ).reindex(employment_order)
        
        # Add a Total column
        status_employment['Total'] = status_employment.sum(axis=1)
        
        # Calculate percentages
        for col in status_employment.columns:
            if col != 'Total':
                status_employment[f'{col} %'] = (status_employment[col] / status_employment['Total'] * 100).round(1)
        
        # Reorder columns to group counts with percentages
        cols = []
        for status in status_employment.columns:
            if status != 'Total' and not status.endswith(' %'):
                cols.append(status)
                cols.append(f'{status} %')
        cols.append('Total')
        
        # Show the table
        st.dataframe(status_employment[cols], use_container_width=True)
        
        # Create a stacked bar chart
        fig = px.bar(
            status_employment.reset_index(),
            x='Employment Category',
            y=[col for col in status_employment.columns if col != 'Total' and not col.endswith(' %')],
            title='Employment Distribution by Status',
            barmode='stack',
            color_discrete_sequence=['#8C1515', '#2E2D29']  # Stanford colors
        )
        
        # Customize layout
        fig.update_layout(
            xaxis={'categoryorder': 'array', 'categoryarray': employment_order},
            xaxis_title="Employment Category",
            yaxis_title="Count of Participants",
            legend_title="Status"
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True, key="plot_855")


def render_industry_analysis(data):
    """Render the Industry analysis visualizations"""
    st.subheader("Industry Analysis")
    
    # Check if data is None
    if data is None:
        st.warning("No data available for analysis. Please upload participant data.")
        return
    
    # Create a copy to work with
    df = data.copy()
    
    # Debug: show column names
    st.caption("Debugging information:")
    with st.expander("Show data columns and sample values"):
        st.write("Available columns:")
        for col in df.columns:
            st.text(f"- {col}")
        
        # Try to find Industry Sector column
        industry_sector_col = None
        for col in df.columns:
            if "industry sector" in col.lower():
                industry_sector_col = col
                break
        
        if industry_sector_col:
            st.write(f"Found Industry Sector column: {industry_sector_col}")
            # Show some sample values
            sample_values = df[industry_sector_col].dropna().head(5).tolist()
            st.write(f"Sample values: {sample_values}")
        else:
            st.write("No Industry Sector column found in the data")
    
    # Check if we need to create Industry Category
    if 'Industry_Category' not in df.columns:
        if industry_sector_col:
            st.info(f"Creating Industry Category from {industry_sector_col}...")
            
            # Define function to categorize industry sector
            def categorize_industry(sector):
                if pd.isna(sector):
                    return None
                
                # Convert to string in case it's not
                sector_str = str(sector)
                
                # Apply categorization rules
                if "Technology" in sector_str:
                    return "Technology"
                elif "Consulting" in sector_str and "Technology" not in sector_str:
                    return "Consulting"
                elif any(term in sector_str for term in ["Finance", "Investment", "Private Equity"]) and \
                     "Technology" not in sector_str and "Consulting" not in sector_str:
                    return "Finance / Investment / Private Equity"
                else:
                    return "All Else"
            
            # Apply the categorization function
            df['Industry_Category'] = df[industry_sector_col].apply(categorize_industry)
            
            # Update session state with the new Industry_Category
            if 'results' in st.session_state and st.session_state.results is not None:
                # Copy the newly created Industry_Category to the results DataFrame
                # First, create a dictionary mapping Encoded ID to Industry_Category
                ind_cat_mapping = dict(zip(df['Encoded ID'], df['Industry_Category']))
                
                # Then apply this mapping to the results DataFrame
                if 'Encoded ID' in st.session_state.results.columns:
                    st.session_state.results['Industry_Category'] = st.session_state.results['Encoded ID'].map(ind_cat_mapping)
                    st.info("Updated results data with Industry Categories")
        else:
            st.warning("Industry Sector data is not available. Please ensure Industry Sector data was included in the uploaded file.")
            return
    
    # Filter out rows with missing Industry Category
    df = df[df['Industry_Category'].notna()]
    
    if len(df) == 0:
        st.warning("No Industry Category data is available after filtering.")
        return
    
    # Define the proper order for Industry Categories
    industry_order = [
        "Technology", "Consulting", "Finance / Investment / Private Equity", "All Else"
    ]
    
    # FIRST: Display diversity within circles IF we have matched circles
    if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
        if not (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty):
            render_industry_diversity_histogram()
        else:
            st.info("Run the matching algorithm to see the Industry diversity within circles.")
    else:
        st.info("Run the matching algorithm to see the Industry diversity within circles.")
    
    # SECOND: Display Distribution of Industry
    st.subheader("Distribution of Industry")
    
    # Count by Industry Category
    industry_counts = df['Industry_Category'].value_counts().reindex(industry_order).fillna(0).astype(int)
    
    # Create a DataFrame for plotting
    industry_df = pd.DataFrame({
        'Industry Category': industry_counts.index,
        'Count': industry_counts.values
    })
    
    # Create histogram using plotly with Stanford cardinal red color
    fig = px.bar(
        industry_df,
        x='Industry Category',
        y='Count',
        title='Distribution of Industry Categories',
        text='Count',  # Display count values on bars
        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
    )
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': industry_order},
        xaxis_title="Industry Category",
        yaxis_title="Count of Participants"
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True, key="plot_985")
    
    # Create a breakdown by Status if Status column exists
    if 'Status' in df.columns:
        st.subheader("Industry by Status")
        
        # Create a crosstab of Industry Category vs Status
        status_industry = pd.crosstab(
            df['Industry_Category'], 
            df['Status'],
            rownames=['Industry Category'],
            colnames=['Status']
        ).reindex(industry_order)
        
        # Add a Total column
        status_industry['Total'] = status_industry.sum(axis=1)
        
        # Calculate percentages
        for col in status_industry.columns:
            if col != 'Total':
                status_industry[f'{col} %'] = (status_industry[col] / status_industry['Total'] * 100).round(1)
        
        # Reorder columns to group counts with percentages
        cols = []
        for status in status_industry.columns:
            if status != 'Total' and not status.endswith(' %'):
                cols.append(status)
                cols.append(f'{status} %')
        cols.append('Total')
        
        # Show the table
        st.dataframe(status_industry[cols], use_container_width=True)
        
        # Create a stacked bar chart
        fig = px.bar(
            status_industry.reset_index(),
            x='Industry Category',
            y=[col for col in status_industry.columns if col != 'Total' and not col.endswith(' %')],
            title='Industry Distribution by Status',
            barmode='stack',
            color_discrete_sequence=['#8C1515', '#2E2D29']  # Stanford colors
        )
        
        # Customize layout
        fig.update_layout(
            xaxis={'categoryorder': 'array', 'categoryarray': industry_order},
            xaxis_title="Industry Category",
            yaxis_title="Count of Participants",
            legend_title="Status"
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True, key="plot_1037")

def render_class_vintage_analysis(data):
    """Render the Class Vintage analysis visualizations"""
    st.subheader("Class Vintage Analysis")
    
    # Check if data is None
    if data is None:
        st.warning("No data available for analysis. Please upload participant data.")
        return
    
    # Create a copy to work with
    df = data.copy()
    
    # Debug: show column names
    st.caption("Debugging information:")
    with st.expander("Show data columns and sample values"):
        st.write("Available columns:")
        for col in df.columns:
            st.text(f"- {col}")
        
        # Try to find GSB Class column
        gsb_class_col = None
        for col in df.columns:
            if any(term in col.lower().replace(" ", "") for term in ['gsbclass', 'gsb class']):
                gsb_class_col = col
                break
        
        if gsb_class_col:
            st.write(f"Found GSB Class column: {gsb_class_col}")
            # Show some sample values
            sample_values = df[gsb_class_col].dropna().head(5).tolist()
            st.write(f"Sample values: {sample_values}")
        else:
            st.write("No GSB Class column found in the data")
    
    # Check if we have Class Vintage data
    if 'Class_Vintage' not in data.columns:
        st.warning("Class Vintage data is not available. Please ensure GSB Class data was included in the uploaded file.")
        
        # Try to calculate Class Vintage on-the-fly if there's a GSB Class column
        if gsb_class_col:
            st.info(f"Attempting to calculate Class Vintage from {gsb_class_col}...")
            try:
                from modules.data_processor import calculate_class_vintage
                
                # Convert to numeric and calculate vintage
                df['GSB_Class_Numeric'] = pd.to_numeric(df[gsb_class_col], errors='coerce')
                df['Class_Vintage'] = df['GSB_Class_Numeric'].apply(calculate_class_vintage)
                
                vintage_counts = df['Class_Vintage'].value_counts()
                st.success(f"Successfully calculated Class Vintage for {len(df[df['Class_Vintage'].notna()])} records")
                st.write(f"Distribution: {vintage_counts}")
                
                # Update session state with the new Class_Vintage
                if 'results' in st.session_state and st.session_state.results is not None:
                    # First, create a dictionary mapping Encoded ID to Class_Vintage
                    vintage_mapping = dict(zip(df['Encoded ID'], df['Class_Vintage']))
                    
                    # Then apply this mapping to the results DataFrame
                    if 'Encoded ID' in st.session_state.results.columns:
                        st.session_state.results['Class_Vintage'] = st.session_state.results['Encoded ID'].map(vintage_mapping)
                        st.info("Updated results data with Class Vintage")
            except Exception as e:
                st.error(f"Error calculating Class Vintage: {str(e)}")
                return
        else:
            return
    
    # Filter out rows with missing Class Vintage
    df = df[df['Class_Vintage'].notna()]
    
    if len(df) == 0:
        st.warning("No Class Vintage data is available after filtering.")
        return
    
    # Define the proper order for Class Vintage categories
    vintage_order = [
        "01-10 yrs", "11-20 yrs", "21-30 yrs", 
        "31-40 yrs", "41-50 yrs", "51-60 yrs", "61+ yrs"
    ]
    
    # FIRST: Display diversity within circles IF we have matched circles
    if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
        if not (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty):
            render_vintage_diversity_histogram()
        else:
            st.info("Run the matching algorithm to see the Class Vintage diversity within circles.")
    else:
        st.info("Run the matching algorithm to see the Class Vintage diversity within circles.")
    
    # SECOND: Display Distribution of Class Vintage
    st.subheader("Distribution of Class Vintage")
    
    # Count by Class Vintage
    vintage_counts = df['Class_Vintage'].value_counts().reindex(vintage_order).fillna(0).astype(int)
    
    # Create a DataFrame for plotting
    vintage_df = pd.DataFrame({
        'Class Vintage': vintage_counts.index,
        'Count': vintage_counts.values
    })
    
    # Create histogram using plotly with Stanford cardinal red color
    fig = px.bar(
        vintage_df,
        x='Class Vintage',
        y='Count',
        title='Distribution of Class Vintage',
        text='Count',  # Display count values on bars
        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
    )
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': vintage_order},
        xaxis_title="Class Vintage (Years Since Graduation)",
        yaxis_title="Count of Participants"
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True, key="plot_1159")
    
    # Create a breakdown by Status
    if 'Status' in df.columns:
        st.subheader("Class Vintage by Status")
        
        # Create a crosstab of Class Vintage vs Status
        status_vintage = pd.crosstab(
            df['Class_Vintage'], 
            df['Status'],
            rownames=['Class Vintage'],
            colnames=['Status']
        ).reindex(vintage_order)
        
        # Add a Total column
        status_vintage['Total'] = status_vintage.sum(axis=1)
        
        # Calculate percentages
        for col in status_vintage.columns:
            if col != 'Total':
                status_vintage[f'{col} %'] = (status_vintage[col] / status_vintage['Total'] * 100).round(1)
        
        # Reorder columns to group counts with percentages
        cols = []
        for status in status_vintage.columns:
            if status != 'Total' and not status.endswith(' %'):
                cols.append(status)
                cols.append(f'{status} %')
        cols.append('Total')
        
        # Show the table
        st.dataframe(status_vintage[cols], use_container_width=True)
        
        # Create a stacked bar chart
        fig = px.bar(
            status_vintage.reset_index(),
            x='Class Vintage',
            y=[col for col in status_vintage.columns if col != 'Total' and not col.endswith(' %')],
            title='Class Vintage Distribution by Status',
            barmode='stack',
            color_discrete_sequence=['#8C1515', '#2E2D29']  # Stanford colors
        )
        
        # Customize layout
        fig.update_layout(
            xaxis={'categoryorder': 'array', 'categoryarray': vintage_order},
            xaxis_title="Class Vintage (Years Since Graduation)",
            yaxis_title="Count of Participants",
            legend_title="Status"
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True, key="plot_1211")
    
    # We have removed the "Class Vintage by Match Status", "Class Vintage Distribution by Match Status", 
    # and "Match Rate by Class Vintage" sections per user request

# East Bay debug tab function was removed to focus exclusively on Seattle test case

def render_metadata_debug_tab():
    """Render the metadata validation debug section"""
    st.subheader("Metadata Validation")
    st.write("This section validates the consistency of circle metadata across the application.")
    
    # Import needed utilities
    from utils.feature_flags import get_flag
    from utils.circle_metadata_manager import CircleMetadataManager, get_manager_from_session_state
    import pandas as pd
    import plotly.express as px
    
    # Check if metadata validation is enabled
    if get_flag('enable_metadata_validation'):
        # Create tabs for different validation aspects
        val_tab1, val_tab2, val_tab3, val_tab4 = st.tabs(["Metadata Sources", "Member List Format", "Max Additions", "Host Status"])
        
        # Get the CircleMetadataManager if available
        manager = get_manager_from_session_state(st.session_state) if 'circle_manager' in st.session_state else None
        
        # Function to get circle data sources for validation
        def get_circle_data_sources():
            sources = {
                'manager': {'available': False, 'circle_count': 0, 'circle_ids': []},
                'df': {'available': False, 'circle_count': 0, 'circle_ids': []}
            }
            
            # Check CircleMetadataManager availability
            if manager:
                sources['manager']['available'] = True
                sources['manager']['circle_count'] = len(manager.circles)
                sources['manager']['circle_ids'] = list(manager.circles.keys())
            
            # Check DataFrame availability
            if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
                circles_df = st.session_state.matched_circles
                sources['df']['available'] = True
                sources['df']['circle_count'] = len(circles_df)
                sources['df']['circle_ids'] = []
                if 'circle_id' in circles_df.columns:
                    sources['df']['circle_ids'] = circles_df['circle_id'].tolist()
            
            return sources
        
        # Get data sources
        sources = get_circle_data_sources()
        
        with val_tab1:
            st.write("#### Metadata Source Analysis")
            
            # Check if both sources are available
            if sources['manager']['available'] and sources['df']['available']:
                manager_count = sources['manager']['circle_count']
                df_count = sources['df']['circle_count']
                
                # Create a comparison
                comp_data = {
                    'Source': ['CircleMetadataManager', 'DataFrame'],
                    'Circle Count': [manager_count, df_count]
                }
                comp_df = pd.DataFrame(comp_data)
                
                # Create a bar chart
                fig = px.bar(
                    comp_df,
                    x='Source',
                    y='Circle Count',
                    title='Circle Count by Data Source',
                    color='Source',
                    color_discrete_sequence=['#8C1515', '#175E54']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Check metadata source field
                if 'matched_circles' in st.session_state:
                    circles_df = st.session_state.matched_circles
                    if 'metadata_source' in circles_df.columns:
                        # Get counts by source
                        source_counts = circles_df['metadata_source'].value_counts().reset_index()
                        source_counts.columns = ['Source', 'Count']
                        
                        # Calculate percentages
                        source_counts['Percentage'] = (source_counts['Count'] / len(circles_df) * 100).round(1)
                        
                        # Show the table
                        st.write("#### Metadata Source Field Values")
                        st.dataframe(source_counts, use_container_width=True)
                        
                        # Create a pie chart
                        fig = px.pie(
                            source_counts,
                            names='Source',
                            values='Count',
                            title='Metadata Source Distribution',
                            color_discrete_sequence=px.colors.qualitative.Dark24
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Check if optimizer metadata flag is aligned with the data
                        optimizer_count = source_counts[source_counts['Source'] == 'optimizer']['Count'].sum() if 'optimizer' in source_counts['Source'].values else 0
                        if get_flag("use_optimizer_metadata") and optimizer_count == 0:
                            st.warning("⚠️ 'use_optimizer_metadata' flag is enabled but no circles have 'optimizer' as the metadata source!")
                        elif not get_flag("use_optimizer_metadata") and optimizer_count > 0:
                            st.info("ℹ️ Some circles have 'optimizer' as the metadata source but 'use_optimizer_metadata' flag is disabled.")
                    else:
                        st.warning("⚠️ No 'metadata_source' column found in circles DataFrame.")
                
                # Check for consistency between manager and dataframe
                manager_ids = set(sources['manager']['circle_ids'])
                df_ids = set(sources['df']['circle_ids'])
                
                in_both = manager_ids.intersection(df_ids)
                only_manager = manager_ids - df_ids
                only_df = df_ids - manager_ids
                
                # Display Venn diagram data as metrics
                st.write("#### Data Source Consistency")
                cols = st.columns(3)
                cols[0].metric("Circles in both sources", len(in_both))
                cols[1].metric("Only in Manager", len(only_manager))
                cols[2].metric("Only in DataFrame", len(only_df))
                
                # Show detailed list of inconsistent circles if any
                if only_manager or only_df:
                    with st.expander("View Inconsistent Circle IDs"):
                        if only_manager:
                            st.write("**Circles only in Manager:**")
                            st.write(", ".join(sorted(list(only_manager))))
                        if only_df:
                            st.write("**Circles only in DataFrame:**")
                            st.write(", ".join(sorted(list(only_df))))
            else:
                if not sources['manager']['available']:
                    st.warning("⚠️ CircleMetadataManager is not available. Run the matching process first.")
                if not sources['df']['available']:
                    st.warning("⚠️ No circles DataFrame available. Run the matching process first.")
        
        with val_tab2:
            st.write("#### Member List Format Validation")
            
            if not manager:
                st.warning("⚠️ CircleMetadataManager is not available. Run the matching process first.")
                return
            
            # Validate member list formats
            format_types = {
                'list': 0,
                'string': 0,
                'dict': 0,
                'other': 0,
                'missing': 0,
                'empty': 0
            }
            
            member_count_validation = {
                'match': 0,
                'mismatch': 0,
                'missing_count': 0
            }
            
            problematic_circles = []
            
            # Analyze each circle
            for circle_id, circle_data in manager.circles.items():
                # Check member list format
                if 'members' in circle_data:
                    members = circle_data['members']
                    
                    if isinstance(members, list):
                        format_types['list'] += 1
                        if len(members) == 0:
                            format_types['empty'] += 1
                            problematic_circles.append((circle_id, 'empty member list'))
                    elif isinstance(members, str):
                        format_types['string'] += 1
                        problematic_circles.append((circle_id, 'string member format'))
                    elif isinstance(members, dict):
                        format_types['dict'] += 1
                        problematic_circles.append((circle_id, 'dict member format'))
                    else:
                        format_types['other'] += 1
                        problematic_circles.append((circle_id, f'unexpected type: {type(members)}'))
                else:
                    format_types['missing'] += 1
                    problematic_circles.append((circle_id, 'missing members field'))
                
                # Validate member count consistency
                if 'members' in circle_data and 'member_count' in circle_data:
                    members = circle_data['members']
                    member_count = circle_data['member_count']
                    
                    if isinstance(members, list):
                        actual_count = len(members)
                        if actual_count == member_count:
                            member_count_validation['match'] += 1
                        else:
                            member_count_validation['mismatch'] += 1
                            problematic_circles.append((circle_id, f'member count mismatch: {member_count} vs {actual_count}'))
                elif 'member_count' not in circle_data:
                    member_count_validation['missing_count'] += 1
                    problematic_circles.append((circle_id, 'missing member_count field'))
            
            # Create metrics
            cols = st.columns(2)
            total_circles = len(manager.circles)
            cols[0].metric("Total Circles", total_circles)
            cols[1].metric("Standardized List Format", format_types['list'], 
                        delta=format_types['list']-total_circles, delta_color="inverse")
            
            # Create bar chart of format types
            format_df = pd.DataFrame({
                'Format': list(format_types.keys()),
                'Count': list(format_types.values())
            })
            
            # Add percentage
            format_df['Percentage'] = (format_df['Count'] / total_circles * 100).round(1)
            
            # Create chart
            fig = px.bar(
                format_df,
                x='Format',
                y='Count',
                title='Member List Format Distribution',
                color='Format',
                color_discrete_sequence=['#8C1515', '#175E54', '#B83A4B', '#820000', '#D2C295', '#3B7EA1']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Member count validation
            st.write("#### Member Count Validation")
            count_validation_df = pd.DataFrame({
                'Status': ['Matches', 'Mismatches', 'Missing Count Field'],
                'Count': [member_count_validation['match'], 
                         member_count_validation['mismatch'],
                         member_count_validation['missing_count']]
            })
            
            # Add percentage
            count_validation_df['Percentage'] = (count_validation_df['Count'] / total_circles * 100).round(1)
            
            # Create chart
            fig = px.pie(
                count_validation_df,
                names='Status',
                values='Count',
                title='Member Count Validation',
                color_discrete_sequence=['#175E54', '#8C1515', '#D2C295']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show problematic circles if any
            if problematic_circles:
                st.write("#### Problematic Circles")
                problem_df = pd.DataFrame(problematic_circles, columns=['Circle ID', 'Issue'])
                st.dataframe(problem_df, use_container_width=True)
            else:
                st.success("✅ No member list format issues detected!")
                
            # Check feature flag consistency
            if format_types['string'] > 0 or format_types['dict'] > 0 or format_types['other'] > 0:
                if get_flag('use_standardized_member_lists'):
                    st.info("ℹ️ 'use_standardized_member_lists' flag is enabled but non-standard formats still exist. These are likely being normalized at access time.")
                else:
                    st.warning("⚠️ Non-standard member list formats exist but 'use_standardized_member_lists' flag is disabled!")
        
        with val_tab3:
            st.write("#### Max Additions Validation")
            
            if not manager:
                st.warning("⚠️ CircleMetadataManager is not available. Run the matching process first.")
                return
            
            # Analyze max_additions values
            max_add_stats = {
                'present': 0,
                'missing': 0,
                'zero': 0,
                'negative': 0,
                'by_value': {}
            }
            
            max_add_problems = []
            
            # Analyze each circle
            for circle_id, circle_data in manager.circles.items():
                if 'max_additions' in circle_data:
                    max_add_stats['present'] += 1
                    max_add = circle_data['max_additions']
                    
                    # Count by value
                    max_add_stats['by_value'][max_add] = max_add_stats['by_value'].get(max_add, 0) + 1
                    
                    # Check special cases
                    if max_add == 0:
                        max_add_stats['zero'] += 1
                    elif max_add < 0:
                        max_add_stats['negative'] += 1
                        max_add_problems.append((circle_id, f'negative max_additions: {max_add}'))
                else:
                    max_add_stats['missing'] += 1
                    max_add_problems.append((circle_id, 'missing max_additions field'))
            
            # Create metrics
            cols = st.columns(3)
            total_circles = len(manager.circles)
            cols[0].metric("Total Circles", total_circles)
            cols[1].metric("Circles with Max Additions", max_add_stats['present'])
            cols[2].metric("Circles Closed to New Members", max_add_stats['zero'],
                      help="Circles with max_additions=0")
            
            # Create histogram of max_additions values
            if max_add_stats['by_value']:
                st.write("#### Distribution of Max Additions Values")
                max_add_items = sorted(max_add_stats['by_value'].items())
                max_add_df = pd.DataFrame(max_add_items, columns=['Max Additions', 'Count'])
                
                # Create chart
                fig = px.bar(
                    max_add_df,
                    x='Max Additions',
                    y='Count',
                    title='Max Additions Distribution',
                    color_discrete_sequence=['#8C1515']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate closed percentage
                closed_pct = (max_add_stats['zero'] / max_add_stats['present']) * 100 if max_add_stats['present'] > 0 else 0
                st.write(f"🔍 {max_add_stats['zero']} out of {max_add_stats['present']} ({closed_pct:.1f}%) circles are closed to new members.")
            
            # Show problems
            if max_add_problems:
                st.write("#### Problematic Max Additions Values")
                problem_df = pd.DataFrame(max_add_problems, columns=['Circle ID', 'Issue'])
                st.dataframe(problem_df, use_container_width=True)
            else:
                st.success("✅ No max_additions issues detected!")
        
        with val_tab4:
            st.write("#### Host Status Validation")
            
            if not manager:
                st.warning("⚠️ CircleMetadataManager is not available. Run the matching process first.")
                return
            
            # Analyze host status values
            host_stats = {
                'ALWAYS': 0,
                'SOMETIMES': 0,
                'NEVER': 0,
                'missing': 0,
                'non_standard': 0
            }
            
            non_standard_values = {}
            host_problems = []
            
            # Analyze each circle
            for circle_id, circle_data in manager.circles.items():
                if 'host_status' in circle_data:
                    status = circle_data['host_status']
                    if status in ['ALWAYS', 'SOMETIMES', 'NEVER']:
                        host_stats[status] += 1
                    else:
                        host_stats['non_standard'] += 1
                        non_standard_values[status] = non_standard_values.get(status, 0) + 1
                        host_problems.append((circle_id, f'non-standard host status: {status}'))
                else:
                    host_stats['missing'] += 1
                    host_problems.append((circle_id, 'missing host_status field'))
            
            # Create metrics
            cols = st.columns(3)
            total_circles = len(manager.circles)
            cols[0].metric("Total Circles", total_circles)
            cols[1].metric("With Standard Host Status", host_stats['ALWAYS'] + host_stats['SOMETIMES'] + host_stats['NEVER'])
            cols[2].metric("Issues", host_stats['missing'] + host_stats['non_standard'], 
                       delta=host_stats['missing'] + host_stats['non_standard'], delta_color="inverse")
            
            # Create host status distribution chart
            host_df = pd.DataFrame({
                'Host Status': list(host_stats.keys()),
                'Count': list(host_stats.values())
            })
            
            # Create chart
            fig = px.bar(
                host_df,
                x='Host Status',
                y='Count',
                title='Host Status Distribution',
                color='Host Status',
                color_discrete_sequence=['#175E54', '#8C1515', '#B83A4B', '#820000', '#D2C295']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show non-standard values if any
            if non_standard_values:
                st.write("#### Non-Standard Host Status Values")
                non_std_items = sorted(non_standard_values.items(), key=lambda x: x[1], reverse=True)
                non_std_df = pd.DataFrame(non_std_items, columns=['Value', 'Count'])
                st.dataframe(non_std_df, use_container_width=True)
            
            # Show problematic circles if any
            if host_problems:
                st.write("#### Circles with Host Status Issues")
                problem_df = pd.DataFrame(host_problems, columns=['Circle ID', 'Issue'])
                st.dataframe(problem_df, use_container_width=True)
            else:
                st.success("✅ All circles have standard host status values!")
            
            # Show original vs normalized host status mapping if debug flag is enabled
            if get_flag('debug_data_standardization'):
                from utils.data_standardization import get_host_standardization_mapping
                
                st.write("#### Host Status Normalization Mapping")
                mapping = get_host_standardization_mapping()
                
                # Create a dataframe for the mapping
                mapping_items = [(k, v) for k, v in mapping.items()]
                mapping_df = pd.DataFrame(mapping_items, columns=['Original Value', 'Normalized Value'])
                
                # Sort by normalized value, then original
                mapping_df = mapping_df.sort_values(['Normalized Value', 'Original Value'])
                
                # Display as a table
                st.dataframe(mapping_df, use_container_width=True)
            
            # Check feature flag consistency
            if host_stats['non_standard'] > 0:
                if get_flag('use_standardized_host_status'):
                    st.info("ℹ️ 'use_standardized_host_status' flag is enabled but non-standard values still exist. These are likely being normalized at access time.")
                else:
                    st.warning("⚠️ Non-standard host status values exist but 'use_standardized_host_status' flag is disabled!")
    else:
        # Metadata validation is disabled
        st.info("Enable metadata validation in the Feature Flags section to see validation results.")
        st.write("This will help diagnose issues with circle metadata processing and display.")
        
        # Add toggle button for quick enable
        if st.button("Enable Metadata Validation"):
            from utils.feature_flags import set_flag
            set_flag('enable_metadata_validation', True)
            st.success("✅ Metadata validation enabled! Refresh this section to see results.")
            

def render_debug_tab():
    """Render the debug tab content"""
    st.subheader("Debug Information")
    
    # Add Feature Flags section
    from utils.feature_flags import render_debug_flags
    render_debug_flags()
    
    # Add a comprehensive log viewer with copy button
    st.write("## Optimization Debug Logs")
    
    # Create tabs for different sections of debug info
    debug_tab1, debug_tab2, debug_tab3, debug_tab4, debug_tab5, debug_tab6 = st.tabs([
        "Circle Capacity Analysis", 
        "Houston Debug Logs", 
        "Circle Eligibility Debug", 
        "All Circles Debug",
        "Seattle Compatibility Analysis",
        "Metadata Debug"
    ])
    
    with debug_tab1:
        st.write("### Circle Capacity Analysis")
        st.write("This section shows why circles with capacity might not be receiving new members")
        
        # CRITICAL DEBUG: Add bridge from eligibility logs to capacity debug
        if 'circle_eligibility_logs' in st.session_state and st.session_state.circle_eligibility_logs:
            print(f"\n🔍 BRIDGE DEBUG: Checking circle_eligibility_logs vs circle_capacity_debug")
            elig_logs = st.session_state.circle_eligibility_logs
            print(f"  Found {len(elig_logs)} circles in eligibility logs")
            
            # Count circles with capacity in eligibility logs
            circles_with_capacity = [c_id for c_id, data in elig_logs.items() 
                                   if data.get('max_additions', 0) > 0]
            print(f"  Found {len(circles_with_capacity)} circles with max_additions > 0 in eligibility logs")
            if circles_with_capacity:
                print(f"  Sample circles with capacity: {circles_with_capacity[:5]}...")
                
            # Create circle_capacity_debug if it doesn't exist
            if 'circle_capacity_debug' not in st.session_state:
                st.session_state.circle_capacity_debug = {}
                
            # Check existing capacity debug data
            capacity_debug = st.session_state.circle_capacity_debug
            print(f"  Found {len(capacity_debug)} circles in capacity_debug")
            
            # Find circles missing from capacity debug
            missing_circles = [c_id for c_id in circles_with_capacity if c_id not in capacity_debug]
            if missing_circles:
                print(f"  ⚠️ Found {len(missing_circles)} circles with capacity missing from capacity_debug")
                print(f"  Missing circles: {missing_circles[:10]}...")
                
                # CRITICAL FIX: Add missing circles to capacity_debug
                print(f"  🔧 Adding {len(missing_circles)} missing circles to capacity_debug")
                for c_id in missing_circles:
                    circle_data = elig_logs[c_id]
                    st.session_state.circle_capacity_debug[c_id] = {
                        'circle_id': c_id,
                        'region': circle_data.get('region', 'Unknown'),
                        'subregion': circle_data.get('subregion', 'Unknown'),
                        'meeting_time': circle_data.get('meeting_time', 'Unknown'),
                        'current_members': circle_data.get('current_members', 0),
                        'max_additions': circle_data.get('max_additions', 0),
                        'viable': True,  # Mark as viable since it has capacity
                        'is_test_circle': circle_data.get('is_test_circle', False),
                        'special_handling': circle_data.get('circle_id') in ['IP-SIN-01', 'IP-LON-04', 'IP-HOU-02']
                    }
        
        if 'circle_capacity_debug' in st.session_state and st.session_state.circle_capacity_debug:
            # Create a DataFrame from the circle capacity debug info
            capacity_data = list(st.session_state.circle_capacity_debug.values())
            capacity_df = pd.DataFrame(capacity_data)
            
            # Add debug logging after creating the DataFrame
            print(f"\n🔍 CAPACITY DF DEBUG: Created DataFrame with {len(capacity_df)} rows")
            if not capacity_df.empty:
                print(f"  Column names: {list(capacity_df.columns)}")
                if 'max_additions' in capacity_df.columns:
                    max_add_counts = capacity_df['max_additions'].value_counts().to_dict()
                    print(f"  max_additions distribution: {max_add_counts}")
                    circles_with_capacity = (capacity_df['max_additions'] > 0).sum()
                    print(f"  Circles with max_additions > 0: {circles_with_capacity}")
            
            # Add columns for clarity
            if not capacity_df.empty:
                st.write(f"Found {len(capacity_df)} circles with capacity for new members")
                
                # Highlight test circles
                capacity_df['test_circle'] = capacity_df['is_test_circle'].apply(
                    lambda x: "✅ YES" if x else "NO")
                capacity_df['special_handling'] = capacity_df['special_handling'].apply(
                    lambda x: "✅ YES" if x else "NO")
                capacity_df['selected_for_optimization'] = capacity_df['viable'].apply(
                    lambda x: "✅ YES" if x else "❌ NO")
                
                # Add column for small circles
                if 'current_members' in capacity_df.columns:
                    capacity_df['small_circle'] = capacity_df['current_members'].apply(
                        lambda x: "✅ YES (<5 members)" if x < 5 else "NO")
                else:
                    capacity_df['small_circle'] = "NO"
                
                # Reorder columns for better readability
                display_cols = ['circle_id', 'region', 'subregion', 'meeting_time', 
                                'current_members', 'max_additions', 'small_circle',
                                'selected_for_optimization', 'test_circle', 'special_handling']
                
                # Create tabs to show different views of the circles
                circle_capacity_tab1, circle_capacity_tab2 = st.tabs(["All Circles", "Selected for Optimization"])
                
                with circle_capacity_tab1:
                    # Display all circles with capacity
                    st.dataframe(capacity_df[display_cols])
                    
                with circle_capacity_tab2:
                    # Only show circles selected for optimization
                    viable_df = capacity_df[capacity_df['viable'] == True]
                    if not viable_df.empty:
                        st.write(f"Found {len(viable_df)} circles that will be considered in optimization")
                        st.dataframe(viable_df[display_cols])
                    else:
                        st.warning("No circles are currently selected for optimization!")
                
                # Provide summary statistics
                st.write("### Summary Statistics")
                viable_count = capacity_df['viable'].sum()
                total_count = len(capacity_df)
                test_count = capacity_df['is_test_circle'].sum()
                special_count = capacity_df['special_handling'].sum()
                small_count = (capacity_df['current_members'] < 5).sum() if 'current_members' in capacity_df.columns else 0
                
                st.write(f"- Total circles with capacity: {total_count}")
                st.write(f"- Circles selected for optimization: {viable_count} ({viable_count/total_count*100:.1f}%)")
                st.write(f"- Small circles (<5 members): {small_count}")
                st.write(f"- Test circles: {test_count}")
                st.write(f"- Circles with special handling: {special_count}")
                
                # Create a text version for the copy button
                capacity_text = "CIRCLE CAPACITY ANALYSIS\n\n"
                capacity_text += f"Total circles with capacity: {total_count}\n"
                capacity_text += f"Circles selected for optimization: {viable_count} ({viable_count/total_count*100:.1f}%)\n"
                capacity_text += f"Test circles: {test_count}\n"
                capacity_text += f"Circles with special handling: {special_count}\n\n"
                
                capacity_text += "CIRCLES WITH CAPACITY:\n"
                for _, row in capacity_df.iterrows():
                    capacity_text += f"- {row['circle_id']} (Region: {row['region']}, Time: {row['meeting_time']})\n"
                    capacity_text += f"  Current members: {row['current_members']}, Max additions: {row['max_additions']}\n"
                    capacity_text += f"  Selected for optimization: {'YES' if row['viable'] else 'NO'}\n"
                    capacity_text += f"  Test circle: {'YES' if row['is_test_circle'] else 'NO'}\n"
                    capacity_text += f"  Special handling: {'YES' if row['special_handling'] else 'NO'}\n\n"
                
                # Create a copy button
                st.text_area("Copy this text to share capacity analysis", capacity_text, height=300)
                
                # Add JavaScript to handle copying to clipboard
                st.markdown("""
                <button onclick="navigator.clipboard.writeText(document.querySelector('textarea').value)">
                    📋 Copy Capacity Analysis to Clipboard
                </button>
                """, unsafe_allow_html=True)
        else:
            st.info("No circle capacity debug information available. Run the optimization to generate this data.")
    
    with debug_tab2:
        st.write("### Houston Debug Logs")
        
        if 'houston_debug_logs' in st.session_state and st.session_state.houston_debug_logs:
            logs = st.session_state.houston_debug_logs
            logs_text = "\n".join(logs)
            
            st.text_area("Houston Debug Logs", logs_text, height=400)
            
            # Add JavaScript to handle copying to clipboard
            st.markdown("""
            <button onclick="navigator.clipboard.writeText(document.querySelectorAll('textarea')[1].value)">
                📋 Copy Houston Debug Logs to Clipboard
            </button>
            """, unsafe_allow_html=True)
        else:
            st.info("No Houston debug logs available. Run the optimization to generate logs.")
    
    with debug_tab3:
        st.write("### Circle Eligibility Debug")
        st.write("This section shows detailed analysis of circle eligibility for new member optimization")
        
        # Add a button to manually test file backup functionality
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Test File Backup"):
                try:
                    from modules.optimizer_new import save_circle_eligibility_logs_to_file
                    
                    # Create test circle eligibility logs
                    test_logs = {
                        'IP-TEST-01': {
                            'circle_id': 'IP-TEST-01',
                            'region': 'Test Region',
                            'subregion': 'Test Subregion',
                            'is_eligible': True,
                            'current_members': 7,
                            'max_additions': 3,
                            'is_small_circle': False,
                            'is_test_circle': True,
                            'has_none_preference': False,
                            'preference_overridden': False,
                            'meeting_time': 'Monday (Evening)',
                            'reason': 'Has capacity'
                        },
                        'IP-TEST-02': {
                            'circle_id': 'IP-TEST-02',
                            'region': 'Test Region',
                            'subregion': 'Test Subregion',
                            'is_eligible': False,
                            'current_members': 10,
                            'max_additions': 0,
                            'is_small_circle': False,
                            'is_test_circle': True,
                            'has_none_preference': True,
                            'preference_overridden': False,
                            'reason': 'Circle is at maximum capacity (10 members)',
                            'meeting_time': 'Wednesday (Evening)'
                        },
                        'IP-TEST-03': {
                            'circle_id': 'IP-TEST-03',
                            'region': 'Test Region',
                            'subregion': 'Test Subregion',
                            'is_eligible': True,
                            'current_members': 4,
                            'max_additions': 6,
                            'is_small_circle': True,
                            'is_test_circle': True,
                            'has_none_preference': True,
                            'preference_overridden': True,
                            'override_reason': 'Small circle override applied',
                            'meeting_time': 'Friday (Evening)',
                            'reason': 'Small circle needs to reach viable size'
                        }
                    }
                    
                    # Save to file for testing
                    saved = save_circle_eligibility_logs_to_file(test_logs, "Test Region")
                    if saved:
                        st.success(f"✅ Successfully saved {len(test_logs)} test logs to file")
                        # Update session state with the test logs
                        st.session_state.circle_eligibility_logs = test_logs
                        st.rerun()  # Refresh the page to see the results
                    else:
                        st.error("❌ Failed to save test logs to file")
                except Exception as e:
                    st.error(f"❌ Error during file operations: {str(e)}")
        
        with col2:
            if st.button("Load From File"):
                try:
                    from modules.optimizer_new import load_circle_eligibility_logs_from_file
                    file_logs = load_circle_eligibility_logs_from_file()
                    if file_logs and len(file_logs) > 0:
                        st.success(f"✅ Loaded {len(file_logs)} logs from file")
                        st.session_state.circle_eligibility_logs = file_logs
                        st.rerun()  # Refresh the page to see the results
                    else:
                        st.warning("No logs found in file")
                except Exception as e:
                    st.error(f"❌ Error loading from file: {str(e)}")
        
        # CRITICAL FIX: Improved debugging for session state logs
        if 'circle_eligibility_logs' in st.session_state:
            # Add diagnostic info about the logs in session state
            print(f"🔍 DEBUG UI: Found circle_eligibility_logs in session state with {len(st.session_state.circle_eligibility_logs)} entries")
            if len(st.session_state.circle_eligibility_logs) > 0:
                print(f"🔍 DEBUG UI: Session state contains logs for circle IDs: {list(st.session_state.circle_eligibility_logs.keys())[:5]}...")
                # Convert logs to a DataFrame for display
                eligibility_data = list(st.session_state.circle_eligibility_logs.values())
                print(f"🔍 DEBUG UI: Converted {len(eligibility_data)} log entries to a list for DataFrame creation")
                eligibility_df = pd.DataFrame(eligibility_data)
                print(f"🔍 DEBUG UI: Created DataFrame with shape {eligibility_df.shape}")
            else:
                st.warning("Circle eligibility logs dictionary exists in session state but has no entries.")
                print("⚠️ WARNING: circle_eligibility_logs exists in session state but is empty!")
                
                # Try to load logs from file as a backup
                try:
                    from modules.optimizer_new import load_circle_eligibility_logs_from_file
                    print("🔄 Attempting to load circle eligibility logs from file...")
                    
                    file_logs = load_circle_eligibility_logs_from_file()
                    if file_logs and len(file_logs) > 0:
                        print(f"✅ Successfully loaded {len(file_logs)} log entries from file")
                        eligibility_data = list(file_logs.values())
                        eligibility_df = pd.DataFrame(eligibility_data)
                        
                        # Update session state with these logs for consistency
                        st.session_state.circle_eligibility_logs = file_logs
                        st.success(f"Loaded {len(file_logs)} circle eligibility logs from file backup")
                    else:
                        print("❌ No logs found in file backup")
                        eligibility_df = pd.DataFrame()
                except Exception as e:
                    print(f"❌ ERROR loading logs from file: {str(e)}")
                    eligibility_df = pd.DataFrame()
            
            if not eligibility_df.empty:
                # Overview metrics
                col1, col2, col3 = st.columns(3)
                total_circles = len(eligibility_df)
                
                with col1:
                    if 'is_eligible' in eligibility_df.columns:
                        eligible_count = eligibility_df['is_eligible'].sum()
                        st.metric("Eligible Circles", f"{eligible_count} / {total_circles}", 
                                 f"{eligible_count/total_circles:.1%}")
                
                with col2:
                    if 'is_small_circle' in eligibility_df.columns:
                        small_circles_count = eligibility_df['is_small_circle'].sum()
                        st.metric("Small Circles (<5 members)", f"{small_circles_count}",
                                 f"{small_circles_count/total_circles:.1%}")
                
                with col3:
                    if 'has_none_preference' in eligibility_df.columns:
                        none_pref_count = eligibility_df['has_none_preference'].sum()
                        st.metric("'None' Preference Circles", f"{none_pref_count}",
                                 f"{none_pref_count/total_circles:.1%}")
                
                st.write(f"Circle eligibility analysis for {len(eligibility_df)} circles:")
                
                # Format the display columns
                if 'is_eligible' in eligibility_df.columns:
                    eligibility_df['eligible_status'] = eligibility_df['is_eligible'].apply(
                        lambda x: "✅ Eligible" if x else "❌ Not Eligible")
                
                if 'is_test_circle' in eligibility_df.columns:
                    eligibility_df['test_circle'] = eligibility_df['is_test_circle'].apply(
                        lambda x: "✅ YES" if x else "NO")
                
                # Add tabs to separate different views
                overview_tab, eligible_tab, ineligible_tab, test_circles_tab = st.tabs([
                    "All Circles Overview", "Eligible Circles", "Ineligible Circles", "Test Circles"
                ])
                
                with overview_tab:
                    st.write(f"### Circle Capacity Overview ({len(eligibility_df)} total circles)")
                    
                    # Add small circle flag if not exists
                    if 'is_small_circle' not in eligibility_df.columns and 'current_members' in eligibility_df.columns:
                        eligibility_df['is_small_circle'] = eligibility_df['current_members'] < 5
                    
                    if 'is_small_circle' in eligibility_df.columns:
                        eligibility_df['small_circle'] = eligibility_df['is_small_circle'].apply(
                            lambda x: "✅ YES (<5 members)" if x else "NO")
                    
                    # Add preference status if possible
                    if 'has_none_preference' in eligibility_df.columns and 'preference_overridden' in eligibility_df.columns:
                        # Create a combined preference column for better readability
                        eligibility_df['preference_status'] = 'Normal'
                        
                        # Set for circles with None preference
                        mask_none_not_overridden = (eligibility_df['has_none_preference'] == True) & (eligibility_df['preference_overridden'] == False)
                        eligibility_df.loc[mask_none_not_overridden, 'preference_status'] = "❌ None (honored)"
                        
                        # Set for circles with None preference that was overridden
                        mask_none_overridden = (eligibility_df['has_none_preference'] == True) & (eligibility_df['preference_overridden'] == True)
                        eligibility_df.loc[mask_none_overridden, 'preference_status'] = "⚠️ None (overridden)"
                    
                    # Display columns for all circles
                    overview_cols = ['circle_id', 'region', 'eligible_status', 'current_members', 
                                    'max_additions', 'small_circle', 'test_circle']
                    
                    # Add preference status if available
                    if 'preference_status' in eligibility_df.columns:
                        overview_cols.append('preference_status')
                    elif 'has_none_preference' in eligibility_df.columns:
                        overview_cols.append('has_none_preference')
                    
                    # Add reason column if available
                    if 'reason' in eligibility_df.columns:
                        overview_cols.append('reason')
                    elif 'override_reason' in eligibility_df.columns:
                        overview_cols.append('override_reason')
                    
                    # Filter to columns that exist
                    overview_cols = [col for col in overview_cols if col in eligibility_df.columns]
                    
                    # Sort by region and circle ID for better readability
                    if 'region' in eligibility_df.columns and 'circle_id' in eligibility_df.columns:
                        display_df = eligibility_df.sort_values(['region', 'circle_id'])[overview_cols].reset_index(drop=True)
                    else:
                        display_df = eligibility_df[overview_cols].reset_index(drop=True)
                    
                    # Display the dataframe
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Group by region to see distribution
                    if 'region' in eligibility_df.columns:
                        st.write("### Circles by Region")
                        region_df = pd.DataFrame()
                        
                        # Total circles by region
                        region_counts = eligibility_df['region'].value_counts().reset_index()
                        region_counts.columns = ['Region', 'Total Circles']
                        region_df['Region'] = region_counts['Region']
                        region_df['Total Circles'] = region_counts['Total Circles']
                        
                        # Eligible circles by region
                        if 'is_eligible' in eligibility_df.columns:
                            eligible_by_region = eligibility_df[eligibility_df['is_eligible']].groupby('region').size()
                            eligible_by_region = eligible_by_region.reindex(region_df['Region']).fillna(0).astype(int)
                            region_df['Eligible Circles'] = eligible_by_region.values
                            region_df['Eligible %'] = (region_df['Eligible Circles'] / region_df['Total Circles']).apply(lambda x: f"{x:.1%}")
                        
                        st.dataframe(region_df, use_container_width=True)
                
                with eligible_tab:
                    # Show only eligible circles
                    if 'is_eligible' in eligibility_df.columns:
                        eligible_df = eligibility_df[eligibility_df['is_eligible']].reset_index(drop=True)
                        
                        if len(eligible_df) > 0:
                            st.write(f"Found {len(eligible_df)} circles with capacity for new members")
                            
                            # Display columns
                            display_cols = ['circle_id', 'region', 'subregion', 'meeting_time', 
                                            'current_members', 'max_additions', 'test_circle']
                            
                            # Add special columns if available
                            for col in ['original_preference', 'override_reason', 'preference_value']:
                                if col in eligible_df.columns and col not in display_cols:
                                    display_cols.append(col)
                            
                            # Show only columns that exist
                            display_cols = [col for col in display_cols if col in eligible_df.columns]
                            
                            # Display the DataFrame
                            st.dataframe(eligible_df[display_cols])
                            
                            # Group by region to see distribution
                            if 'region' in eligible_df.columns:
                                st.write("### Eligible Circles by Region")
                                region_counts = eligible_df['region'].value_counts().reset_index()
                                region_counts.columns = ['Region', 'Count']
                                st.dataframe(region_counts)
                            
                            # Look for test circles with special handling
                            test_circles = eligible_df[eligible_df['is_test_circle'] == True]
                            if len(test_circles) > 0:
                                st.write(f"### Test Circles ({len(test_circles)})")
                                st.dataframe(test_circles[display_cols])
                            
                            # Find circles with override reason (universal fix applied)
                            if 'override_reason' in eligible_df.columns:
                                small_circles = eligible_df[eligible_df['override_reason'].str.contains('small circle', case=False, na=False)]
                                if len(small_circles) > 0:
                                    st.write(f"### Small Circles With Universal Fix Applied ({len(small_circles)})")
                                    st.dataframe(small_circles[display_cols])
                        else:
                            st.info("No eligible circles found.")
                    else:
                        st.warning("Eligibility status not available in data.")
                
                with ineligible_tab:
                    # Show only ineligible circles
                    if 'is_eligible' in eligibility_df.columns:
                        ineligible_df = eligibility_df[~eligibility_df['is_eligible']].reset_index(drop=True)
                        
                        if len(ineligible_df) > 0:
                            st.write(f"Found {len(ineligible_df)} circles that are NOT eligible for new members")
                            
                            # Display columns
                            display_cols = ['circle_id', 'region', 'subregion', 'meeting_time', 
                                            'current_members', 'max_additions', 'test_circle']
                            
                            # Add reason column if available
                            if 'reason' in ineligible_df.columns:
                                display_cols.append('reason')
                            
                            # Add original preference if available
                            if 'original_preference' in ineligible_df.columns:
                                display_cols.append('original_preference')
                            
                            # Show only columns that exist
                            display_cols = [col for col in display_cols if col in ineligible_df.columns]
                            
                            # Display the DataFrame
                            st.dataframe(ineligible_df[display_cols])
                            
                            # Group by reason if available
                            if 'reason' in ineligible_df.columns:
                                st.write("### Reasons for Ineligibility")
                                reason_counts = ineligible_df['reason'].value_counts().reset_index()
                                reason_counts.columns = ['Reason', 'Count']
                                st.dataframe(reason_counts)
                            
                            # Find small circles that could benefit from universal fix
                            if 'current_members' in ineligible_df.columns:
                                small_circles = ineligible_df[ineligible_df['current_members'] < 5]
                                if len(small_circles) > 0:
                                    st.write(f"### Small Ineligible Circles (Under 5 Members)")
                                    st.write(f"These circles could potentially benefit from the universal fix:")
                                    st.dataframe(small_circles[display_cols])
                        else:
                            st.success("All circles are eligible for new members.")
                    else:
                        st.warning("Eligibility status not available in data.")
                
                with test_circles_tab:
                    # Focus only on test circles
                    if 'is_test_circle' in eligibility_df.columns:
                        test_df = eligibility_df[eligibility_df['is_test_circle'] == True].reset_index(drop=True)
                        
                        if len(test_df) > 0:
                            st.write(f"### Test Circles Analysis ({len(test_df)} circles)")
                            st.write("Special test circles with controlled behavior for testing the algorithm.")
                            
                            # Display columns with comprehensive information
                            display_cols = ['circle_id', 'region', 'eligible_status', 'current_members', 
                                           'max_additions', 'small_circle']
                            
                            # Add preference information
                            if 'preference_status' in test_df.columns:
                                display_cols.append('preference_status')
                            elif 'has_none_preference' in test_df.columns:
                                display_cols.append('has_none_preference')
                            
                            if 'preference_overridden' in test_df.columns:
                                display_cols.append('preference_overridden') 
                                
                            # Add reason/override information
                            if 'reason' in test_df.columns:
                                display_cols.append('reason')
                            if 'override_reason' in test_df.columns:
                                display_cols.append('override_reason')
                                
                            # Show only columns that exist
                            display_cols = [col for col in display_cols if col in test_df.columns]
                            
                            # Display the DataFrame
                            st.dataframe(test_df[display_cols], use_container_width=True)
                            
                            # Special analysis for each test circle
                            st.write("### Individual Test Circle Details")
                            
                            for idx, row in test_df.iterrows():
                                with st.expander(f"Test Circle: {row.get('circle_id', 'Unknown')}"):
                                    st.write(f"**Circle ID:** {row.get('circle_id', 'Unknown')}")
                                    st.write(f"**Region:** {row.get('region', 'Unknown')}")
                                    st.write(f"**Status:** {row.get('eligible_status', 'Unknown')}")
                                    st.write(f"**Current Members:** {row.get('current_members', 'Unknown')}")
                                    st.write(f"**Maximum New Members:** {row.get('max_additions', 'Unknown')}")
                                    
                                    if 'has_none_preference' in row:
                                        st.write(f"**Has 'None' Preference:** {'Yes' if row['has_none_preference'] else 'No'}")
                                    
                                    if 'preference_overridden' in row:
                                        st.write(f"**Preference Overridden:** {'Yes' if row['preference_overridden'] else 'No'}")
                                    
                                    if 'reason' in row and pd.notna(row['reason']):
                                        st.write(f"**Ineligibility Reason:** {row['reason']}")
                                    
                                    if 'override_reason' in row and pd.notna(row['override_reason']):
                                        st.write(f"**Override Reason:** {row['override_reason']}")
                        else:
                            st.info("No test circles found in the data.")
                    else:
                        st.warning("Test circle information not available in data.")
                
                # Summary statistics for all circles
                st.write("### Overall Eligibility Statistics")
                if 'is_eligible' in eligibility_df.columns:
                    eligible_count = eligibility_df['is_eligible'].sum()
                    total_count = len(eligibility_df)
                    
                    st.write(f"- Total circles: {total_count}")
                    st.write(f"- Eligible circles: {eligible_count} ({eligible_count/total_count*100:.1f}%)")
                    
                    # Group by reason
                    if 'reason' in eligibility_df.columns:
                        reason_counts = eligibility_df['reason'].value_counts()
                        st.write("### Reasons for Ineligibility")
                        for reason, count in reason_counts.items():
                            st.write(f"- {reason}: {count}")
                
                # Create a text version for the copy button
                eligibility_text = "CIRCLE ELIGIBILITY ANALYSIS\n\n"
                
                if 'is_eligible' in eligibility_df.columns:
                    eligible_count = eligibility_df['is_eligible'].sum()
                    total_count = len(eligibility_df)
                    eligibility_text += f"Total circles: {total_count}\n"
                    eligibility_text += f"Eligible circles: {eligible_count} ({eligible_count/total_count*100:.1f}%)\n\n"
                    
                    # Add universal fix statistics
                    if 'override_reason' in eligibility_df.columns:
                        small_circle_fixes = sum(eligibility_df['override_reason'].str.contains('small circle', case=False, na=False))
                        eligibility_text += f"Small circles with universal fix applied: {small_circle_fixes}\n\n"
                
                # Eligible circles section
                eligibility_text += "ELIGIBLE CIRCLE DETAILS:\n"
                for _, row in eligibility_df[eligibility_df['is_eligible']].iterrows():
                    circle_text = []
                    circle_text.append(f"- {row.get('circle_id', 'Unknown')}")
                    
                    if 'region' in row:
                        circle_text.append(f"  Region: {row['region']}")
                    if 'subregion' in row:
                        circle_text.append(f"  Subregion: {row['subregion']}")
                    if 'meeting_time' in row:
                        circle_text.append(f"  Meeting time: {row['meeting_time']}")
                    if 'current_members' in row:
                        circle_text.append(f"  Current members: {row['current_members']}")
                    if 'max_additions' in row:
                        circle_text.append(f"  Max additions: {row['max_additions']}")
                    if 'override_reason' in row and pd.notna(row['override_reason']):
                        circle_text.append(f"  Override reason: {row['override_reason']}")
                    if 'original_preference' in row and pd.notna(row['original_preference']):
                        circle_text.append(f"  Original preference: {row['original_preference']}")
                    if 'is_test_circle' in row:
                        circle_text.append(f"  Test circle: {'✅ YES' if row['is_test_circle'] else 'NO'}")
                    
                    eligibility_text += "\n".join(circle_text) + "\n\n"
                
                # Ineligible circles section
                eligibility_text += "\nINELIGIBLE CIRCLE DETAILS:\n"
                for _, row in eligibility_df[~eligibility_df['is_eligible']].iterrows():
                    circle_text = []
                    circle_text.append(f"- {row.get('circle_id', 'Unknown')}")
                    
                    if 'region' in row:
                        circle_text.append(f"  Region: {row['region']}")
                    if 'subregion' in row:
                        circle_text.append(f"  Subregion: {row['subregion']}")
                    if 'meeting_time' in row:
                        circle_text.append(f"  Meeting time: {row['meeting_time']}")
                    if 'current_members' in row:
                        circle_text.append(f"  Current members: {row['current_members']}")
                    if 'max_additions' in row:
                        circle_text.append(f"  Max additions: {row['max_additions']}")
                    if 'reason' in row:
                        circle_text.append(f"  Reason: {row['reason']}")
                    if 'original_preference' in row and pd.notna(row['original_preference']):
                        circle_text.append(f"  Original preference: {row['original_preference']}")
                    if 'is_test_circle' in row:
                        circle_text.append(f"  Test circle: {'YES' if row['is_test_circle'] else 'NO'}")
                    
                    eligibility_text += "\n".join(circle_text) + "\n\n"
                
                # Create a text area with the info
                st.text_area("Copy this text to share circle eligibility", eligibility_text, height=300)
                
                # Add a user-friendly copy button using Streamlit components
                if st.button("📋 Copy Circle Eligibility Analysis to Clipboard", key="copy_eligibility"):
                    # This uses st.write with JavaScript for clipboard functionality
                    st.write(
                        f"""
                        <script>
                        navigator.clipboard.writeText(`{eligibility_text}`);
                        </script>
                        """
                    , unsafe_allow_html=True)
                    st.success("Copied to clipboard!")
            else:
                st.info("No circle eligibility data available to display.")
        else:
            st.info("No circle eligibility logs available. Run the optimization to generate this data.")
        
        # Add debug information for matched circles
        if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
            circle_df = st.session_state.matched_circles
            
            # Analysis of new assignments to existing circles
            st.write("#### Analysis of New Assignments to Existing Circles")
            
            # Create a list to store circle assignment analysis
            circle_analysis = []
            
            # Extract circle assignment patterns
            for _, circle_row in circle_df.iterrows():
                circle_id = circle_row.get('circle_id', 'Unknown')
                existing_members = circle_row.get('existing_members', 0)
                new_members = circle_row.get('new_members', 0)
                is_new = circle_row.get('is_new_circle', False)
                
                if not is_new and new_members > 0:
                    members_list = circle_row.get('members', [])
                    # Handle different representations of members list
                    if isinstance(members_list, str) and members_list.startswith('['):
                        try:
                            members_list = eval(members_list)
                        except:
                            members_list = []
                    
                    circle_analysis.append({
                        'circle_id': circle_id,
                        'region': circle_row.get('region', 'Unknown'),
                        'time_slot': circle_row.get('meeting_time', 'Unknown'),
                        'existing_members': existing_members,
                        'new_members': new_members,
                        'total_members': existing_members + new_members,
                        'is_test_circle': circle_id in ['IP-SIN-01', 'IP-LON-04', 'IP-HOU-02'],
                        'member_list': members_list
                    })
            
            if circle_analysis:
                # Convert to DataFrame for easier analysis
                circle_analysis_df = pd.DataFrame(circle_analysis)
                
                # Sort by new members (descending) 
                circle_analysis_df = circle_analysis_df.sort_values(by='new_members', ascending=False)
                
                # Add highlight column for test circles
                circle_analysis_df['test_circle'] = circle_analysis_df['is_test_circle'].apply(
                    lambda x: "✅ TEST CIRCLE" if x else "")
                
                # Display the DataFrame
                st.dataframe(circle_analysis_df.drop(columns=['member_list', 'is_test_circle']))
                
                # Summary statistics
                st.write("##### Summary of Circle Assignments")
                
                total_existing_circles = len(circle_analysis_df)
                test_circles_getting_members = circle_analysis_df[circle_analysis_df['is_test_circle']].shape[0]
                non_test_circles_getting_members = total_existing_circles - test_circles_getting_members
                
                st.write(f"- Total existing circles getting new members: {total_existing_circles}")
                st.write(f"- Test circles getting new members: {test_circles_getting_members}")
                st.write(f"- Non-test circles getting new members: {non_test_circles_getting_members}")
                
                # Only if we have capacity debug info
                if 'circle_capacity_debug' in st.session_state and st.session_state.circle_capacity_debug:
                    # Count total circles with capacity
                    capacity_data = list(st.session_state.circle_capacity_debug.values())
                    capacity_df = pd.DataFrame(capacity_data)
                    
                    if not capacity_df.empty:
                        total_with_capacity = len(capacity_df)
                        viable_count = capacity_df['viable'].sum()
                        
                        # Calculate percentage of viable circles that got new members
                        if viable_count > 0:
                            pct_viable_filled = (total_existing_circles / viable_count) * 100
                            st.write(f"- Percentage of viable circles that received new members: {pct_viable_filled:.1f}%")
                        
                        # Calculate percentage of all circles with capacity that got new members
                        if total_with_capacity > 0:
                            pct_capacity_filled = (total_existing_circles / total_with_capacity) * 100
                            st.write(f"- Percentage of all circles with capacity that received new members: {pct_capacity_filled:.1f}%")
                
                # Create text version for copy
                circles_text = "EXISTING CIRCLES WITH NEW MEMBERS\n\n"
                circles_text += f"Total existing circles getting new members: {total_existing_circles}\n"
                circles_text += f"Test circles getting new members: {test_circles_getting_members}\n"
                circles_text += f"Non-test circles getting new members: {non_test_circles_getting_members}\n\n"
                
                # Add details for each circle
                circles_text += "DETAILS:\n"
                for _, row in circle_analysis_df.iterrows():
                    circles_text += f"- {row['circle_id']} ({row['region']}, {row['time_slot']})\n"
                    circles_text += f"  Existing members: {int(row['existing_members'])}, New members: {int(row['new_members'])}\n"
                    circles_text += f"  {'✅ TEST CIRCLE' if row['is_test_circle'] else ''}\n\n"
                
                # Create a text area with the info
                st.text_area("Copy this text to share circle assignments", circles_text, height=300)
                
                # Add a user-friendly copy button using Streamlit components
                if st.button("📋 Copy Circle Assignments to Clipboard", key="copy_assignments"):
                    # This uses st.write with JavaScript for clipboard functionality
                    st.write(
                        f"""
                        <script>
                        navigator.clipboard.writeText(`{circles_text}`);
                        </script>
                        """
                    , unsafe_allow_html=True)
                    st.success("Copied to clipboard!")
            else:
                st.info("No existing circles received new members in the current run.")
        else:
            st.info("No matched circles data available. Run the optimization first.")
    
    with debug_tab4:
        st.write("### All Circles Debug")
        st.write("This section shows all circle information for deep debugging")
        
        if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
            circle_df = st.session_state.matched_circles
            
            # Display all circle data
            st.write(f"Found {len(circle_df)} circles in the results")
            
            # Fix for PyArrow error - ensure members column is consistently formatted
            if 'members' in circle_df.columns:
                # Create a deep copy to avoid modifying the original dataframe
                display_df = circle_df.copy(deep=True)
                
                # Convert members lists to string representation for display
                def format_members(members_data):
                    if members_data is None:
                        return "[]"
                    if isinstance(members_data, list):
                        # Extract member IDs if they are dictionaries
                        if members_data and isinstance(members_data[0], dict):
                            ids = []
                            for member in members_data:
                                member_id = member.get('Encoded ID', member.get('participant_id', 'unknown'))
                                ids.append(str(member_id))
                            return f"[{len(ids)} members]" # Short representation
                        else:
                            return f"[{len(members_data)} members]" # Short representation
                    # If it's already a string or other type, just convert to string
                    return str(members_data)
                
                # Apply the transformation
                display_df['members'] = display_df['members'].apply(format_members)
                
                # Display the modified dataframe
                st.dataframe(display_df)
            else:
                # If no members column, display the original dataframe
                st.dataframe(circle_df)
            
            # Basic statistics
            st.write("### Circle Statistics")
            total_circles = len(circle_df)
            new_circles = circle_df['is_new_circle'].sum() if 'is_new_circle' in circle_df.columns else 0
            existing_circles = total_circles - new_circles
            total_members = circle_df['member_count'].sum() if 'member_count' in circle_df.columns else 0
            
            st.write(f"- Total circles: {total_circles}")
            st.write(f"- New circles: {new_circles}")
            st.write(f"- Continuing circles: {existing_circles}")
            st.write(f"- Total members across all circles: {total_members}")
            
            # Create a searchable text area with all circle data
            circle_text = "ALL CIRCLES DEBUG DATA\n\n"
            circle_text += f"Total circles: {total_circles}\n"
            circle_text += f"New circles: {new_circles}\n"
            circle_text += f"Continuing circles: {existing_circles}\n"
            circle_text += f"Total members: {total_members}\n\n"
            
            # Add details for each circle
            circle_text += "CIRCLE DETAILS:\n"
            for _, row in circle_df.iterrows():
                circle_text += f"- Circle ID: {row.get('circle_id', 'Unknown')}\n"
                circle_text += f"  Region: {row.get('region', 'Unknown')}\n"
                circle_text += f"  Meeting time: {row.get('meeting_time', 'Unknown')}\n"
                circle_text += f"  Member count: {row.get('member_count', 0)}\n"
                circle_text += f"  New members: {row.get('new_members', 0)}\n"
                circle_text += f"  Is new circle: {row.get('is_new_circle', False)}\n"
                circle_text += "\n"
            
            # Add copy functionality
            st.text_area("Copy this text to share all circle data", circle_text, height=300)
            
            # Add copy button
            st.markdown("""
            <button onclick="navigator.clipboard.writeText(document.querySelectorAll('textarea')[3].value)">
                📋 Copy All Circles Debug to Clipboard
            </button>
            """, unsafe_allow_html=True)
        else:
            st.info("No matched circles data available. Run the optimization first.")
            
    # Original Houston Circles Debug section
    st.write("## Original Houston Circles Debug")
    
    # Show a powerful data verification section FIRST
    st.subheader("🔴 Data Verification & Variable Tracking")
    st.write("This section directly inspects input data and model variables to diagnose issues")
    
    if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        st.success(f"✅ Processed data available with {len(df)} participants")
        
        # Check directly for our test participant
        test_id = '72549701782'
        if test_id in df['Encoded ID'].values:
            st.success(f"✅ TEST PARTICIPANT FOUND: {test_id}")
            
            # Show test participant details
            test_row = df[df['Encoded ID'] == test_id].iloc[0]
            test_info = {
                "ID": test_id,
                "Status": test_row.get('Status', 'N/A'),
                "Current Region": test_row.get('Current_Region', 'N/A'),
                "Location Preferences": [
                    test_row.get('first_choice_location', 'N/A'),
                    test_row.get('second_choice_location', 'N/A'),
                    test_row.get('third_choice_location', 'N/A')
                ],
                "Time Preferences": [
                    test_row.get('first_choice_time', 'N/A'),
                    test_row.get('second_choice_time', 'N/A'),
                    test_row.get('third_choice_time', 'N/A')
                ]
            }
            st.json(test_info)
        else:
            st.error(f"❌ TEST PARTICIPANT NOT FOUND: {test_id}")
            st.info("The test participant should be in the data but isn't - this is a critical issue")
        
        # Check for Houston circles
        houston_filter = df['Current_Circle_ID'].astype(str).str.contains('HOU', na=False)
        houston_circles = df[houston_filter]
        
        if len(houston_circles) > 0:
            st.success(f"✅ Found {len(houston_circles)} participants in Houston circles")
            
            # Get unique circle IDs
            circle_ids = houston_circles['Current_Circle_ID'].unique()
            st.write(f"Houston circles: {', '.join(circle_ids)}")
            
            # Check specifically for IP-HOU-02
            ip_hou_02 = df[df['Current_Circle_ID'] == 'IP-HOU-02']
            if len(ip_hou_02) > 0:
                st.success(f"✅ IP-HOU-02 circle found with {len(ip_hou_02)} members")
                
                # Show detailed info about this circle
                with st.expander("IP-HOU-02 Details"):
                    st.dataframe(ip_hou_02)
                    
                    # Try to extract meeting time with multiple column name options
                    day_column = None
                    time_column = None
                    
                    # Try different potential column names for meeting day - same logic as in optimizer
                    for col_name in ['Current_Meeting_Day', 'Current Meeting Day', 'Current/ Continuing Meeting Day']:
                        if col_name in ip_hou_02.columns:
                            day_column = col_name
                            break
                    
                    # Try different potential column names for meeting time
                    for col_name in ['Current_Meeting_Time', 'Current Meeting Time', 'Current/ Continuing Meeting Time']:
                        if col_name in ip_hou_02.columns:
                            time_column = col_name
                            break
                    
                    # Get meeting day and time, defaulting to 'Not available' if not found
                    meeting_day = ip_hou_02[day_column].iloc[0] if day_column else 'Not available'
                    meeting_time = ip_hou_02[time_column].iloc[0] if time_column else 'Not available'
                    
                    # Format and display
                    st.write(f"Meeting day: {meeting_day}")
                    st.write(f"Meeting time: {meeting_time}")
                    st.write(f"Formatted meeting time: {meeting_day} ({meeting_time})")
            else:
                st.error("❌ IP-HOU-02 circle not found in the data")
                st.info("The test circle should be in the data but isn't - this is a critical issue")
        else:
            st.error("❌ No Houston circles found in the data")
    else:
        st.error("❌ No processed data available in session state")
    
    # Now check results if available
    if 'results' in st.session_state and st.session_state.results is not None:
        results_df = st.session_state.results
        st.success(f"✅ Results data available with {len(results_df)} participants")
        
        # Check unmatched counts
        if 'proposed_NEW_circles_id' in results_df.columns:
            unmatched = results_df[results_df['proposed_NEW_circles_id'] == 'UNMATCHED']
            st.info(f"Unmatched participants: {len(unmatched)} out of {len(results_df)} ({len(unmatched)/len(results_df)*100:.2f}%)")
        
        # Check for missing preference data
        missing_prefs_cols = []
        for pref_type in ['location', 'time']:
            for choice_num in range(1, 4):
                col_name = f'first_choice_{pref_type}'
                if col_name in results_df.columns:
                    missing_count = results_df[col_name].isna().sum()
                    if missing_count > 0:
                        missing_prefs_cols.append(f"{col_name}: {missing_count} missing")
        
        if missing_prefs_cols:
            st.warning("⚠️ Missing preference data:")
            for col_info in missing_prefs_cols:
                st.text(f"- {col_info}")
        else:
            st.success("✅ All preference columns are populated")
        
        # Examine unmatched reasons
        if 'unmatched_reason' in results_df.columns:
            unmatched_reasons = results_df['unmatched_reason'].value_counts()
            if not unmatched_reasons.empty:
                st.subheader("Unmatched Reasons")
                
                # Create a DataFrame for plotting
                reason_df = pd.DataFrame({
                    'Reason': unmatched_reasons.index,
                    'Count': unmatched_reasons.values
                })
                
                # Create a bar chart
                fig = px.bar(
                    reason_df,
                    x='Reason',
                    y='Count',
                    title='Distribution of Unmatched Reasons',
                    text='Count',
                    color_discrete_sequence=['#8C1515']
                )
                
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    xaxis_title="Unmatched Reason",
                    yaxis_title="Count of Participants"
                )
                
                st.plotly_chart(fig, use_container_width=True, key="plot_1365")
                
                # Create a table
                st.dataframe(reason_df)
                
                # Examine specific reasons
                if 'NO_LOCATION_MATCH' in unmatched_reasons:
                    no_location_match = results_df[results_df['unmatched_reason'] == 'NO_LOCATION_MATCH']
                    
                    with st.expander(f"NO_LOCATION_MATCH ({len(no_location_match)} participants)"):
                        # Extract regions
                        regions = []
                        if 'Current_Region' in no_location_match.columns:
                            regions = no_location_match['Current_Region'].value_counts()
                        elif 'Region' in no_location_match.columns:
                            regions = no_location_match['Region'].value_counts()
                        
                        if not regions.empty:
                            st.write("Regions of participants with NO_LOCATION_MATCH:")
                            st.dataframe(pd.DataFrame({
                                'Region': regions.index,
                                'Count': regions.values
                            }))
                
                # Add a debugging function to check time format issues
                with st.expander("Time Format Debugging"):
                    st.subheader("🔍 Time Format Investigation")
                    st.write("This section helps identify any issues with time format parsing")
                    
                    # Find relevant time columns
                    time_columns = []
                    for col in results_df.columns:
                        if col.endswith('_time') and 'choice' in col:
                            time_columns.append(col)
                    
                    if time_columns:
                        # Select a column to examine
                        selected_time_col = st.selectbox("Select time column to examine", 
                                                        options=time_columns,
                                                        index=0)
                        
                        # Define a demonstration function to show time parsing
                        def extract_days_and_period_demo(time_str):
                            """Demonstration version that returns detailed info for display"""
                            if pd.isna(time_str) or not time_str:
                                return {
                                    "original": str(time_str),
                                    "is_valid": False,
                                    "days_found": [],
                                    "error": "Empty or null value"
                                }
                            
                            # Extract day information
                            days_mapping = {
                                'mon': 'Monday',
                                'tue': 'Tuesday', 
                                'wed': 'Wednesday',
                                'thu': 'Thursday',
                                'fri': 'Friday',
                                'sat': 'Saturday',
                                'sun': 'Sunday'
                            }
                            
                            # Convert to lowercase for easier matching
                            time_str_lower = str(time_str).lower()
                            
                            # Find days mentioned in the string
                            days_found = []
                            for day_abbr, day_full in days_mapping.items():
                                if day_abbr in time_str_lower:
                                    days_found.append(day_full)
                            
                            # Check for morning/afternoon/evening
                            period_found = None
                            if any(term in time_str_lower for term in ['am', 'morning']):
                                period_found = 'Morning'
                            elif any(term in time_str_lower for term in ['pm', 'afternoon']):
                                period_found = 'Afternoon'
                            elif 'evening' in time_str_lower:
                                period_found = 'Evening'
                            
                            # Return detailed parsing results
                            return {
                                "original": time_str,
                                "is_valid": bool(days_found),
                                "days_found": days_found,
                                "period_found": period_found,
                                "error": None if days_found else "No valid days found"
                            }
                        
                        # Simplified version that just returns extracted days
                        def extract_days_demo(time_str):
                            """Simplified version of extract_days_and_period_demo specifically for UI display"""
                            result = extract_days_and_period_demo(time_str)
                            return result["days_found"]
                        
                        # Sample a few values to show parsing
                        sample_df = results_df[selected_time_col].dropna().sample(min(5, len(results_df))).tolist()
                        
                        st.write("Sample parsing results:")
                        for sample in sample_df:
                            parsing_result = extract_days_and_period_demo(sample)
                            st.write(f"Original: '{parsing_result['original']}'")
                            st.write(f"Valid: {parsing_result['is_valid']}")
                            st.write(f"Days found: {', '.join(parsing_result['days_found']) if parsing_result['days_found'] else 'None'}")
                            st.write(f"Period: {parsing_result.get('period_found', 'Not found')}")
                            st.write("---")
                        
                        # Count parsing successes and failures
                        time_values = results_df[selected_time_col].dropna()
                        parsing_success = sum(1 for t in time_values if extract_days_demo(t))
                        parsing_failure = len(time_values) - parsing_success
                        
                        st.metric("Time values successfully parsed", 
                                f"{parsing_success} / {len(time_values)} ({parsing_success/len(time_values)*100:.1f}%)")
                        
                        if parsing_failure > 0:
                            st.warning(f"⚠️ {parsing_failure} time values could not be parsed")
                            
                            # Show problematic values
                            problem_values = [t for t in time_values if not extract_days_demo(t)]
                            if problem_values:
                                st.write("Problematic time values:")
                                for idx, val in enumerate(problem_values[:10]):  # Show at most 10
                                    st.text(f"{idx+1}. '{val}'")
                                if len(problem_values) > 10:
                                    st.text(f"... and {len(problem_values) - 10} more")
                    else:
                        st.info("No time-related columns found in the data")
    else:
        st.error("❌ No results data available in session state")
    
    with debug_tab5:
        st.write("### Seattle Compatibility Analysis")
        st.write("This section shows detailed analysis of Seattle circles, especially IP-SEA-01.")
        
        st.write("Seattle compatibility analysis would appear here")
    
    with debug_tab6:
        # Use our standalone function for metadata debug
        render_metadata_debug_tab()
    
    # Check for matched circles
    if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
        circles_df = st.session_state.matched_circles
        if hasattr(circles_df, 'empty') and circles_df.empty:
            st.error("❌ Matched circles dataframe is empty")
        else:
            st.success(f"✅ Matched circles data available with {len(circles_df)} circles")
            
            # Count total members
            total_members = circles_df['member_count'].sum() if 'member_count' in circles_df.columns else "N/A"
            st.metric("Total matched participants", total_members)
            
            # Circle size distribution
            if 'member_count' in circles_df.columns:
                circle_sizes = circles_df['member_count'].value_counts().sort_index()
                
                st.subheader("Circle Size Distribution")
                
                # Create a DataFrame for plotting
                size_df = pd.DataFrame({
                    'Circle Size': circle_sizes.index,
                    'Number of Circles': circle_sizes.values
                })
                
                # Create a bar chart
                fig = px.bar(
                    size_df,
                    x='Circle Size',
                    y='Number of Circles',
                    title='Distribution of Circle Sizes',
                    text='Number of Circles',
                    color_discrete_sequence=['#8C1515']
                )
                
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    xaxis_title="Number of Members in Circle",
                    yaxis_title="Number of Circles"
                )
                
                st.plotly_chart(fig, use_container_width=True, key="plot_1537")
    else:
        st.error("❌ No matched circles data available in session state")
    
    # Show information about current session state
    st.subheader("🔢 Session State Variables")
    session_keys = list(st.session_state.keys())
    selected_keys = st.multiselect("Select session state variables to inspect", 
                                 options=session_keys,
                                 default=session_keys[:3] if len(session_keys) > 3 else session_keys)
    
    if selected_keys:
        for key in selected_keys:
            value = st.session_state[key]
            st.write(f"**{key}**:")
            if isinstance(value, pd.DataFrame):
                st.write(f"DataFrame with {len(value)} rows and {len(value.columns)} columns")
                st.dataframe(value.head(5))
            elif value is None:
                st.write("None")
            elif isinstance(value, (list, tuple, set)):
                st.write(f"{type(value).__name__} with {len(value)} items")
                st.write(str(value)[:1000] + "..." if len(str(value)) > 1000 else str(value))
            else:
                st.write(str(value)[:1000] + "..." if len(str(value)) > 1000 else str(value))


def render_split_circle_summary(key_prefix="overview"):
    """Render a summary of split circles
    
    Args:
        key_prefix (str): Prefix to use for Streamlit widget keys to ensure uniqueness
    """
    print(f"\n🔍 CHECKING FOR SPLIT CIRCLE SUMMARY DATA (key_prefix={key_prefix})")
    
    # Check if we should skip rendering based on session state flag
    if hasattr(st.session_state, 'skip_split_circle_summary') and st.session_state.skip_split_circle_summary:
        print("ℹ️ Skipping split circle summary display due to skip_split_circle_summary flag")
        return
    
    if 'split_circle_summary' not in st.session_state:
        print("⚠️ No split_circle_summary found in session state")
        st.info("No circle splitting summary available. Circle splitting may not have been needed or didn't meet requirements.")
        return
    
    split_summary = st.session_state.split_circle_summary
    
    # Check the format of the summary - if it's the new format, it will have 'total_circles_examined'
    if 'total_circles_examined' in split_summary:
        # New circle splitter format
        st.subheader("Circle Splitting Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Large Circles Found", split_summary.get("total_large_circles_found", 0))
            st.metric("Circles Successfully Split", split_summary.get("total_circles_successfully_split", 0))
            
        with col2:
            st.metric("New Circles Created", split_summary.get("total_new_circles_created", 0))
            st.metric("Total Circles Examined", split_summary.get("total_circles_examined", 0))
        
        # If we have details, show a table of specific splits
        if "split_details" in split_summary and split_summary["split_details"]:
            st.subheader("Split Details")
            
            # Create a list to hold the data for the table
            table_data = []
            
            for detail in split_summary["split_details"]:
                original_id = detail.get("original_circle_id", "Unknown")
                new_ids = detail.get("new_circle_ids", [])
                member_counts = detail.get("member_counts", [])
                always_hosts = detail.get("always_hosts", [0] * len(new_ids))
                sometimes_hosts = detail.get("sometimes_hosts", [0] * len(new_ids))
                
                # Format the new IDs and member counts
                new_ids_str = ", ".join(new_ids)
                member_counts_str = ", ".join([str(count) for count in member_counts])
                
                # Add host information with enhanced details
                host_info = []
                for i in range(len(new_ids)):
                    # Get host counts
                    always = always_hosts[i] if i < len(always_hosts) else 0
                    sometimes = sometimes_hosts[i] if i < len(sometimes_hosts) else 0
                    
                    # Get host IDs if available
                    always_host_ids = detail.get("always_host_ids", [])
                    sometimes_host_ids = detail.get("sometimes_host_ids", [])
                    
                    # Get the counts for this specific circle
                    always_count = len(always_host_ids[i]) if i < len(always_host_ids) else always
                    sometimes_count = len(sometimes_host_ids[i]) if i < len(sometimes_host_ids) else sometimes
                    
                    # Use the better count (either from the IDs or the original count)
                    always_final = max(always, always_count)
                    sometimes_final = max(sometimes, sometimes_count)
                    
                    # Format as "XA/YS" (X Always hosts, Y Sometimes hosts)
                    host_info.append(f"{always_final}A/{sometimes_final}S")
                
                host_info_str = ", ".join(host_info)
                
                # Add to table data
                table_data.append({
                    "Original Circle": original_id,
                    "New Circles": new_ids_str,
                    "Member Distribution": member_counts_str,
                    "Host Distribution": host_info_str
                })
            
            # Display as a DataFrame
            st.dataframe(pd.DataFrame(table_data))
            
            # Calculate total members before and after splitting
            total_members_before = sum([sum(detail.get("member_counts", [])) for detail in split_summary["split_details"]])
            total_members_after = sum([sum(detail.get("member_counts", [])) for detail in split_summary["split_details"]])
            
            # Show verification message
            st.info(f"**Verification**: Total members before splitting: {total_members_before}, after splitting: {total_members_after}. " 
                    f"All members were preserved in the splitting process.")
                    
            # Check if we have a metadata manager in the session state
            if 'circle_metadata_manager' in st.session_state and st.session_state.circle_metadata_manager:
                manager = st.session_state.circle_metadata_manager
                
                # Count total split circles tracked by the manager
                split_count = len(manager.split_circles)
                original_count = len(manager.original_circles)
                
                if split_count > 0:
                    st.success(f"The CircleMetadataManager is tracking {split_count} split circles from {original_count} original circles.")
                    
                    # Show a few examples
                    # Use a dynamic prefix based on where it's called from plus a unique identifier
                    # Combine the key_prefix with a random number to ensure uniqueness
                    unique_id = f"{key_prefix}_{id(manager)}_{random.randint(10000, 99999)}"
                    checkbox_key = f"metadata_manager_split_details_{unique_id}"
                    if st.checkbox("Show Split Circle Details from Metadata Manager", key=checkbox_key):
                        st.subheader("Sample of Split Circles in Metadata Manager")
                        
                        # Get up to 5 split circle IDs
                        sample_split_ids = list(manager.split_circles.keys())[:5]
                        
                        # Create a list for the table
                        manager_data = []
                        
                        for split_id in sample_split_ids:
                            original_id = manager.get_original_circle_id(split_id)
                            circle_data = manager.circles.get(split_id, {})
                            
                            manager_data.append({
                                "Split Circle ID": split_id,
                                "Original Circle ID": original_id,
                                "Member Count": circle_data.get("member_count", "N/A"),
                                "Max Additions": circle_data.get("max_additions", "N/A"),
                                "Always Hosts": circle_data.get("always_hosts", "N/A"),
                                "Sometimes Hosts": circle_data.get("sometimes_hosts", "N/A")
                            })
                        
                        # Display the table
                        st.dataframe(pd.DataFrame(manager_data))
                else:
                    st.warning("No split circles found in the CircleMetadataManager.")
            
    # Handle legacy format for backward compatibility
    elif 'total_circles_eligible_for_splitting' in split_summary:
        print(f"✅ Found legacy split_circle_summary with {split_summary['total_circles_eligible_for_splitting']} eligible circles")
        
        # Only show this section if there were circles eligible for splitting
        if split_summary['total_circles_eligible_for_splitting'] == 0:
            print("ℹ️ No circles eligible for splitting, skipping summary display")
            st.info("No circle splitting was needed - all circles are optimally sized.")
            return
        
        print(f"🔍 Displaying split summary for {split_summary['total_circles_successfully_split']} split circles")
        st.subheader("Circle Splitting Summary")
        
        # Create metrics for split circles
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Circles Split", f"{split_summary['total_circles_successfully_split']} of {split_summary['total_circles_eligible_for_splitting']} eligible")
        
        with col2:
            st.metric("New Circles Created", split_summary['total_new_circles_created'])
    
    # For completely unknown format
    else:
        st.info("Circle splitting data is in an unknown format. Please check the logs for details.")
        st.json(split_summary)
    
    # Show details of each split
    if split_summary['split_details']:
        with st.expander("View Split Circle Details", expanded=False):
            for split in split_summary['split_details']:
                original_id = split['original_circle_id']
                new_ids = split['new_circle_ids']
                
                st.write(f"**Original Circle**: {original_id} with {split['member_count']} members")
                st.write(f"**Split Into**: {len(new_ids)} circles")
                
                # List new circle IDs
                for i, new_id in enumerate(new_ids):
                    st.write(f"- {new_id}")
                
                st.markdown("---")
    
    # Show circles that couldn't be split due to host requirements
    if split_summary['circles_unable_to_split']:
        with st.expander("Circles Unable to Split", expanded=False):
            st.write("The following circles had 11+ members but couldn't be split because they didn't meet host requirements:")
            
            for circle in split_summary['circles_unable_to_split']:
                st.write(f"**{circle['circle_id']}** with {circle['member_count']} members")
                st.write(f"Reason: {circle['reason']}")
                st.markdown("---")

def render_results_overview():
    """Render the results overview section"""
    if ('matched_circles' not in st.session_state or 
        st.session_state.matched_circles is None or 
        (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty)):
        st.warning("No matching results available. Please run the matching algorithm first.")
        return
    
    st.subheader("Matching Results Overview")
    
    # Add the split circle summary right after the main header
    render_split_circle_summary(key_prefix="results_overview")
    
    # Get the data
    matched_df = st.session_state.matched_circles
    results_df = st.session_state.results if 'results' in st.session_state else None
    
    # Create columns for the metrics
    col1, col2, col3 = st.columns(3)
    
    # Column 1: Circle stats
    with col1:
        st.metric("Number of Circles", len(matched_df))
        
        # Average circle size
        if 'member_count' in matched_df.columns:
            avg_size = matched_df['member_count'].mean()
            st.metric("Average Circle Size", f"{avg_size:.1f}")
    
    # Column 2: Participant stats
    with col2:
        # Use statistics from session state if available, otherwise calculate them
        if 'match_statistics' in st.session_state:
            # Use our standardized statistics
            match_stats = st.session_state.match_statistics
            print("\n🔍 DETAILS TAB - USING STANDARDIZED STATISTICS")
            print(f"  Using standardized statistics from session state")
            
            # Use the matched_participants count from our standardized calculations
            total_matched = match_stats['matched_participants']
            unmatched_count = match_stats['unmatched_participants']
            
            # Log diagnostic info
            print(f"  Using matched count: {total_matched}")
            print(f"  Using unmatched count: {unmatched_count}")
            
            # If there was a discrepancy between methods, note it
            if 'details_matched_count' in match_stats:
                print(f"  Note: circle member_count sum would give {match_stats['details_matched_count']} (diff: {match_stats['match_discrepancy']})")
        else:
            # Fall back to old calculation method for backwards compatibility
            print("\n⚠️ DETAILS TAB - FALLING BACK TO OLD CALCULATION METHOD")
            print(f"  match_statistics not found in session state - calculating directly")
            
            # Calculate matched participants using circle member counts
            total_matched = matched_df['member_count'].sum() if 'member_count' in matched_df.columns else 0
            print(f"  Calculated total_matched (from circle member_count sum): {total_matched}")
            
            # Count unmatched using results DataFrame
            if results_df is not None and 'proposed_NEW_circles_id' in results_df.columns:
                unmatched_count = len(results_df[results_df['proposed_NEW_circles_id'] == 'UNMATCHED'])
            else:
                unmatched_count = 0
                
            print(f"  Calculated unmatched_count: {unmatched_count}")
            
            # Import and use our standardized calculation for future page loads
            from utils.helpers import calculate_matching_statistics
            match_stats = calculate_matching_statistics(results_df, matched_df)
            
            # Store in session state for next time
            st.session_state.match_statistics = match_stats
            print(f"  Stored standardized statistics in session state for future use")
        
        # Display the metrics in the UI
        st.metric("Participants Matched", total_matched)
        st.metric("Participants Unmatched", unmatched_count)
    
    # Column 3: Success rates and Diversity
    with col3:
        # Use our helper function to calculate the diversity score consistently
        total_diversity_score = calculate_total_diversity_score(matched_df, results_df)
        
        # Display Diversity Score metric
        st.metric("Diversity Score", total_diversity_score)
        
        # Use match rate from standardized statistics if available
        if 'match_statistics' in st.session_state:
            match_rate = st.session_state.match_statistics['match_rate']
            print(f"  Using standardized match rate: {match_rate:.1f}%")
        else:
            # Fall back to calculating it directly
            if results_df is not None and 'proposed_NEW_circles_id' in results_df.columns:
                total_participants = len(results_df)
                match_rate = (total_matched / total_participants) * 100 if total_participants > 0 else 0
                print(f"  Calculated match rate directly: {match_rate:.1f}%")
            else:
                match_rate = 0
                
        # Display match rate metric
        st.metric("Match Success Rate", f"{match_rate:.1f}%")
        
        # Circles with target size
        if 'member_count' in matched_df.columns and 'target_size' in matched_df.columns:
            target_count = len(matched_df[matched_df['member_count'] == matched_df['target_size']])
            target_pct = (target_count / len(matched_df)) * 100 if len(matched_df) > 0 else 0
            st.metric("Circles at Target Size", f"{target_count} ({target_pct:.1f}%)")
    
    # Distribution of circle sizes
    if 'member_count' in matched_df.columns:
        st.subheader("Circle Size Distribution")
        
        # Count circles by size
        size_counts = matched_df['member_count'].value_counts().sort_index()
        
        # Create a DataFrame for plotting
        size_df = pd.DataFrame({
            'Circle Size': size_counts.index,
            'Count': size_counts.values
        })
        
        # Create a bar chart
        fig = px.bar(
            size_df,
            x='Circle Size',
            y='Count',
            title='Distribution of Circle Sizes',
            text='Count',
            color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
        )
        
        # Format
        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis=dict(
                title="Number of Members",
                tickmode='linear',
                dtick=1
            ),
            yaxis_title="Number of Circles"
        )
        
        # Plot with unique key
        st.plotly_chart(fig, use_container_width=True, key="details_overview_circle_size_dist")
    
    # Show unmatched reasons if available
    render_unmatched_table()
    
    # Show circle composition table
    render_circle_table()


def render_circle_table():
    """Render the circle composition table"""
    # ENHANCED: Use rebuild_circle_member_lists to get the most accurate data
    import pandas as pd
    from utils.circle_rebuilder import rebuild_circle_member_lists
    
    # Check if we have data to display
    if ('matched_circles' not in st.session_state or 
        st.session_state.matched_circles is None or
        (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty)):
        return
    
    st.subheader("Circle Composition")
    
    print("\n🔍 CIRCLE COMPOSITION TABLE DEBUG:")
    
    # Use the matched_circles from session state as the starting point
    base_circles_df = st.session_state.matched_circles.copy()
    
    # Get the participant data
    participants_df = st.session_state.processed_data if 'processed_data' in st.session_state else None
    results_df = st.session_state.results if 'results' in st.session_state else None
    
    # Use the participant data that has the most information (results is preferred)
    participants_data = results_df if results_df is not None else participants_df
    
    # If we have the necessary data, rebuild the circle member lists for accuracy
    if participants_data is not None and not base_circles_df.empty:
        print("  ✅ Using rebuild_circle_member_lists to get accurate circle data")
        circles_df = rebuild_circle_member_lists(base_circles_df, participants_data)
        print(f"  Rebuilt data for {len(circles_df)} circles")
    else:
        print("  ⚠️ Missing data to rebuild circles, using base circle data")
        circles_df = base_circles_df
    
    # Filter to include only active circles (including active split circles)
    print(f"  Total circles before filtering: {len(circles_df)}")
    
    # Check if 'is_active' column exists, if not, create it
    if 'is_active' not in circles_df.columns:
        circles_df['is_active'] = True
        # Mark original circles that have been split as inactive
        from utils.circle_metadata_manager import get_manager_from_session_state
        manager = get_manager_from_session_state(st.session_state)
        if manager:
            original_circles = list(manager.original_circles.keys())
            print(f"  Original circles that were split: {original_circles}")
            for circle_id in original_circles:
                if circle_id in circles_df['circle_id'].values:
                    circles_df.loc[circles_df['circle_id'] == circle_id, 'is_active'] = False
                    print(f"  Marked {circle_id} as inactive (was split)")
    
    # Filter to only show active circles
    active_circles_df = circles_df[circles_df['is_active'] == True].copy()
    print(f"  Active circles after filtering: {len(active_circles_df)}")
    
    # Count split circles
    split_circles = active_circles_df[active_circles_df['circle_id'].str.contains('SPLIT', case=True, na=False)]
    print(f"  Found {len(split_circles)} active split circles")
    if not split_circles.empty:
        print(f"  Split circle IDs: {split_circles['circle_id'].tolist()}")
    
    # Final dataframe to display
    circles_df = active_circles_df
    
    # Add special diagnostic section for test circles
    with st.expander("Circle Inspector (Debug Tool)"):
        st.write("Examine details of specific circles for debugging")
        
        # Input for circle ID
        circle_id = st.text_input("Enter Circle ID to inspect (e.g., IP-BOS-04):", key="circle_inspector_id")
        
        if circle_id:
            if manager is not None:
                # Get circle data from metadata manager
                circle_data = manager.get_circle_data(circle_id)
                
                if circle_data:
                    st.subheader(f"Circle {circle_id} Details")
                    st.write("Basic information:")
                    st.write(f"- Region: {circle_data.get('region', 'Unknown')}")
                    st.write(f"- Subregion: {circle_data.get('subregion', 'Unknown')}")
                    st.write(f"- Meeting Time: {circle_data.get('meeting_time', 'Unknown')}")
                    st.write(f"- Member Count: {circle_data.get('member_count', 0)}")
                    st.write(f"- Always Hosts: {circle_data.get('always_hosts', 0)}")
                    st.write(f"- Sometimes Hosts: {circle_data.get('sometimes_hosts', 0)}")
                    st.write(f"- Max Additions: {circle_data.get('max_additions', 0)}")
                    st.write(f"- New Members: {circle_data.get('new_members', 0)}")
                    
                    # Get member details
                    members = circle_data.get('members', [])
                    if members:
                        st.write(f"\nMembers ({len(members)}):")  
                        members_df = manager.get_circle_member_data(circle_id)
                        
                        if not members_df.empty:
                            # Extract key columns for display
                            display_cols = ['Encoded ID']
                            
                            # Add host column if it exists
                            host_col = None
                            for col in ['host', 'Host', 'willing_to_host']:
                                if col in members_df.columns:
                                    host_col = col
                                    display_cols.append(host_col)
                                    break
                            
                            # Show member data
                            st.dataframe(members_df[display_cols])
                            
                            # Add special host analysis
                            if host_col:
                                st.subheader("Host Status Analysis")
                                host_counts = members_df[host_col].value_counts().reset_index()
                                host_counts.columns = ['Host Status', 'Count']
                                st.dataframe(host_counts)
                                
                                # Use standardized host status normalization from data_standardization module
                                from utils.data_standardization import normalize_host_status
                                
                                # Count using standardized method
                                always_count = 0
                                sometimes_count = 0
                                
                                # Show detailed host status analysis
                                st.write("### Host Status Normalization")
                                st.write("This shows how each member's host status is standardized:")
                                
                                host_analysis = []
                                for _, row in members_df.iterrows():
                                    raw_value = row[host_col] if not pd.isna(row[host_col]) else None
                                    normalized = normalize_host_status(raw_value)
                                    
                                    # Count based on standardized values
                                    if normalized == 'ALWAYS':
                                        always_count += 1
                                    elif normalized == 'SOMETIMES':
                                        sometimes_count += 1
                                        
                                    host_analysis.append({
                                        'Encoded ID': row['Encoded ID'],
                                        'Raw Host Value': str(raw_value),
                                        'Standardized Value': normalized
                                    })
                                
                                # Show the normalization analysis
                                host_analysis_df = pd.DataFrame(host_analysis)
                                st.dataframe(host_analysis_df)
                                
                                # Show the standardized counts
                                st.write(f"**Standardized host counts:** {always_count} Always, {sometimes_count} Sometimes")
                                st.write(f"**Metadata manager values:** {circle_data.get('always_hosts', 0)} Always, {circle_data.get('sometimes_hosts', 0)} Sometimes")
                                
                                # Highlight discrepancies
                                if always_count != circle_data.get('always_hosts', 0):
                                    st.error(f"DISCREPANCY: Circle shows {circle_data.get('always_hosts', 0)} Always Hosts but standardized count is {always_count}")
                                    
                                    # Show explanation for debugging
                                    st.info("This discrepancy could be caused by:"
                                           "\n1. The metadata hasn't been updated with the latest standardization"
                                           "\n2. Different standardization logic was used when metadata was created"
                                           "\n3. Data has changed since metadata was last calculated")
                                
                                if sometimes_count != circle_data.get('sometimes_hosts', 0):
                                    st.error(f"DISCREPANCY: Circle shows {circle_data.get('sometimes_hosts', 0)} Sometimes Hosts but standardized count is {sometimes_count}")
                                    
                                    # Show explanation for debugging
                                    st.info("This discrepancy could be caused by:"
                                           "\n1. The metadata hasn't been updated with the latest standardization"
                                           "\n2. Different standardization logic was used when metadata was created"
                                           "\n3. Data has changed since metadata was last calculated")
                        else:
                            st.warning("Could not retrieve member details from results data.")
                    else:
                        st.warning("No members found in this circle.")
                else:
                    st.error(f"Circle '{circle_id}' not found in metadata manager.")
            else:
                st.error("Metadata manager not available. Please run the optimization first.")
                
        # Add quick links to problematic circles
        st.write("Quick access to test circles:")
        test_circles = ['IP-BOS-04', 'IP-BOS-05']
        cols = st.columns(len(test_circles))
        for i, tc in enumerate(test_circles):
            if cols[i].button(tc, key=f"quick_access_{tc}"):
                # This will only work on next run due to how Streamlit works
                st.session_state.circle_inspector_id = tc
    
    results_df = st.session_state.results.copy() if 'results' in st.session_state else None
    
    # Diagnostics for debugging circle data
    print(f"  Circle DataFrame shape: {circles_df.shape if hasattr(circles_df, 'shape') else 'unknown'}")
    print(f"  Available columns: {list(circles_df.columns) if hasattr(circles_df, 'columns') else 'unknown'}")
    
    # CRITICAL: Fix any split circle data to make sure always_hosts and sometimes_hosts are integers
    if 'always_hosts' in circles_df.columns:
        for idx, row in circles_df.iterrows():
            # Fix NaN or None values in numeric columns
            for col in ['always_hosts', 'sometimes_hosts', 'member_count', 'max_additions', 'new_members']:
                if col in circles_df.columns:
                    if pd.isna(row[col]) or row[col] is None:
                        circles_df.at[idx, col] = 0
    
    # Show the table
    if 'circle_id' in circles_df.columns:
        # Create a display table with key information
        # ENHANCED: Added region, subregion, new_members, max_additions, always_hosts, sometimes_hosts columns
        display_cols = ['circle_id', 'region', 'subregion', 'meeting_time', 'member_count', 
                       'new_members', 'max_additions', 'always_hosts', 'sometimes_hosts']
        
        # Filter to only include columns that exist
        available_cols = [col for col in display_cols if col in circles_df.columns]
        
        # Log which columns we're displaying
        print(f"  Displaying columns: {available_cols}")
        
        if available_cols:
            display_df = circles_df[available_cols].copy()
            
            # Check for split info directly from the enhanced metadata in the DataFrame
            if 'circle_id' in circles_df.columns:
                # Check if we have split status from the metadata manager
                if 'split_status' not in circles_df.columns:
                    # Create the split_status column if it doesn't exist already
                    circles_df['split_status'] = ''
                    
                    # Derive split status from circle ID if it's not already set from metadata manager
                    is_split_circle = circles_df['circle_id'].str.contains('SPLIT', case=True, na=False)
                    circles_df['is_split_circle'] = is_split_circle
                    
                    # For split circles, determine the split letter
                    if is_split_circle.any():
                        for idx in circles_df[is_split_circle].index:
                            circle_id = circles_df.loc[idx, 'circle_id']
                            # Extract split letter from circle ID
                            if "-SPLIT-" in circle_id:
                                split_letter = circle_id[-1]  # Last character is the split letter
                                circles_df.loc[idx, 'split_status'] = f"Split {split_letter}"
                            else:
                                circles_df.loc[idx, 'split_status'] = "Split"
                
                # Add original circle ID to display if present
                if 'original_circle_id' in circles_df.columns:
                    # Only add this column if we have some split circles
                    if 'is_split_circle' in circles_df.columns and circles_df['is_split_circle'].any():
                        display_cols.append('original_circle_id')
                        if 'original_circle_id' not in available_cols:
                            available_cols.append('original_circle_id')
                            display_df = circles_df[available_cols].copy()
                
                # Add split status column to display
                display_cols.append('split_status')
                if 'split_status' not in available_cols:
                    available_cols.append('split_status')
                    display_df = circles_df[available_cols].copy()
            
            # Rename columns for display
            display_df.columns = [col.replace('_', ' ').title() for col in available_cols]
            
            # Sort by circle ID
            if 'Circle Id' in display_df.columns:
                display_df = display_df.sort_values('Circle Id')
            
            # Debug check for Max Additions column
            if 'max_additions' in circles_df.columns:
                # Check the range of max_additions values
                max_add_values = circles_df['max_additions'].unique()
                print(f"  Max Additions values: {max_add_values}")
                
                # Check for specific circles that might have issues
                for circle_id in ['IP-BOS-04', 'IP-BOS-05']:
                    if circle_id in circles_df['circle_id'].values:
                        # If we have a manager, use it to get consistent data
                        if manager:
                            circle_data = manager.get_circle_data(circle_id)
                            # Add null check to handle case where circle_data is None
                            if circle_data is not None:
                                print(f"  {circle_id} info (from manager): max_additions={circle_data.get('max_additions', 'N/A')}, "  
                                      f"member_count={circle_data.get('member_count', 'N/A')}, "
                                      f"new_members={circle_data.get('new_members', 'N/A')}, "
                                      f"always_hosts={circle_data.get('always_hosts', 'N/A')}")
                            else:
                                print(f"  ⚠️ {circle_id} not found in metadata manager")
                        else:
                            # Fall back to DataFrame lookup
                            row = circles_df[circles_df['circle_id'] == circle_id].iloc[0]
                            print(f"  {circle_id} info (from DataFrame): max_additions={row.get('max_additions', 'N/A')}, "  
                                  f"member_count={row.get('member_count', 'N/A')}, "
                                  f"new_members={row.get('new_members', 'N/A')}, "
                                  f"always_hosts={row.get('always_hosts', 'N/A')}")
            
            # Add styling to highlight split circles and large circles that should be split
            def highlight_circles(row):
                # Check if circle ID contains 'SPLIT' to identify split circles
                circle_id = row.get('Circle Id', '')
                
                # Check if it's explicitly marked as a split circle or the ID contains SPLIT
                is_split = ('Split Status' in row and row['Split Status']) or (isinstance(circle_id, str) and 'SPLIT' in circle_id)
                
                if is_split:
                    # Enhanced styling for split circles - light blue background with a subtle border
                    return ['background-color: #e6f3ff; border-left: 3px solid #4b89dc;'] * len(row)
                
                # Check if it's a large circle (11+ members) that should have been split
                if 'Member Count' in row and isinstance(row['Member Count'], (int, float)) and row['Member Count'] >= 11:
                    # This is a large circle that wasn't split - highlight in a different color
                    print(f"🔍 Found large circle with {row['Member Count']} members that wasn't split: {row.get('Circle Id', 'unknown')}")
                    return ['background-color: #ffec99; border-left: 3px solid #f6b26b;'] * len(row)  # Light yellow with orange border
                
                return [''] * len(row)
            
            # Apply the styling function
            styled_df = display_df.style.apply(highlight_circles, axis=1)
            
            # Show the styled table
            st.dataframe(styled_df, use_container_width=True)
            
            # Add an export option with unique key
            if st.button("Export Circle Data to CSV", key="details_export_circle_data_button"):
                # Convert DataFrame to CSV
                csv = circles_df.to_csv(index=False)
                
                # Create a download link
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="circle_data.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("Circle data doesn't contain the expected columns.")
    else:
        st.warning("Circle data doesn't contain circle_id or meeting_time columns.")


def render_unmatched_table():
    """Render the unmatched participants table"""
    if 'results' not in st.session_state or st.session_state.results is None:
        return
    
    # Get unmatched participants
    results_df = st.session_state.results.copy()
    
    if 'proposed_NEW_circles_id' not in results_df.columns:
        return
    
    unmatched_df = results_df[results_df['proposed_NEW_circles_id'] == 'UNMATCHED']
    
    if len(unmatched_df) == 0:
        st.success("All participants were successfully matched!")
        return
    
    st.subheader(f"Unmatched Participants ({len(unmatched_df)})")
    
    # Show reasons for being unmatched
    if 'unmatched_reason' in unmatched_df.columns:
        reasons = unmatched_df['unmatched_reason'].value_counts()
        
        # Create a DataFrame for plotting
        reason_df = pd.DataFrame({
            'Reason': reasons.index,
            'Count': reasons.values
        })
        
        # Create a bar chart
        fig = px.bar(
            reason_df,
            x='Reason',
            y='Count',
            title='Reasons for Unmatched Participants',
            text='Count',
            color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
        )
        
        # Format
        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis_title="Unmatched Reason",
            yaxis_title="Number of Participants"
        )
        
        # Plot with unique key
        st.plotly_chart(fig, use_container_width=True, key="details_unmatched_reasons_chart")
    
    # Show the table of unmatched participants
    display_cols = ['Last Family Name', 'First Given Name', 'Encoded ID', 
                    'Current_Region', 'Status', 'unmatched_reason']
    
    # Filter to available columns
    display_cols = [col for col in display_cols if col in unmatched_df.columns]
    
    if display_cols:
        st.dataframe(unmatched_df[display_cols], use_container_width=True)
    else:
        st.warning("Unmatched participant data doesn't contain the expected columns.")


def render_circle_details():
    """Render detailed information about each circle"""
    # ENHANCED: Use CircleMetadataManager if available, fall back to direct session state access
    from utils.circle_metadata_manager import get_manager_from_session_state
    
    # Try to get the circle manager first
    manager = get_manager_from_session_state(st.session_state) if 'circle_manager' in st.session_state else None
    
    # Check if we have data to display
    if not manager and ('matched_circles' not in st.session_state or 
                     st.session_state.matched_circles is None or
                     (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty)):
        st.warning("No matching results available. Please run the matching algorithm first.")
        return
    
    # Get results DataFrame
    results_df = st.session_state.results.copy() if 'results' in st.session_state else None
    
    # If results aren't available, we can't show member details
    if results_df is None:
        st.warning("Participant data is not available. Cannot show detailed circle information.")
        return
    
    # Get circle data - from manager or directly from session state
    if manager:
        print("\n🔍 CIRCLE DETAILS DEBUG (Using CircleMetadataManager):")
        circles_df = manager.get_circles_dataframe()
        print(f"  Retrieved {len(circles_df)} circles from CircleMetadataManager")
    else:
        print("\n🔍 CIRCLE DETAILS DEBUG (Using session state directly):")
        circles_df = st.session_state.matched_circles.copy()
    
    # Filter out participants with null Encoded IDs
    from utils.helpers import get_valid_participants
    results_df = get_valid_participants(results_df)
    print(f"🔍 Circle details: Using {len(results_df)} valid participants with non-null Encoded IDs")
    
    # Get all circle IDs
    circle_ids = circles_df['circle_id'].tolist() if 'circle_id' in circles_df.columns else []
    
    if not circle_ids:
        st.warning("No circle IDs found in the matching results.")
        return
    
    # Sort circle IDs for better UX
    circle_ids.sort()
    
    # Create a selection widget to choose a circle
    selected_circle = st.selectbox("Select a circle to view details", options=circle_ids, key="circle_details_selector")
    
    # Get the selected circle's data - using manager if available
    if manager:
        circle_data = manager.get_circle_data(selected_circle)
        if circle_data is not None:
            # For display consistency, we'll need a similar dict structure to what we'd get from DataFrame
            circle_row = circle_data
            print(f"  Using CircleMetadataManager to get data for {selected_circle}")
        else:
            # Fall back to DataFrame lookup if the manager doesn't have the circle data
            print(f"  ⚠️ {selected_circle} not found in metadata manager, falling back to DataFrame")
            try:
                circle_row = circles_df[circles_df['circle_id'] == selected_circle].iloc[0]
                print(f"  Using DataFrame to get data for {selected_circle}")
            except:
                st.error(f"Could not find data for circle {selected_circle}")
                return
    else:
        # Fall back to DataFrame lookup
        try:
            circle_row = circles_df[circles_df['circle_id'] == selected_circle].iloc[0]
            print(f"  Using DataFrame to get data for {selected_circle}")
        except:
            st.error(f"Could not find data for circle {selected_circle}")
            return
    
    # Create columns for the display
    col1, col2 = st.columns([1, 2])
    
    # First column: Circle metadata
    with col1:
        st.subheader(f"Circle: {selected_circle}")
        
        # Show enhanced circle metadata (added max_additions, region, subregion)
        metadata = {
            "Region": circle_row.get('region', 'Unknown'),
            "Subregion": circle_row.get('subregion', 'Unknown'),
            "Meeting Time": circle_row.get('meeting_time', 'Not specified'),
            "Meeting Location": circle_row.get('meeting_location', 'Not specified'),
            "Member Count": circle_row.get('member_count', 'Unknown'),
            "New Members": circle_row.get('new_members', 'Unknown'),
            "Max Additions": circle_row.get('max_additions', 'Unknown'),
            "Always Hosts": circle_row.get('always_hosts', 'Unknown'),
            "Sometimes Hosts": circle_row.get('sometimes_hosts', 'Unknown'),
        }
        
        # Add split circle information if applicable
        circle_id = selected_circle
        if 'SPLIT' in circle_id:
            st.info("This circle was created by splitting a large circle with 11+ members. Split circles can accept new members up to a maximum of 8 total.")
            
            # Use a more noticeable UI element to highlight split circle status
            st.markdown("""
            <div style="background-color: #e6f3ff; padding: 10px; border-left: 4px solid #4b89dc; margin-bottom: 15px;">
                <strong>Split Circle</strong>: This circle was created through automatic splitting to maintain optimal group sizes.
            </div>
            """, unsafe_allow_html=True)
            
            # Add original circle information if available
            if 'original_circle_id' in circle_row:
                metadata["Original Circle"] = circle_row.get('original_circle_id', 'Unknown')
                
            # Add split letter if available
            if 'split_letter' in circle_row:
                metadata["Split Group"] = circle_row.get('split_letter', 'Unknown')
            elif '-SPLIT-' in circle_id:
                # Extract split letter from the end of the ID
                split_letter = circle_id[-1]
                metadata["Split Group"] = split_letter
        
        for key, value in metadata.items():
            st.write(f"**{key}:** {value}")
    
    # Second column: Members table
    with col2:
        st.subheader("Members")
        
        # Get member IDs - use the manager's dedicated method if available
        if manager:
            member_ids = manager.get_circle_members(selected_circle)
            if not member_ids:
                st.warning("No members found for this circle.")
                return
        else:  
            # Fall back to parsing from the DataFrame
            member_ids = []
            if 'members' in circle_row:
                # Handle both list and string representations
                if isinstance(circle_row['members'], list):
                    member_ids = circle_row['members']
                elif isinstance(circle_row['members'], str):
                    try:
                        # Try to evaluate if it's a string representation of a list
                        if circle_row['members'].startswith('['):
                            member_ids = eval(circle_row['members'])
                        else:
                            member_ids = [circle_row['members']]
                    except:
                        st.error(f"Could not parse member list: {circle_row['members']}")
            
            if not member_ids:
                st.warning("No members found for this circle.")
                return
        
        # Get member data - use the manager's dedicated method if available
        if manager:
            members_df = manager.get_circle_member_data(selected_circle)
            if members_df.empty:
                st.warning("Could not retrieve member data for this circle.")
                return
        else:
            # Fallback to manual lookup
            members_df = results_df[results_df['Encoded ID'].isin(member_ids)]
        
        # Create a display table
        display_cols = ['Last Family Name', 'First Given Name', 'Encoded ID', 
                        'Current_Region', 'Status', 'first_choice_time', 'host']
        
        # Filter to available columns
        display_cols = [col for col in display_cols if col in members_df.columns]
        
        # Import standardization utilities if not already imported
        if 'normalize_host_status' not in locals():
            from utils.data_standardization import normalize_host_status
        
        # Add standardized host status column if host column exists
        host_col = None
        for col in ['host', 'Host', 'willing_to_host']:
            if col in members_df.columns:
                host_col = col
                # Add normalized host status column
                members_df['Standardized Host'] = members_df[host_col].apply(normalize_host_status)
                # Add to display columns
                display_cols.append('Standardized Host')
                break
        
        if display_cols:
            st.dataframe(members_df[display_cols], use_container_width=True)
        else:
            st.warning("Member data doesn't contain the expected columns.")
        
        # Show host status analysis
        if host_col:
            with st.expander("Host Status Analysis"):
                st.write("### Host Status Distribution")
                
                # Show counts by standardized host status
                host_counts = members_df['Standardized Host'].value_counts().reset_index()
                host_counts.columns = ['Host Status', 'Count']
                
                # Create a small bar chart
                fig = px.bar(
                    host_counts,
                    x='Host Status',
                    y='Count',
                    color='Host Status',
                    title='Host Status Distribution',
                    color_discrete_map={
                        'ALWAYS': '#2E8B57',  # Sea Green
                        'SOMETIMES': '#4682B4',  # Steel Blue
                        'NEVER': '#CD5C5C'  # Indian Red
                    }
                )
                
                # Show the chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Compare with metadata
                always_count = (members_df['Standardized Host'] == 'ALWAYS').sum()
                sometimes_count = (members_df['Standardized Host'] == 'SOMETIMES').sum()
                
                st.write("### Host Counts Verification")
                st.write(f"**Calculated from member data:** {always_count} Always, {sometimes_count} Sometimes")
                st.write(f"**From circle metadata:** {circle_row.get('always_hosts', 0)} Always, {circle_row.get('sometimes_hosts', 0)} Sometimes")
                
                # Highlight discrepancies
                if always_count != circle_row.get('always_hosts', 0):
                    st.error(f"DISCREPANCY: Circle metadata shows {circle_row.get('always_hosts', 0)} Always Hosts but calculated count is {always_count}")
                
                if sometimes_count != circle_row.get('sometimes_hosts', 0):
                    st.error(f"DISCREPANCY: Circle metadata shows {circle_row.get('sometimes_hosts', 0)} Sometimes Hosts but calculated count is {sometimes_count}")
        
        # Debug info
        print(f"  Circle {selected_circle} has {len(member_ids)} member IDs")
        print(f"  Found {len(members_df)} matching records in results DataFrame")
        print(f"  Available member columns: {list(members_df.columns)}")
        if len(members_df) < len(member_ids):
            print(f"  ⚠️ WARNING: Could not find all member data. Expected {len(member_ids)} members but found {len(members_df)}.")
            missing_members = set(member_ids) - set(members_df['Encoded ID'].values if 'Encoded ID' in members_df.columns else [])
            print(f"  ⚠️ Missing members: {list(missing_members)[:5]}" + ("..." if len(missing_members) > 5 else ""))
    
    # Show visualizations specific to this circle
    st.subheader("Circle Analysis")
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2 = st.tabs(["Time Preferences", "Location Preferences"])
    
    with viz_tab1:
        # Time preference visualization
        st.write("Time Preferences Distribution")
        
        # Extract time preferences
        time_prefs = []
        for choice_num in range(1, 4):
            col_name = f'first_choice_time'
            if col_name in members_df.columns:
                prefs = members_df[col_name].dropna().tolist()
                time_prefs.extend(prefs)
        
        if time_prefs:
            # Create a simple display of common times
            from collections import Counter
            time_counts = Counter(time_prefs)
            
            # Create a DataFrame for plotting
            time_df = pd.DataFrame({
                'Time Preference': list(time_counts.keys()),
                'Count': list(time_counts.values())
            }).sort_values('Count', ascending=False)
            
            # Show as a table
            st.dataframe(time_df, use_container_width=True)
        else:
            st.info("No time preference data available for this circle.")
    
    with viz_tab2:
        # Location preference visualization
        st.write("Location Preferences Distribution")
        
        # Extract location preferences
        loc_prefs = []
        for choice_num in range(1, 4):
            col_name = f'first_choice_location'
            if col_name in members_df.columns:
                prefs = members_df[col_name].dropna().tolist()
                loc_prefs.extend(prefs)
        
        if loc_prefs:
            # Create a simple display of common locations
            from collections import Counter
            loc_counts = Counter(loc_prefs)
            
            # Create a DataFrame for plotting
            loc_df = pd.DataFrame({
                'Location Preference': list(loc_counts.keys()),
                'Count': list(loc_counts.values())
            }).sort_values('Count', ascending=False)
            
            # Show as a table
            st.dataframe(loc_df, use_container_width=True)
        else:
            st.info("No location preference data available for this circle.")


def render_participant_details():
    """Render detailed information about individual participants"""
    if 'results' not in st.session_state or st.session_state.results is None:
        st.warning("No participant data available. Please run the matching algorithm first.")
        return
    
    # Get the data
    results_df = st.session_state.results.copy()
    
    # Filter out participants with null Encoded IDs
    from utils.helpers import get_valid_participants
    results_df = get_valid_participants(results_df)
    print(f"🔍 Participant details: Using {len(results_df)} valid participants with non-null Encoded IDs")
    
    # Create a search box for participants
    search_type = st.radio("Search by", ["Name", "ID"], horizontal=True)
    
    if search_type == "Name":
        # Create search boxes for first and last name
        col1, col2 = st.columns(2)
        
        with col1:
            first_name_col = 'First Given Name' if 'First Given Name' in results_df.columns else None
            if first_name_col:
                first_name_options = [''] + sorted(results_df[first_name_col].dropna().unique().tolist())
                first_name = st.selectbox("First Name", options=first_name_options, key="participant_first_name_filter")
            else:
                st.warning("First name column not found in data")
                first_name = None
        
        with col2:
            last_name_col = 'Last Family Name' if 'Last Family Name' in results_df.columns else None
            if last_name_col:
                last_name_options = [''] + sorted(results_df[last_name_col].dropna().unique().tolist())
                last_name = st.selectbox("Last Name", options=last_name_options, key="participant_last_name_filter")
            else:
                st.warning("Last name column not found in data")
                last_name = None
        
        # Filter based on name
        filtered_df = results_df.copy()
        if first_name and first_name_col:
            filtered_df = filtered_df[filtered_df[first_name_col] == first_name]
        if last_name and last_name_col:
            filtered_df = filtered_df[filtered_df[last_name_col] == last_name]
    else:
        # Search by ID
        id_col = 'Encoded ID' if 'Encoded ID' in results_df.columns else None
        if id_col:
            id_options = [''] + sorted(results_df[id_col].dropna().unique().tolist())
            participant_id = st.selectbox("Participant ID", options=id_options, key="participant_id_filter")
            
            # Filter based on ID
            filtered_df = results_df[results_df[id_col] == participant_id] if participant_id else results_df
        else:
            st.warning("ID column not found in data")
            filtered_df = results_df
    
    # Show the filtered results
    if len(filtered_df) == 0:
        st.info("No participants found matching the search criteria.")
        return
    elif len(filtered_df) > 10:
        st.warning(f"Found {len(filtered_df)} participants matching the search criteria. Please refine your search.")
        
        # Show a sample of the results
        st.dataframe(filtered_df.head(10), use_container_width=True)
        return
    
    # Show detailed information for each matching participant
    for idx, participant in filtered_df.iterrows():
        # Get the participant ID
        participant_id = participant.get('Encoded ID', f"Participant {idx}")
        
        # Create an expander for this participant
        with st.expander(f"Participant: {participant_id}", expanded=True):
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            # Column 1: Basic information
            with col1:
                st.subheader("Basic Information")
                
                # Show basic participant info
                basic_fields = ['First Given Name', 'Last Family Name', 'Encoded ID', 
                                'Current_Region', 'Status']
                
                for field in basic_fields:
                    if field in participant:
                        st.write(f"**{field.replace('_', ' ').title()}:** {participant[field]}")
            
            # Column 2: Match information
            with col2:
                st.subheader("Match Information")
                
                # Show circle assignment if matched
                if 'proposed_NEW_circles_id' in participant:
                    circle_id = participant['proposed_NEW_circles_id']
                    
                    if circle_id == 'UNMATCHED':
                        st.error("**Status:** Unmatched")
                        
                        # Show reason if available
                        if 'unmatched_reason' in participant:
                            st.write(f"**Reason:** {participant['unmatched_reason']}")
                    else:
                        st.success(f"**Assigned Circle:** {circle_id}")
                        
                        # Show circle details if available
                        # First try using the CircleMetadataManager if available
                        from utils.circle_metadata_manager import get_manager_from_session_state
                        manager = get_manager_from_session_state(st.session_state) if 'circle_manager' in st.session_state else None
                        
                        if manager:
                            # Use CircleMetadataManager to get circle data
                            circle_data = manager.get_circle_data(circle_id)
                            if circle_data:
                                # Create a dict for consistent display
                                circle_row = circle_data
                                circle_info_available = True
                            else:
                                circle_info_available = False
                        else:
                            # Fall back to direct session state access
                            if ('matched_circles' in st.session_state and 
                                st.session_state.matched_circles is not None and
                                'circle_id' in st.session_state.matched_circles.columns):
                                
                                circles_df = st.session_state.matched_circles
                                circle_info = circles_df[circles_df['circle_id'] == circle_id]
                                
                                if not circle_info.empty:
                                    circle_row = circle_info.iloc[0]
                                    circle_info_available = True
                                else:
                                    circle_info_available = False
                            else:
                                circle_info_available = False
                        
                        if circle_info_available:
                                st.write(f"**Meeting Time:** {circle_row.get('meeting_time', 'Not specified')}")
                                st.write(f"**Meeting Location:** {circle_row.get('meeting_location', 'Not specified')}")
                                st.write(f"**Circle Size:** {circle_row.get('member_count', 'Unknown')}")
            
            # Show preferences section
            st.subheader("Preferences")
            pref_cols = st.columns(2)
            
            with pref_cols[0]:
                st.write("**Time Preferences**")
                for i in range(1, 4):
                    time_col = f"{['first', 'second', 'third'][i-1]}_choice_time"
                    if time_col in participant and pd.notna(participant[time_col]):
                        st.write(f"{i}. {participant[time_col]}")
            
            with pref_cols[1]:
                st.write("**Location Preferences**")
                for i in range(1, 4):
                    loc_col = f"{['first', 'second', 'third'][i-1]}_choice_location"
                    if loc_col in participant and pd.notna(participant[loc_col]):
                        st.write(f"{i}. {participant[loc_col]}")


def render_visualizations():
    """Render visualizations of the matching results"""
    if ('matched_circles' not in st.session_state or 
        st.session_state.matched_circles is None or
        (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty) or
        'results' not in st.session_state or 
        st.session_state.results is None):
        st.warning("No matching results available. Please run the matching algorithm first.")
        return
    
    st.subheader("Matching Visualizations")
    
    # Get the data with CircleMetadataManager if available
    from utils.circle_metadata_manager import get_manager_from_session_state
    
    # Try to get the circle manager first
    manager = get_manager_from_session_state(st.session_state) if 'circle_manager' in st.session_state else None
    
    # Get circle data - from manager or directly from session state
    if manager:
        print("\n🔍 VISUALIZATIONS DEBUG (Using CircleMetadataManager):")
        circles_df = manager.get_circles_dataframe()
        print(f"  Retrieved {len(circles_df)} circles from CircleMetadataManager")
    else:
        print("\n🔍 VISUALIZATIONS DEBUG (Using session state directly):")
        circles_df = st.session_state.matched_circles.copy()
        
    # Get participant data from session state
    results_df = st.session_state.results.copy()
    
    # Filter out participants with null Encoded IDs
    from utils.helpers import get_valid_participants
    results_df = get_valid_participants(results_df)
    print(f"🔍 Visualizations: Using {len(results_df)} valid participants with non-null Encoded IDs")
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Circle Composition", "Regional Distribution", "Preference Satisfaction"])
    
    with viz_tab1:
        # Circle composition visualization
        st.write("Circle Composition Analysis")
        
        # Circle size distribution
        if 'member_count' in circles_df.columns:
            # Count circles by size
            size_counts = circles_df['member_count'].value_counts().sort_index()
            
            # Create a DataFrame for plotting
            size_df = pd.DataFrame({
                'Circle Size': size_counts.index,
                'Number of Circles': size_counts.values
            })
            
            # Create a bar chart
            fig = px.bar(
                size_df,
                x='Circle Size',
                y='Number of Circles',
                title='Distribution of Circle Sizes',
                text='Number of Circles',
                color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
            )
            
            # Format
            fig.update_traces(textposition='outside')
            fig.update_layout(
                xaxis=dict(
                    title="Number of Members",
                    tickmode='linear',
                    dtick=1
                ),
                yaxis_title="Number of Circles"
            )
            
            # Plot
            st.plotly_chart(fig, use_container_width=True, key="plot_2113")
            
            # Calculate statistics
            total_circles = len(circles_df)
            total_members = circles_df['member_count'].sum()
            avg_size = circles_df['member_count'].mean()
            
            # Show metrics
            cols = st.columns(3)
            cols[0].metric("Total Circles", total_circles)
            cols[1].metric("Total Matched Participants", total_members)
            cols[2].metric("Average Circle Size", f"{avg_size:.1f}")
        else:
            st.warning("Circle size data not available.")
    
    with viz_tab2:
        # Regional distribution visualization
        st.write("Regional Distribution Analysis")
        
        # Check for region column
        region_col = None
        if 'Current_Region' in results_df.columns:
            region_col = 'Current_Region'
        elif 'Region' in results_df.columns:
            region_col = 'Region'
        
        if region_col:
            # Count participants by region
            region_counts = results_df[region_col].value_counts()
            
            # Create a DataFrame for plotting
            region_df = pd.DataFrame({
                'Region': region_counts.index,
                'Count': region_counts.values
            }).sort_values('Count', ascending=False)
            
            # Create a bar chart
            fig = px.bar(
                region_df,
                x='Region',
                y='Count',
                title='Participant Distribution by Region',
                text='Count',
                color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
            )
            
            # Format
            fig.update_traces(textposition='outside')
            fig.update_layout(
                xaxis_title="Region",
                yaxis_title="Number of Participants"
            )
            
            # Plot
            st.plotly_chart(fig, use_container_width=True, key="plot_2167")
            
            # Show table
            st.dataframe(region_df, use_container_width=True)
            
            # Create a matched vs unmatched breakdown by region
            if 'proposed_NEW_circles_id' in results_df.columns:
                # Create a function to determine match status
                def get_match_status(circle_id):
                    return "Unmatched" if circle_id == "UNMATCHED" else "Matched"
                
                # Add match status column
                results_df['Match Status'] = results_df['proposed_NEW_circles_id'].apply(get_match_status)
                
                # Create a crosstab
                region_match = pd.crosstab(
                    results_df[region_col], 
                    results_df['Match Status'],
                    normalize='index'
                ) * 100
                
                # Sort by match rate
                if 'Matched' in region_match.columns:
                    region_match = region_match.sort_values('Matched', ascending=False)
                
                # Create a bar chart
                fig = px.bar(
                    region_match.reset_index(),
                    x=region_col,
                    y=['Matched', 'Unmatched'] if 'Matched' in region_match.columns and 'Unmatched' in region_match.columns else region_match.columns,
                    title='Match Success Rate by Region (%)',
                    barmode='stack',
                    color_discrete_sequence=['#175E54', '#820000']  # Stanford secondary colors
                )
                
                # Format
                fig.update_layout(
                    xaxis_title="Region",
                    yaxis_title="Percentage",
                    yaxis=dict(ticksuffix='%')
                )
                
                # Plot
                st.plotly_chart(fig, use_container_width=True, key="plot_2210")
        else:
            st.warning("Region data not available.")
    
    with viz_tab3:
        # Preference satisfaction visualization
        st.write("Preference Satisfaction Analysis")
        
        # Check if we have the necessary preference data
        if ('proposed_NEW_circles_id' in results_df.columns and 
            'matched_circles' in st.session_state):
            
            # Get only matched participants
            matched_df = results_df[results_df['proposed_NEW_circles_id'] != 'UNMATCHED'].copy()
            
            # Time preference satisfaction
            st.subheader("Time Preference Satisfaction")
            
            # Find columns with time choices
            time_cols = [col for col in matched_df.columns if 'choice_time' in col]
            
            if time_cols and 'proposed_NEW_circles_id' in matched_df.columns:
                # Import CircleMetadataManager
                from utils.circle_metadata_manager import get_manager_from_session_state
                
                # Try to get the circle manager
                manager = get_manager_from_session_state(st.session_state) if 'circle_manager' in st.session_state else None
                
                # Add debug logging for consistency
                if manager:
                    print("\n🔍 TIME PREFERENCE DEBUG (Using CircleMetadataManager)")
                else:
                    print("\n🔍 TIME PREFERENCE DEBUG (Using session state directly)")

                
                # Function to check if assigned time matches preferences
                def check_time_match(row):
                    circle_id = row['proposed_NEW_circles_id']
                    if circle_id == 'UNMATCHED' or pd.isna(circle_id):
                        return None
                    
                    # Get the assigned circle's time - first try using CircleMetadataManager
                    if manager:
                        circle_data = manager.get_circle_data(circle_id)
                        if circle_data and 'meeting_time' in circle_data:
                            assigned_time = circle_data['meeting_time']
                        else:
                            # Fall back to DataFrame lookup
                            circle_info = circles_df[circles_df['circle_id'] == circle_id]
                            if circle_info.empty or 'meeting_time' not in circle_info.columns:
                                return None
                            assigned_time = circle_info.iloc[0]['meeting_time']
                    else:
                        # Fall back to DataFrame lookup
                        circle_info = circles_df[circles_df['circle_id'] == circle_id]
                        if circle_info.empty or 'meeting_time' not in circle_info.columns:
                            return None
                        assigned_time = circle_info.iloc[0]['meeting_time']
                    
                    # Check for missing time value
                    if pd.isna(assigned_time):
                        return None
                    
                    # Check each preference
                    for i, col in enumerate(time_cols):
                        if col in row and pd.notna(row[col]):
                            preferred_time = row[col]
                            # Very simple check - just look for exact match or substring
                            if preferred_time == assigned_time or preferred_time in assigned_time or assigned_time in preferred_time:
                                return i+1  # Return the preference number (1, 2, 3)
                    
                    return 0  # No match
                
                # Apply the function
                matched_df['time_preference_match'] = matched_df.apply(check_time_match, axis=1)
                
                # Count by match result
                time_match_counts = matched_df['time_preference_match'].value_counts().sort_index()
                
                # Create labels
                match_labels = {
                    0: "No Match",
                    1: "1st Choice",
                    2: "2nd Choice",
                    3: "3rd Choice",
                    None: "Not Available"
                }
                
                # Create a DataFrame for plotting
                match_df = pd.DataFrame({
                    'Preference Match': [match_labels.get(idx, str(idx)) for idx in time_match_counts.index],
                    'Count': time_match_counts.values
                })
                
                # Create a pie chart
                fig = px.pie(
                    match_df,
                    names='Preference Match',
                    values='Count',
                    title='Time Preference Satisfaction',
                    color_discrete_sequence=px.colors.qualitative.D3
                )
                
                # Format
                fig.update_traces(textposition='inside', textinfo='percent+label')
                
                # Plot
                st.plotly_chart(fig, use_container_width=True, key="plot_2291")
                
                # Calculate the percentage of participants who got one of their preferences
                matched_count = sum(time_match_counts.get(i, 0) for i in [1, 2, 3])
                total_count = sum(time_match_counts.values)
                match_pct = (matched_count / total_count) * 100 if total_count > 0 else 0
                
                st.metric("Participants receiving a preferred time", 
                         f"{matched_count} out of {total_count} ({match_pct:.1f}%)")
            else:
                st.warning("Time preference data not available.")
            
            # Location preference satisfaction
            st.subheader("Location Preference Satisfaction")
            
            # Find columns with location choices
            loc_cols = [col for col in matched_df.columns if 'choice_location' in col]
            
            if loc_cols and 'proposed_NEW_circles_id' in matched_df.columns:
                # Import CircleMetadataManager if not already imported
                if 'get_manager_from_session_state' not in locals():
                    from utils.circle_metadata_manager import get_manager_from_session_state
                    
                # Try to get the circle manager if not already obtained
                if 'manager' not in locals() or manager is None:
                    manager = get_manager_from_session_state(st.session_state) if 'circle_manager' in st.session_state else None
                    
                # Add debug logging for consistency
                if manager:
                    print("\n🔍 LOCATION PREFERENCE DEBUG (Using CircleMetadataManager)")
                else:
                    print("\n🔍 LOCATION PREFERENCE DEBUG (Using session state directly)")

                
                # Function to check if assigned location matches preferences
                def check_location_match(row):
                    circle_id = row['proposed_NEW_circles_id']
                    if circle_id == 'UNMATCHED' or pd.isna(circle_id):
                        return None
                    
                    # Get the assigned circle's location - first try using CircleMetadataManager
                    if manager:
                        circle_data = manager.get_circle_data(circle_id)
                        if circle_data and 'meeting_location' in circle_data:
                            assigned_loc = circle_data['meeting_location']
                        else:
                            # Fall back to DataFrame lookup
                            circle_info = circles_df[circles_df['circle_id'] == circle_id]
                            if circle_info.empty or 'meeting_location' not in circle_info.columns:
                                return None
                            assigned_loc = circle_info.iloc[0]['meeting_location']
                    else:
                        # Fall back to DataFrame lookup
                        circle_info = circles_df[circles_df['circle_id'] == circle_id]
                        if circle_info.empty or 'meeting_location' not in circle_info.columns:
                            return None
                        assigned_loc = circle_info.iloc[0]['meeting_location']
                    
                    # Check for missing location value
                    if pd.isna(assigned_loc):
                        return None
                    
                    # Check each preference
                    for i, col in enumerate(loc_cols):
                        if col in row and pd.notna(row[col]):
                            preferred_loc = row[col]
                            # Very simple check - just look for exact match or substring
                            if preferred_loc == assigned_loc or preferred_loc in assigned_loc or assigned_loc in preferred_loc:
                                return i+1  # Return the preference number (1, 2, 3)
                    
                    return 0  # No match
                
                # Apply the function
                matched_df['location_preference_match'] = matched_df.apply(check_location_match, axis=1)
                
                # Count by match result
                loc_match_counts = matched_df['location_preference_match'].value_counts().sort_index()
                
                # Create a DataFrame for plotting
                match_df = pd.DataFrame({
                    'Preference Match': [match_labels.get(idx, str(idx)) for idx in loc_match_counts.index],
                    'Count': loc_match_counts.values
                })
                
                # Create a pie chart
                fig = px.pie(
                    match_df,
                    names='Preference Match',
                    values='Count',
                    title='Location Preference Satisfaction',
                    color_discrete_sequence=px.colors.qualitative.D3
                )
                
                # Format
                fig.update_traces(textposition='inside', textinfo='percent+label')
                
                # Plot
                st.plotly_chart(fig, use_container_width=True, key="plot_2360")
                
                # Calculate the percentage of participants who got one of their preferences
                matched_count = sum(loc_match_counts.get(i, 0) for i in [1, 2, 3])
                total_count = sum(loc_match_counts.values)
                match_pct = (matched_count / total_count) * 100 if total_count > 0 else 0
                
                st.metric("Participants receiving a preferred location", 
                         f"{matched_count} out of {total_count} ({match_pct:.1f}%)")
            else:
                st.warning("Location preference data not available.")
        else:
            st.warning("Preference satisfaction analysis requires matched circle data.")
            
def render_children_analysis(data):
    """Render the Children analysis visualizations"""
    st.subheader("Children Analysis")
    
    # Check if data is None
    if data is None:
        st.warning("No data available for analysis. Please upload participant data.")
        return
    
    # Create a copy to work with
    df = data.copy()
    
    # Try to find Children column
    children_col = None
    for col in df.columns:
        if "children" in col.lower():
            children_col = col
            break
    
    # Define function to categorize children status
    def categorize_children(children_value):
        if pd.isna(children_value):
            return None
            
        children_value = str(children_value).strip()
        
        # No children category
        if "No children" in children_value:
            return "No children"
        
        # First time <=5s
        if any(term in children_value for term in ["Children 5 and under", "Children under 5", "Pregnant"]):
            return "First time <=5s"
        
        # First time 6-18s
        if any(term in children_value for term in ["Children 6 through 17", "Children 6-18"]) and not any(term in children_value for term in ["Adult Children", "Adult children"]):
            return "First time 6-18s"
        
        # Adult children / all else
        return "Adult children / all else"
    
    # Check if we need to create Children Category
    if 'Children_Category' not in df.columns:
        if children_col:
            # Apply the categorization function
            df['Children_Category'] = df[children_col].apply(categorize_children)
            
            # Update the underlying processed data if it exists
            if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
                # Check if the column exists in the original processed data
                if children_col in st.session_state.processed_data.columns:
                    # Create a temporary dataframe with just ID and Children_Category
                    if 'Encoded ID' in df.columns and 'Children_Category' in df.columns:
                        # Get the mapping between ID and Children_Category
                        mapping_df = df[['Encoded ID', 'Children_Category']].dropna(subset=['Children_Category'])
                        
                        # Create a dictionary for quick lookup
                        children_category_map = dict(zip(mapping_df['Encoded ID'], mapping_df['Children_Category']))
                        
                        # Update processed_data with the Children_Category
                        if 'Encoded ID' in st.session_state.processed_data.columns:
                            st.session_state.processed_data['Children_Category'] = st.session_state.processed_data['Encoded ID'].map(children_category_map)
                
                # Also update the results dataframe if it exists
                if 'results' in st.session_state and st.session_state.results is not None:
                    # Check if the column exists in the results dataframe
                    if children_col in st.session_state.results.columns:
                        # Update the results dataframe with the Children_Category
                        if 'Encoded ID' in st.session_state.results.columns:
                            # Create a temporary dataframe with just ID and Children_Category
                            mapping_df = df[['Encoded ID', 'Children_Category']].dropna(subset=['Children_Category'])
                            
                            # Create a dictionary for quick lookup
                            children_category_map = dict(zip(mapping_df['Encoded ID'], mapping_df['Children_Category']))
                            
                            # Update results with the Children_Category
                            st.session_state.results['Children_Category'] = st.session_state.results['Encoded ID'].map(children_category_map)
        else:
            st.warning("Children data is not available. Please ensure Children data was included in the uploaded file.")
            return
    
    # Filter out rows with missing Children Category
    df = df[df['Children_Category'].notna()]
    
    if len(df) == 0:
        st.warning("No Children Category data is available after filtering.")
        return
    
    # Define the proper order for Children Categories
    children_order = ["No children", "First time <=5s", "First time 6-18s", "Adult children / all else"]
    
    # FIRST: Display diversity within circles IF we have matched circles
    if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
        if not (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty):
            render_children_diversity_histogram()
        else:
            st.info("Run the matching algorithm to see the Children diversity within circles.")
    else:
        st.info("Run the matching algorithm to see the Children diversity within circles.")
    
    # THIRD: Display Distribution of Children Categories
    st.subheader("Distribution of Children")
    
    # Count by Children Category
    children_counts = df['Children_Category'].value_counts().reindex(children_order).fillna(0).astype(int)
    
    # Create a DataFrame for plotting
    children_df = pd.DataFrame({
        'Children Category': children_counts.index,
        'Count': children_counts.values
    })
    
    # Create histogram using plotly with Stanford cardinal red color
    fig = px.bar(
        children_df,
        x='Children Category',
        y='Count',
        title='Distribution of Children Categories',
        text='Count',  # Display count values on bars
        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
    )
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': children_order},
        xaxis_title="Children Category",
        yaxis_title="Count of Participants"
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True, key="children_distribution")
    
    # FOURTH: Create a breakdown by Status if Status column exists
    if 'Status' in df.columns:
        st.subheader("Children by Status")
        
        # Create a crosstab of Children Category vs Status
        status_children = pd.crosstab(
            df['Children_Category'], 
            df['Status'],
            rownames=['Children Category'],
            colnames=['Status']
        ).reindex(children_order)
        
        # Add a Total column
        status_children['Total'] = status_children.sum(axis=1)
        
        # Calculate percentages
        for col in status_children.columns:
            if col != 'Total':
                status_children[f'{col} %'] = (status_children[col] / status_children['Total'] * 100).round(1)
        
        # Reorder columns to group counts with percentages
        cols = []
        for status in status_children.columns:
            if status != 'Total' and not status.endswith(' %'):
                cols.append(status)
                cols.append(f'{status} %')
        cols.append('Total')
        
        # Show the table
        st.dataframe(status_children[cols], use_container_width=True)
        
        # Create a stacked bar chart
        fig = px.bar(
            status_children.reset_index(),
            x='Children Category',
            y=[col for col in status_children.columns if col != 'Total' and not col.endswith(' %')],
            title='Children Distribution by Status',
            barmode='stack',
            color_discrete_sequence=['#8C1515', '#2E2D29']  # Stanford colors
        )
        
        # Customize layout
        fig.update_layout(
            xaxis={'categoryorder': 'array', 'categoryarray': children_order},
            xaxis_title="Children Category",
            yaxis_title="Count of Participants",
            legend_title="Status"
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True, key="children_status_distribution")
    
    # Add definition section
    st.subheader("Definition")
    st.write(""""Children Category" consolidates responses to as follows:
- "No children": use this if the response to Children contains "No children". E.g., if the cell value is "No children,Other (specify)", Children Category is "No children"
- "First time <=5s": Use this if Children is ("Children 5 and under" OR "Children under 5" OR "Pregnant")
- "First time 6-18s": Use this if Children contains ("Children 6 through 17", "Children 6-18") AND does NOT contain "Adult Children" or "Adult children")
- "Adult children / all else": All other Children values.""")

def render_children_diversity_histogram():
    """
    Create a histogram showing the distribution of circles based on 
    the number of different children categories they contain
    """
    # First check if we have the necessary data
    if 'matched_circles' not in st.session_state or st.session_state.matched_circles is None:
        st.warning("No matched circles data available to analyze children diversity.")
        return
        
    # Check if the matched_circles dataframe is empty
    if hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty:
        st.warning("No matched circles data available (empty dataframe).")
        return
        
    if 'results' not in st.session_state or st.session_state.results is None:
        st.warning("No results data available to analyze children diversity.")
        return
    
    # Get the circle data
    circles_df = st.session_state.matched_circles.copy()
    results_df = st.session_state.results.copy()
    
    # Create score column if it doesn't exist
    if 'children_score' not in circles_df.columns:
        circles_df['children_score'] = 0
    
    # Check if Children_Category doesn't exist, add it from the Children column
    if 'Children_Category' not in results_df.columns:
        # Find the Children column in the results dataframe
        children_col = None
        for col in results_df.columns:
            if "children" in col.lower():
                children_col = col
                break
        
        if children_col:
            # Define function to categorize children status
            def categorize_children(children_value):
                if pd.isna(children_value):
                    return None
                    
                children_value = str(children_value).strip()
                
                # No children category
                if "No children" in children_value:
                    return "No children"
                
                # First time <=5s
                if any(term in children_value for term in ["Children 5 and under", "Children under 5", "Pregnant"]):
                    return "First time <=5s"
                
                # First time 6-18s
                if any(term in children_value for term in ["Children 6 through 17", "Children 6-18"]) and not any(term in children_value for term in ["Adult Children", "Adult children"]):
                    return "First time 6-18s"
                
                # Adult children / all else
                return "Adult children / all else"
            
            # Apply the categorization function to create the column in results_df
            results_df['Children_Category'] = results_df[children_col].apply(categorize_children)
            
            # Update session state with the modified results dataframe
            st.session_state.results = results_df
    
    # Filter out circles with no members
    if 'member_count' not in circles_df.columns:
        st.warning("Circles data does not have member_count column")
        return
    
    circles_df = circles_df[circles_df['member_count'] > 0]
    
    if len(circles_df) == 0:
        st.warning("No circles with members available for analysis.")
        return
    
    # Dictionary to track unique children categories per circle
    circle_children_counts = {}
    circle_children_diversity_scores = {}
    circles_with_no_children_data = []
    circles_with_parsing_errors = []
    circles_included = []
    
    # Track special debug circles
    debug_circles = ['IP-ATL-1', 'IP-BOS-01']
    
    # Get children data for each member of each circle
    for idx, circle_row in circles_df.iterrows():
        circle_id = circle_row['circle_id']
        is_debug_circle = circle_id in debug_circles
        
        # Initialize empty set to track unique children categories
        unique_children_categories = set()
        has_parsing_error = False
        
        # IMPROVED APPROACH: Get the list of members for this circle
        # Method 1: Try to get members from the circle_row['members']
        members_from_row = []
        if 'members' in circle_row and circle_row['members'] and not pd.isna(circle_row['members']).all():
            # For list representation
            if isinstance(circle_row['members'], list):
                members_from_row = [m for m in circle_row['members'] if not pd.isna(m)]
            # For string representation - convert to list
            elif isinstance(circle_row['members'], str):
                try:
                    if circle_row['members'].startswith('['):
                        members_from_row = [m for m in eval(circle_row['members']) if not pd.isna(m)]
                    else:
                        members_from_row = [circle_row['members']]
                except Exception as e:
                    # Record parsing errors
                    has_parsing_error = True
                    circles_with_parsing_errors.append(circle_id)
            else:
                has_parsing_error = True
                circles_with_parsing_errors.append(circle_id)
        
        # Method 2: Get members by looking up the circle_id in the results dataframe
        members_from_lookup = []
        if 'proposed_NEW_circles_id' in results_df.columns:
            # Find all participants assigned to this circle
            circle_members = results_df[results_df['proposed_NEW_circles_id'] == circle_id]
            if not circle_members.empty and 'Encoded ID' in circle_members.columns:
                members_from_lookup = circle_members['Encoded ID'].dropna().tolist()
        
        # Combine both methods, prioritizing non-empty results
        member_ids = []
        if members_from_lookup:
            member_ids = members_from_lookup
        else:
            member_ids = members_from_row
        
        if is_debug_circle:
            print(f"CHILDREN DEBUG - {circle_id}: Found {len(member_ids)} members using improved extraction")
                
        # For each member, look up their children category in results_df
        for member_id in member_ids:
            # Skip NaN or invalid member IDs
            if pd.isna(member_id):
                continue
                
            # Try exact match first
            member_data = results_df[results_df['Encoded ID'] == member_id]
            
            # If no match, try converting both to strings for comparison
            if member_data.empty:
                # Convert to string and try again
                member_data = results_df[results_df['Encoded ID'].astype(str) == str(member_id)]
                
                # If still no match and member_id has numeric format
                if member_data.empty and str(member_id).replace('.', '', 1).isdigit():
                    try:
                        # Try as float
                        float_id = float(member_id)
                        member_data = results_df[results_df['Encoded ID'].astype(float) == float_id]
                        
                        # Try as int if it's a whole number
                        if member_data.empty and float_id.is_integer():
                            int_id = int(float_id)
                            member_data = results_df[results_df['Encoded ID'].astype(int) == int_id]
                    except:
                        pass
            
            if not member_data.empty and 'Children_Category' in member_data.columns:
                children_category = member_data['Children_Category'].iloc[0]
                if pd.notna(children_category):
                    unique_children_categories.add(children_category)
        
        # Store the count of unique children categories for this circle
        if unique_children_categories:  # Only include if there's at least one valid category
            count = len(unique_children_categories)
            circle_children_counts[circle_id] = count
            # Calculate diversity score: 1 point if everyone is in the same category,
            # 2 points if two categories, 3 points if three categories, etc.
            circle_children_diversity_scores[circle_id] = count
            
            # CRITICAL FIX: Update the dataframe with the children score for this circle
            circles_df.at[idx, 'children_score'] = count
            
            if is_debug_circle:
                print(f"CHILDREN DEBUG - {circle_id}: Updated score to {count} (found {len(unique_children_categories)} unique categories)")
        else:
            # Circle has no valid children data - include with 0 count
            circle_children_counts[circle_id] = 0
            circle_children_diversity_scores[circle_id] = 0
            circles_df.at[idx, 'children_score'] = 0
            
            if is_debug_circle:
                print(f"CHILDREN DEBUG - {circle_id}: No children data found, set score to 0")
            
            if not has_parsing_error:
                circles_with_no_children_data.append(circle_id)
    
    # Create histogram data from the children counts
    if not circle_children_counts:
        st.warning("No children data available for circles.")
        return
        
    # Count circles by number of unique children categories
    diversity_counts = pd.Series(circle_children_counts).value_counts().sort_index()
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Number of Children Categories': diversity_counts.index,
        'Number of Circles': diversity_counts.values
    })
    
    st.subheader("Children Diversity Within Circles")
    
    # Create histogram using plotly with Stanford cardinal red color
    fig = px.bar(
        plot_df,
        x='Number of Children Categories',
        y='Number of Circles',
        title='Distribution of Circles by Number of Children Categories',
        text='Number of Circles',  # Display count values on bars
        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
    )
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis=dict(
            title="Number of Different Children Categories",
            tickmode='linear',
            dtick=1,  # Force integer labels
            range=[0.5, 4.5]  # Since we have 4 categories max
        ),
        yaxis_title="Number of Circles"
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True, key="children_diversity_histogram")
    
    # Show a table with the data
    st.caption("Data table:")
    st.dataframe(plot_df, hide_index=True)
    
    # Display the Children diversity score
    st.subheader("Children Diversity Score")
    st.write("""
    For each circle, the children diversity score is calculated as follows:
    - 1 point: All members in the same children category
    - 2 points: Members from two different children categories
    - 3 points: Members from three different children categories
    - 4 points: Members from all four children categories
    """)
    
    # Calculate average and total diversity scores
    total_diversity_score = sum(circle_children_diversity_scores.values()) if circle_children_diversity_scores else 0
    avg_diversity_score = total_diversity_score / len(circle_children_diversity_scores) if circle_children_diversity_scores else 0
    
    # Store the total score in session state for use in the Match tab
    st.session_state.children_diversity_score = total_diversity_score
    
    # Create two columns for the metrics
    col1, col2 = st.columns(2)
    
    # Show average and total scores
    with col1:
        st.metric("Average Children Diversity Score", f"{avg_diversity_score:.2f}")
    with col2:
        st.metric("Total Children Diversity Score", f"{total_diversity_score}")
    
    # Add a brief explanation
    total_circles = sum(diversity_counts.values)
    diverse_circles = sum(diversity_counts[diversity_counts.index > 1].values)
    diverse_pct = (diverse_circles / total_circles * 100) if total_circles > 0 else 0
    
    st.write(f"Out of {total_circles} total circles, {diverse_circles} ({diverse_pct:.1f}%) contain members from multiple children categories.")
    
    # CRITICAL FIX: Update the session state with our modified circles_df that now has diversity scores
    st.session_state.matched_circles = circles_df
    print(f"CHILDREN HISTOGRAM UPDATE - Updated session state matched_circles with calculated children scores for {len(circles_df)} circles")

def render_racial_identity_diversity_histogram():
    """
    Create a histogram showing the distribution of circles based on 
    the number of different racial identity categories they contain
    """
    # First check if we have the necessary data
    if 'matched_circles' not in st.session_state or st.session_state.matched_circles is None:
        st.warning("No matched circles data available to analyze racial identity diversity.")
        return
        
    # Check if the matched_circles dataframe is empty
    if hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty:
        st.warning("No matched circles data available (empty dataframe).")
        return
        
    if 'results' not in st.session_state or st.session_state.results is None:
        st.warning("No results data available to analyze racial identity diversity.")
        return
    
    # Get the circle data
    circles_df = st.session_state.matched_circles.copy()
    results_df = st.session_state.results.copy()
    
    # Create score column if it doesn't exist
    if 'racial_identity_score' not in circles_df.columns:
        circles_df['racial_identity_score'] = 0
    
    # Filter out circles with no members
    if 'member_count' not in circles_df.columns:
        st.warning("Circles data does not have member_count column")
        return
    
    circles_df = circles_df[circles_df['member_count'] > 0]
    
    if len(circles_df) == 0:
        st.warning("No circles with members available for analysis.")
        return
    
    # Dictionary to track unique racial identities per circle
    circle_racial_identity_counts = {}
    circle_racial_identity_diversity_scores = {}
    circles_with_no_racial_data = []
    circles_with_parsing_errors = []
    circles_included = []
    
    # Track special debug circles
    debug_circles = ['IP-ATL-1', 'IP-BOS-01']
    
    # Get racial identity data for each member of each circle
    for idx, circle_row in circles_df.iterrows():
        circle_id = circle_row['circle_id']
        is_debug_circle = circle_id in debug_circles
        
        # Initialize empty set to track unique racial identities
        unique_racial_identities = set()
        has_parsing_error = False
        
        # IMPROVED APPROACH: Get the list of members for this circle
        # Method 1: Try to get members from the circle_row['members']
        members_from_row = []
        if 'members' in circle_row and circle_row['members'] and not pd.isna(circle_row['members']).all():
            # For list representation
            if isinstance(circle_row['members'], list):
                members_from_row = [m for m in circle_row['members'] if not pd.isna(m)]
            # For string representation - convert to list
            elif isinstance(circle_row['members'], str):
                try:
                    if circle_row['members'].startswith('['):
                        members_from_row = [m for m in eval(circle_row['members']) if not pd.isna(m)]
                    else:
                        members_from_row = [circle_row['members']]
                except Exception as e:
                    # Record parsing errors
                    has_parsing_error = True
                    circles_with_parsing_errors.append(circle_id)
            else:
                has_parsing_error = True
                circles_with_parsing_errors.append(circle_id)
        
        # Method 2: Get members by looking up the circle_id in the results dataframe
        members_from_lookup = []
        if 'proposed_NEW_circles_id' in results_df.columns:
            # Find all participants assigned to this circle
            circle_members = results_df[results_df['proposed_NEW_circles_id'] == circle_id]
            if not circle_members.empty and 'Encoded ID' in circle_members.columns:
                members_from_lookup = circle_members['Encoded ID'].dropna().tolist()
        
        # Combine both methods, prioritizing non-empty results
        member_ids = []
        if members_from_lookup:
            member_ids = members_from_lookup
        else:
            member_ids = members_from_row
        
        if is_debug_circle:
            print(f"RACIAL IDENTITY DEBUG - {circle_id}: Found {len(member_ids)} members using improved extraction")
                
        # For each member, look up their racial identity category in results_df
        for member_id in member_ids:
            # Skip NaN or invalid member IDs
            if pd.isna(member_id):
                continue
                
            # Try exact match first
            member_data = results_df[results_df['Encoded ID'] == member_id]
            
            # If no match, try converting both to strings for comparison
            if member_data.empty:
                # Convert to string and try again
                member_data = results_df[results_df['Encoded ID'].astype(str) == str(member_id)]
                
                # If still no match and member_id has numeric format
                if member_data.empty and str(member_id).replace('.', '', 1).isdigit():
                    try:
                        # Try as float
                        float_id = float(member_id)
                        member_data = results_df[results_df['Encoded ID'].astype(float) == float_id]
                        
                        # Try as int if it's a whole number
                        if member_data.empty and float_id.is_integer():
                            int_id = int(float_id)
                            member_data = results_df[results_df['Encoded ID'].astype(int) == int_id]
                    except:
                        pass
            
            if not member_data.empty and 'Racial_Identity_Category' in member_data.columns:
                racial_identity = member_data['Racial_Identity_Category'].iloc[0]
                if pd.notna(racial_identity):
                    unique_racial_identities.add(racial_identity)
        
        # Store the count of unique racial identities for this circle
        if unique_racial_identities:  # Only include if there's at least one valid category
            count = len(unique_racial_identities)
            circle_racial_identity_counts[circle_id] = count
            # Calculate diversity score: 1 point if everyone is in the same category,
            # 2 points if two categories, 3 points if three categories
            circle_racial_identity_diversity_scores[circle_id] = count
            
            # CRITICAL FIX: Update the dataframe with the racial identity score for this circle
            circles_df.at[idx, 'racial_identity_score'] = count
            
            if is_debug_circle:
                print(f"RACIAL IDENTITY DEBUG - {circle_id}: Updated score to {count} (found {len(unique_racial_identities)} unique categories)")
        else:
            # Circle has no valid racial identity data - include with 0 count
            circle_racial_identity_counts[circle_id] = 0
            circle_racial_identity_diversity_scores[circle_id] = 0
            circles_df.at[idx, 'racial_identity_score'] = 0
            
            if is_debug_circle:
                print(f"RACIAL IDENTITY DEBUG - {circle_id}: No racial identity data found, set score to 0")
            
            if not has_parsing_error:
                circles_with_no_racial_data.append(circle_id)
    
    # Create histogram data from the racial identity counts
    if not circle_racial_identity_counts:
        st.warning("No racial identity data available for circles.")
        return
        
    # Count circles by number of unique racial identity categories
    diversity_counts = pd.Series(circle_racial_identity_counts).value_counts().sort_index()
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Number of Racial Identity Categories': diversity_counts.index,
        'Number of Circles': diversity_counts.values
    })
    
    st.subheader("Racial Identity Diversity Within Circles")
    
    # Create histogram using plotly with Stanford cardinal red color
    fig = px.bar(
        plot_df,
        x='Number of Racial Identity Categories',
        y='Number of Circles',
        title='Distribution of Circles by Number of Racial Identity Categories',
        text='Number of Circles',  # Display count values on bars
        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
    )
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis=dict(
            title="Number of Different Racial Identity Categories",
            tickmode='linear',
            dtick=1,  # Force integer labels
            range=[0.5, 3.5]  # Since we have 3 categories max
        ),
        yaxis_title="Number of Circles"
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True, key="racial_diversity_histogram")
    
    # Show a table with the data
    st.caption("Data table:")
    st.dataframe(plot_df, hide_index=True)
    
    # Display the Racial Identity diversity score
    st.subheader("Racial Identity Diversity Score")
    st.write("""
    For each circle, the racial identity diversity score is calculated as follows:
    - 1 point: All members in the same racial identity category
    - 2 points: Members from two different racial identity categories
    - 3 points: Members from all three racial identity categories
    """)
    
    # Calculate average and total diversity scores
    total_diversity_score = sum(circle_racial_identity_diversity_scores.values()) if circle_racial_identity_diversity_scores else 0
    avg_diversity_score = total_diversity_score / len(circle_racial_identity_diversity_scores) if circle_racial_identity_diversity_scores else 0
    
    # Store the total score in session state for use in the Match tab
    st.session_state.racial_identity_diversity_score = total_diversity_score
    
    # Create two columns for the metrics
    col1, col2 = st.columns(2)
    
    # Show average and total scores
    with col1:
        st.metric("Average Racial Identity Diversity Score", f"{avg_diversity_score:.2f}")
    with col2:
        st.metric("Total Racial Identity Diversity Score", f"{total_diversity_score}")
    
    # Add a brief explanation
    total_circles = sum(diversity_counts.values)
    diverse_circles = sum(diversity_counts[diversity_counts.index > 1].values)
    diverse_pct = (diverse_circles / total_circles * 100) if total_circles > 0 else 0
    
    st.write(f"Out of {total_circles} total circles, {diverse_circles} ({diverse_pct:.1f}%) contain members from multiple racial identity categories.")
    
    # CRITICAL FIX: Update the session state with our modified circles_df that now has diversity scores
    st.session_state.matched_circles = circles_df
    print(f"RACIAL IDENTITY HISTOGRAM UPDATE - Updated session state matched_circles with calculated racial identity scores for {len(circles_df)} circles")
    
def render_circles_detail():
    """Render the Circles Detail tab with comprehensive diversity metrics for each circle"""
    st.subheader("Circles Detail")
    
    # Explanation text
    st.write("""
    Diversity scores represent the number of categories represented in a circle. 
    For example:
    - If a circle has members from two different vintage categories, its Vintage Score is 2
    - If a circle has members from three different industry categories, its Industry Score is 3
    
    Total Diversity Score is the simple sum of all five individual diversity scores:
    Vintage + Employment + Industry + Racial Identity + Children
    """)
    
    # First check if we have the necessary data
    if 'matched_circles' not in st.session_state or st.session_state.matched_circles is None:
        st.warning("No matched circles data available. Please run the matching algorithm first.")
        return
        
    # Check if the matched_circles dataframe is empty
    if hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty:
        st.warning("No matched circles data available (empty dataframe).")
        return
        
    if 'results' not in st.session_state or st.session_state.results is None:
        st.warning("No results data available to analyze circle diversity.")
        return
    
    # Debug - Print session state keys
    print("DEBUG SESSION STATE KEYS:")
    for key in st.session_state:
        print(f"  - {key}")
    
    # Debug - Look for diversity scores in session state
    if 'diversity_scores' in st.session_state:
        print("DEBUG - Found diversity_scores in session state")
        for key, value in st.session_state.diversity_scores.items():
            print(f"  {key}: {value}")
    
    # Get the circle data and results data 
    circles_df = st.session_state.matched_circles.copy()
    results_df = st.session_state.results.copy()
    
    # Debug - Check what's in circles_df
    print(f"DEBUG CIRCLES_DF - Shape: {circles_df.shape}")
    print(f"DEBUG CIRCLES_DF - Columns: {circles_df.columns.tolist()}")
    print(f"DEBUG CIRCLES_DF - First few circle IDs: {circles_df['circle_id'].head(5).tolist()}")
    
    # Debug - Special check for IP-ATL-1 and IP-BOS-01
    for test_id in ['IP-ATL-1', 'IP-BOS-01']:
        test_circle = circles_df[circles_df['circle_id'] == test_id]
        if not test_circle.empty:
            print(f"DEBUG - Found test circle {test_id}")
            print(f"  Member count: {test_circle['member_count'].iloc[0]}")
            print(f"  Members: {test_circle['members'].iloc[0]}")
    
    # Filter out circles with no members
    if 'member_count' not in circles_df.columns:
        st.warning("Circles data does not have member_count column")
        return
    
    circles_df = circles_df[circles_df['member_count'] > 0]
    
    if len(circles_df) == 0:
        st.warning("No circles with members available for analysis.")
        return
        
    # Region filter dropdown
    # Debug circle columns to determine correct region column
    print(f"DEBUG - Circles DataFrame columns: {circles_df.columns.tolist()}")
    
    # First check if our reconstructed circles use uppercase 'Region'
    region_col = None 
    for possible_col in ['region', 'Region', 'Current_Region', 'proposed_NEW_Region']:
        if possible_col in circles_df.columns:
            region_col = possible_col
            print(f"DEBUG - Found region column: {region_col}")
            break
            
    if not region_col:
        # Fallback if no region column is found
        print("DEBUG - No region column found in circles_df, using placeholder")
        available_regions = ["All Regions"]
    else:
        # Use the found region column
        available_regions = sorted(circles_df[region_col].dropna().unique().tolist())
        available_regions = ["All Regions"] + available_regions
    
    selected_region = st.selectbox("Filter by Region", options=available_regions, index=0, key="circles_detail_region_filter")
    
    # Apply region filter if not "All Regions" and we found a region column
    if selected_region != "All Regions" and region_col:
        circles_df = circles_df[circles_df[region_col] == selected_region]
        
    if len(circles_df) == 0:
        st.warning(f"No circles found in the {selected_region} region.")
        return
        
    # Create a dataframe to store circle diversity metrics
    diversity_data = []
    
    # Process each circle to calculate diversity metrics
    for _, circle_row in circles_df.iterrows():
        circle_id = circle_row['circle_id']
        
        # Use the previously determined region column or fallback
        if region_col:
            region = circle_row.get(region_col, 'Unknown')
        else:
            region = 'Unknown'
            
        # Try different versions of subregion column name
        subregion_col = None
        for col in ['subregion', 'Subregion', 'Current_Subregion', 'proposed_NEW_Subregion']:
            if col in circle_row and pd.notna(circle_row[col]):
                subregion_col = col
                break
                
        if subregion_col:
            subregion = circle_row[subregion_col]
        else:
            subregion = 'Unknown'
        
        # Get member count
        member_count = circle_row.get('member_count', 0)
        
        # IMPROVED APPROACH: Get the list of members for this circle
        # Method 1: Try to get members from the circle_row['members']
        members_from_row = []
        if 'members' in circle_row and circle_row['members'] and not pd.isna(circle_row['members']).all():
            # For list representation
            if isinstance(circle_row['members'], list):
                members_from_row = [m for m in circle_row['members'] if not pd.isna(m)]
            # For string representation - convert to list
            elif isinstance(circle_row['members'], str):
                try:
                    if circle_row['members'].startswith('['):
                        members_from_row = [m for m in eval(circle_row['members']) if not pd.isna(m)]
                    else:
                        members_from_row = [circle_row['members']]
                except Exception:
                    pass
        
        # Method 2: Get members by looking up the circle_id in the results dataframe's proposed_NEW_circles_id column
        members_from_lookup = []
        if 'proposed_NEW_circles_id' in results_df.columns:
            # Find all participants assigned to this circle
            circle_members = results_df[results_df['proposed_NEW_circles_id'] == circle_id]
            if not circle_members.empty and 'Encoded ID' in circle_members.columns:
                members_from_lookup = circle_members['Encoded ID'].dropna().tolist()
        
        # Combine both methods, prioritizing non-empty results
        members = members_from_lookup if members_from_lookup else members_from_row
        
        # Initialize sets to track unique categories for each diversity type
        unique_vintages = set()
        unique_employment_categories = set()
        unique_industry_categories = set()
        unique_ri_categories = set()
        unique_children_categories = set()
        
        # Track category counts for determining top categories
        vintage_counts = {}
        employment_counts = {}
        industry_counts = {}
        ri_counts = {}
        children_counts = {}
        
        # For each member, look up their demographic data
        for member_id in members:
            # Skip NaN or invalid member IDs
            if pd.isna(member_id):
                continue
                
            # Try exact match first
            member_data = results_df[results_df['Encoded ID'] == member_id]
            
            # If no match, try converting both to strings for comparison
            if member_data.empty:
                # Convert to string and try again
                member_data = results_df[results_df['Encoded ID'].astype(str) == str(member_id)]
                
                # If still no match and member_id has numeric format but might be int vs float
                if member_data.empty and str(member_id).replace('.', '', 1).isdigit():
                    try:
                        # Try as float
                        float_id = float(member_id)
                        member_data = results_df[results_df['Encoded ID'].astype(float) == float_id]
                        
                        # Try as int if it's a whole number
                        if member_data.empty and float_id.is_integer():
                            int_id = int(float_id)
                            member_data = results_df[results_df['Encoded ID'].astype(int) == int_id]
                    except:
                        pass
            
            if not member_data.empty:
                # Vintage diversity
                if 'Class_Vintage' in member_data.columns:
                    vintage = member_data['Class_Vintage'].iloc[0]
                    if pd.notna(vintage):
                        unique_vintages.add(vintage)
                        vintage_counts[vintage] = vintage_counts.get(vintage, 0) + 1
                
                # Employment diversity
                if 'Employment_Category' in member_data.columns:
                    employment = member_data['Employment_Category'].iloc[0]
                    if pd.notna(employment):
                        unique_employment_categories.add(employment)
                        employment_counts[employment] = employment_counts.get(employment, 0) + 1
                
                # Industry diversity
                if 'Industry_Category' in member_data.columns:
                    industry = member_data['Industry_Category'].iloc[0]
                    if pd.notna(industry):
                        unique_industry_categories.add(industry)
                        industry_counts[industry] = industry_counts.get(industry, 0) + 1

                # Racial Identity diversity
                if 'Racial_Identity_Category' in member_data.columns:
                    ri = member_data['Racial_Identity_Category'].iloc[0]
                    if pd.notna(ri):
                        unique_ri_categories.add(ri)
                        ri_counts[ri] = ri_counts.get(ri, 0) + 1
                
                # Children diversity
                if 'Children_Category' in member_data.columns:
                    children = member_data['Children_Category'].iloc[0]
                    if pd.notna(children):
                        unique_children_categories.add(children)
                        children_counts[children] = children_counts.get(children, 0) + 1
        
        # Calculate diversity scores
        vintage_score = len(unique_vintages) if unique_vintages else 0
        employment_score = len(unique_employment_categories) if unique_employment_categories else 0
        industry_score = len(unique_industry_categories) if unique_industry_categories else 0
        ri_score = len(unique_ri_categories) if unique_ri_categories else 0
        children_score = len(unique_children_categories) if unique_children_categories else 0
        
        # Calculate total diversity score
        total_score = vintage_score + employment_score + industry_score + ri_score + children_score
        
        # Determine top categories (with percentages)
        vintage_top = "None"
        employment_top = "None"
        industry_top = "None"
        ri_top = "None"
        children_top = "None"
        
        if vintage_counts:
            top_vintage = max(vintage_counts.items(), key=lambda x: x[1])
            vintage_pct = (top_vintage[1] / member_count) * 100 if member_count > 0 else 0
            vintage_top = f"{top_vintage[0]} ({vintage_pct:.1f}%)"
        
        if employment_counts:
            top_employment = max(employment_counts.items(), key=lambda x: x[1])
            employment_pct = (top_employment[1] / member_count) * 100 if member_count > 0 else 0
            employment_top = f"{top_employment[0]} ({employment_pct:.1f}%)"
        
        if industry_counts:
            top_industry = max(industry_counts.items(), key=lambda x: x[1])
            industry_pct = (top_industry[1] / member_count) * 100 if member_count > 0 else 0
            industry_top = f"{top_industry[0]} ({industry_pct:.1f}%)"
        
        if ri_counts:
            top_ri = max(ri_counts.items(), key=lambda x: x[1])
            ri_pct = (top_ri[1] / member_count) * 100 if member_count > 0 else 0
            ri_top = f"{top_ri[0]} ({ri_pct:.1f}%)"
        
        if children_counts:
            top_children = max(children_counts.items(), key=lambda x: x[1])
            children_pct = (top_children[1] / member_count) * 100 if member_count > 0 else 0
            children_top = f"{top_children[0]} ({children_pct:.1f}%)"
        
        # Add to diversity data
        diversity_data.append({
            'Circle ID': circle_id,
            'Region': region,
            'Subregion': subregion,
            'Participants': member_count,
            'Total Diversity Score': total_score,
            'Vintage Score': vintage_score,
            'Vintage Top Category': vintage_top,
            'Employment Score': employment_score,
            'Employment Top Category': employment_top,
            'Industry Score': industry_score,
            'Industry Top Category': industry_top,
            'RI Score': ri_score,
            'RI Top Category': ri_top,
            'Children Score': children_score,
            'Children Top Category': children_top
        })
        
        # CRITICAL FIX: Update the original circles_df with diversity scores
        # Find the row index in the original dataframe
        idx = circles_df.index[circles_df['circle_id'] == circle_id].tolist()
        if idx:
            # Add or update the scores directly in the dataframe
            circles_df.loc[idx[0], 'vintage_score'] = vintage_score
            circles_df.loc[idx[0], 'employment_score'] = employment_score
            circles_df.loc[idx[0], 'industry_score'] = industry_score
            circles_df.loc[idx[0], 'ri_score'] = ri_score
            circles_df.loc[idx[0], 'children_score'] = children_score
            circles_df.loc[idx[0], 'total_diversity_score'] = total_score
            
            # Log successful update
            if circle_id in ['IP-ATL-1', 'IP-BOS-01']:
                print(f"DEBUG - Updated {circle_id} in circles_df with diversity scores: V({vintage_score}), E({employment_score}), I({industry_score}), RI({ri_score}), C({children_score}), Total({total_score})")
    
    # Create DataFrame from diversity data
    if diversity_data:
        diversity_df = pd.DataFrame(diversity_data)
        
        # Show the table with sorting enabled
        st.dataframe(
            diversity_df,
            hide_index=True,
            use_container_width=True
        )
        
        # Summary statistics
        st.subheader("Summary Statistics")
        
        # First row of metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Circles", len(diversity_df))
        
        with col2:
            avg_total_score = diversity_df['Total Diversity Score'].mean()
            st.metric("Avg Total Score", f"{avg_total_score:.2f}")
        
        with col3:
            avg_vintage_score = diversity_df['Vintage Score'].mean()
            st.metric("Avg Vintage Score", f"{avg_vintage_score:.2f}")
        
        with col4:
            avg_employment_score = diversity_df['Employment Score'].mean()
            st.metric("Avg Employment Score", f"{avg_employment_score:.2f}")
        
        with col5:
            avg_industry_score = diversity_df['Industry Score'].mean()
            st.metric("Avg Industry Score", f"{avg_industry_score:.2f}")
            
        # Second row of metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            avg_ri_score = diversity_df['RI Score'].mean()
            st.metric("Avg RI Score", f"{avg_ri_score:.2f}")
            
        with col2:
            avg_children_score = diversity_df['Children Score'].mean()
            st.metric("Avg Children Score", f"{avg_children_score:.2f}")
            
        with col5:
            avg_participants = diversity_df['Participants'].mean()
            st.metric("Avg Participants", f"{avg_participants:.1f}")
    else:
        st.warning("No diversity data could be calculated for the selected circles.")
    
    # CRITICAL FIX: Update the session state with our modified circles_df that now has diversity scores
    st.session_state.matched_circles = circles_df
    print(f"DIVERSITY UPDATE - Updated session state matched_circles with diversity scores for {len(circles_df)} circles")

def render_racial_identity_analysis(data):
    """Render the Racial Identity analysis visualizations"""
    st.subheader("Racial Identity Analysis")
    
    # Check if data is None
    if data is None:
        st.warning("No data available for analysis. Please upload participant data.")
        return
    
    # Create a copy to work with
    df = data.copy()
    
    # Debug: show column names
    st.caption("Debugging information:")
    with st.expander("Show data columns and sample values"):
        st.write("Available columns:")
        for col in df.columns:
            st.text(f"- {col}")
        
        # Try to find Racial Identity column
        racial_identity_col = None
        for col in df.columns:
            if "racial identity" in col.lower():
                racial_identity_col = col
                break
        
        if racial_identity_col:
            st.write(f"Found Racial Identity column: {racial_identity_col}")
            # Show some sample values
            sample_values = df[racial_identity_col].dropna().head(5).tolist()
            st.write(f"Sample values: {sample_values}")
        else:
            st.write("No Racial Identity column found in the data")
    
    # Check if we need to create Racial Identity Category
    if 'Racial_Identity_Category' not in df.columns:
        if racial_identity_col:
            st.info(f"Creating Racial Identity Category from {racial_identity_col}...")
            
            # Define function to categorize racial identity
            def categorize_racial_identity(identity):
                if pd.isna(identity):
                    return None
                
                # Convert to string in case it's not
                identity_str = str(identity)
                
                # Apply categorization rules
                if identity_str.startswith("White"):
                    return "White"
                elif "Asian" in identity_str:
                    return "Asian"
                else:
                    return "All Else"
            
            # Apply the categorization function
            df['Racial_Identity_Category'] = df[racial_identity_col].apply(categorize_racial_identity)
            
            # Update session state with the new Racial_Identity_Category
            if 'results' in st.session_state and st.session_state.results is not None:
                # Copy the newly created Racial_Identity_Category to the results DataFrame
                # First, create a dictionary mapping Encoded ID to Racial_Identity_Category
                race_cat_mapping = dict(zip(df['Encoded ID'], df['Racial_Identity_Category']))
                
                # Then apply this mapping to the results DataFrame
                if 'Encoded ID' in st.session_state.results.columns:
                    st.session_state.results['Racial_Identity_Category'] = st.session_state.results['Encoded ID'].map(race_cat_mapping)
                    st.info("Updated results data with Racial Identity Categories")
        else:
            st.warning("Racial Identity data is not available. Please ensure Racial Identity data was included in the uploaded file.")
            return
    
    # Filter out rows with missing Racial Identity Category
    df = df[df['Racial_Identity_Category'].notna()]
    
    if len(df) == 0:
        st.warning("No Racial Identity Category data is available after filtering.")
        return
    
    # Define the proper order for Racial Identity Categories
    racial_identity_order = ["White", "Asian", "All Else"]
    
    # FIRST: Display diversity within circles IF we have matched circles
    if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
        if not (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty):
            render_racial_identity_diversity_histogram()
        else:
            st.info("Run the matching algorithm to see the Racial Identity diversity within circles.")
    else:
        st.info("Run the matching algorithm to see the Racial Identity diversity within circles.")
    
    # SECOND: Display Distribution of Racial Identity
    st.subheader("Distribution of Racial Identity")
    
    # Count by Racial Identity Category
    racial_identity_counts = df['Racial_Identity_Category'].value_counts().reindex(racial_identity_order).fillna(0).astype(int)
    
    # Create a DataFrame for plotting
    racial_identity_df = pd.DataFrame({
        'Racial Identity Category': racial_identity_counts.index,
        'Count': racial_identity_counts.values
    })
    
    # Create histogram using plotly with Stanford cardinal red color
    fig = px.bar(
        racial_identity_df,
        x='Racial Identity Category',
        y='Count',
        title='Distribution of Racial Identity Categories',
        text='Count',  # Display count values on bars
        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
    )
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': racial_identity_order},
        xaxis_title="Racial Identity Category",
        yaxis_title="Count of Participants"
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True, key="racial_identity_distribution")
    
    # Create a breakdown by Status if Status column exists
    if 'Status' in df.columns:
        st.subheader("Racial Identity by Status")
        
        # Create a crosstab of Racial Identity Category vs Status
        status_racial_identity = pd.crosstab(
            df['Racial_Identity_Category'], 
            df['Status'],
            rownames=['Racial Identity Category'],
            colnames=['Status']
        ).reindex(racial_identity_order)
        
        # Add a Total column
        status_racial_identity['Total'] = status_racial_identity.sum(axis=1)
        
        # Calculate percentages
        for col in status_racial_identity.columns:
            if col != 'Total':
                status_racial_identity[f'{col} %'] = (status_racial_identity[col] / status_racial_identity['Total'] * 100).round(1)
        
        # Reorder columns to group counts with percentages
        cols = []
        for status in status_racial_identity.columns:
            if status != 'Total' and not status.endswith(' %'):
                cols.append(status)
                cols.append(f'{status} %')
        cols.append('Total')
        
        # Show the table
        st.dataframe(status_racial_identity[cols], use_container_width=True)
        
        # Create a stacked bar chart
        fig = px.bar(
            status_racial_identity.reset_index(),
            x='Racial Identity Category',
            y=[col for col in status_racial_identity.columns if col != 'Total' and not col.endswith(' %')],
            title='Racial Identity Distribution by Status',
            barmode='stack',
            color_discrete_sequence=['#8C1515', '#2E2D29']  # Stanford colors
        )
        
        # Customize layout
        fig.update_layout(
            xaxis={'categoryorder': 'array', 'categoryarray': racial_identity_order},
            xaxis_title="Racial Identity Category",
            yaxis_title="Count of Participants",
            legend_title="Status"
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True, key="racial_identity_status_distribution")