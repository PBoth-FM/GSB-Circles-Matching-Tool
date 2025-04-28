"""
Critical fixes for the circle optimizer algorithm. 
This module contains targeted fixes for issues with CURRENT-CONTINUING participants
and improving the "optimize" mode for continuing circles.
"""

import pandas as pd
import streamlit as st

def find_current_circle_id(participant_row):
    """
    Robust method to find a current circle ID for a participant across multiple potential columns.
    
    Args:
        participant_row: Row from participant dataframe
        
    Returns:
        str: Circle ID if found, None otherwise
    """
    # Check if this isn't a CURRENT-CONTINUING participant
    if participant_row.get('Status') not in ['CURRENT-CONTINUING', 'Current-CONTINUING']:
        return None
        
    # 1. Try standard column names first
    standard_column_names = [
        'Current Circle ID', 'Current_Circle_ID', 'current_circles_id', 
        'Current Circles ID', 'Current/ Continuing Circle ID'
    ]
    
    for col_name in standard_column_names:
        if col_name in participant_row.index and not pd.isna(participant_row[col_name]) and participant_row[col_name]:
            return str(participant_row[col_name]).strip()
    
    # 2. If still not found, check all columns with "circle" and "current" in their name
    for col in participant_row.index:
        col_lower = str(col).lower()
        if ('circle' in col_lower) and ('current' in col_lower or 'id' in col_lower):
            if not pd.isna(participant_row[col]) and participant_row[col]:
                return str(participant_row[col]).strip()
    
    # 3. Try a more aggressive approach looking for any circle-related data
    for col in participant_row.index:
        col_lower = str(col).lower()
        if 'circle' in col_lower:
            if not pd.isna(participant_row[col]) and participant_row[col]:
                circle_value = str(participant_row[col]).strip()
                # Check if this looks like a circle ID (contains letters, numbers, and dashes)
                if '-' in circle_value and any(c.isalpha() for c in circle_value) and any(c.isdigit() for c in circle_value):
                    return circle_value
    
    # No circle ID found
    return None


def preprocess_continuing_members(participants_df, circle_ids):
    """
    Pre-process all CURRENT-CONTINUING members to ensure they stay in their current circles.
    
    Args:
        participants_df: DataFrame with all participants
        circle_ids: List of valid circle IDs
        
    Returns:
        dict: Mapping of participant IDs to their preassigned circles
        list: List of participants with missing circle IDs
    """
    # Filter for CURRENT-CONTINUING participants
    continuing_mask = participants_df['Status'].isin(['CURRENT-CONTINUING', 'Current-CONTINUING'])
    continuing_participants = participants_df[continuing_mask]
    
    # Pre-assign CURRENT-CONTINUING participants to their circles
    preassigned = {}
    problem_participants = []
    
    for _, row in continuing_participants.iterrows():
        p_id = row['Encoded ID']
        current_circle = find_current_circle_id(row)
        
        if current_circle:
            # Check if circle exists in valid circle IDs
            if current_circle in circle_ids:
                preassigned[p_id] = current_circle
            else:
                # Circle ID not found in valid circles
                problem_participants.append({
                    'participant_id': p_id,
                    'circle_id': current_circle,
                    'reason': 'Circle ID not in valid circles'
                })
        else:
            # No circle ID found
            problem_participants.append({
                'participant_id': p_id,
                'circle_id': None,
                'reason': 'No circle ID found'
            })
    
    return preassigned, problem_participants


def optimize_circle_capacity(viable_circles, existing_circle_handling, min_circle_size=5):
    """
    Optimize circle capacity for circles based on the circle handling mode.
    
    Args:
        viable_circles: Dict of circle IDs to circle info
        existing_circle_handling: Mode (preserve, optimize, dissolve)
        min_circle_size: Minimum viable circle size
        
    Returns:
        dict: Updated viable circles with capacity adjustments
    """
    updated_circles = {}
    
    for circle_id, circle_info in viable_circles.items():
        current_members = circle_info.get('member_count', 0)
        max_additions = circle_info.get('max_additions', 0)
        updated_info = circle_info.copy()
        
        # For any circle with less than min_circle_size members, allow more to reach viable size
        if current_members < min_circle_size:
            needed = min_circle_size - current_members
            if max_additions < needed:
                print(f"✅ Small circle override: {circle_id} with {current_members} members had max_additions={max_additions}, now {needed}")
                updated_info['max_additions'] = needed
        
        # In optimize mode, ensure all continuing circles can accept at least one new member
        elif existing_circle_handling == 'optimize' and max_additions == 0 and current_members < 10:
            print(f"✅ Optimize mode override: {circle_id} now allows 1 new member")
            updated_info['max_additions'] = 1
        
        updated_circles[circle_id] = updated_info
    
    return updated_circles


def force_compatibility(participant_id, circle_id, compatibility_matrix):
    """
    Force a participant to be compatible with a specific circle.
    
    Args:
        participant_id: ID of the participant
        circle_id: ID of the circle to force compatibility with
        compatibility_matrix: Existing compatibility matrix to update
        
    Returns:
        dict: Updated compatibility matrix
    """
    # Create a copy to avoid modifying the original
    updated_matrix = compatibility_matrix.copy()
    
    # Set compatibility for this participant with this circle to 1 (compatible)
    updated_matrix[(participant_id, circle_id)] = 1
    
    return updated_matrix


def ensure_current_continuing_matched(results, unmatched, participants_df, circle_ids):
    """
    Final check to ensure all CURRENT-CONTINUING members are matched to their circles.
    
    Args:
        results: List of results from optimization
        unmatched: Dict of unmatched participants
        participants_df: DataFrame with all participants
        circle_ids: List of valid circle IDs
        
    Returns:
        list: Updated results list
        dict: Updated unmatched dict
    """
    # Copy inputs to avoid modifying originals
    updated_results = results.copy()
    updated_unmatched = unmatched.copy()
    
    # Get IDs of matched participants
    matched_ids = [r.get('participant_id') for r in updated_results]
    
    # Find CURRENT-CONTINUING participants who are still unmatched
    continuing_mask = participants_df['Status'].isin(['CURRENT-CONTINUING', 'Current-CONTINUING'])
    continuing_participants = participants_df[continuing_mask]
    
    for _, row in continuing_participants.iterrows():
        p_id = row['Encoded ID']
        
        # Skip if already matched
        if p_id in matched_ids:
            continue
        
        # Try to find circle ID
        current_circle = find_current_circle_id(row)
        
        if current_circle and current_circle in circle_ids:
            # Create a result for this participant
            new_result = {
                'participant_id': p_id,
                'proposed_NEW_circles_id': current_circle,
                'location_score': 3,  # Maximum score
                'time_score': 3,      # Maximum score
                'total_score': 6,     # Sum of scores
                'region': row.get('Current_Region', row.get('Derived_Region', 'Unknown')),
                'status': 'CURRENT-CONTINUING'
            }
            
            # Add to results
            updated_results.append(new_result)
            
            # Remove from unmatched if present
            if p_id in updated_unmatched:
                updated_unmatched[p_id]['unmatched_reason'] = 'FIXED: Manually assigned to continuing circle'
            
            print(f"✅ FINAL CHECK: Manually assigned CURRENT-CONTINUING participant {p_id} to {current_circle}")
    
    return updated_results, updated_unmatched