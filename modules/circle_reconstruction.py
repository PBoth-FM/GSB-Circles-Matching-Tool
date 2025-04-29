"""
Circle reconstruction utilities to ensure all assigned circles, including post-processed ones,
appear correctly in UI components.
"""

import pandas as pd
import numpy as np

def safe_isna(val):
    """Safely check if a value is NA, handling both scalar and array-like objects."""
    if isinstance(val, (pd.Series, pd.DataFrame)):
        # For pandas objects, check if all values are NA
        return val.isna().all()
    elif isinstance(val, (np.ndarray, list)):
        # For numpy arrays or lists
        return all(pd.isna(x) for x in val)
    else:
        # For scalar values
        return pd.isna(val)

def reconstruct_circles_from_results(results, original_circles=None):
    """
    Reconstruct circles dataframe from individual participant results.
    This is crucial after post-processing to ensure that all circles, 
    including those with post-processed participants, are properly represented.
    
    Args:
        results: List of participant results with assignments
        original_circles: Original circles dataframe (optional)
        
    Returns:
        DataFrame: Updated circles dataframe with all assigned circles
    """
    print("\n🔄 RECONSTRUCTING CIRCLES FROM PARTICIPANT RESULTS")
    
    # Convert results to DataFrame if it's a list
    if isinstance(results, list):
        # Check first if it's non-empty
        if not results:
            print("  ⚠️ Results list is empty!")
            return pd.DataFrame()
            
        # Handle differences in column naming between result entries
        # Some results may have 'participant_id', others may have 'Encoded ID'
        id_column = 'participant_id'
        if 'Encoded ID' in results[0]:
            id_column = 'Encoded ID'
            
        # Create a dataframe from the results list
        results_df = pd.DataFrame(results)
        print(f"  Created DataFrame with {len(results_df)} participants")
    else:
        # Already a DataFrame
        results_df = results.copy()
        print(f"  Using existing DataFrame with {len(results_df)} participants")
        # Try to determine ID column
        id_column = 'participant_id' if 'participant_id' in results_df.columns else 'Encoded ID'
    
    # Check for the circle assignment column
    circle_column = None
    for col in ['proposed_NEW_circles_id', 'circle_id', 'assigned_circle']:
        if col in results_df.columns:
            circle_column = col
            break
            
    if not circle_column:
        print("  ⚠️ Could not find circle assignment column!")
        return pd.DataFrame()
        
    print(f"  Using ID column: {id_column}, Circle column: {circle_column}")
    
    # Create a mapping of circle IDs to lists of member IDs
    circle_members = {}
    circle_metadata = {}
    
    # Extract all participants assigned to circles (not UNMATCHED)
    matched_df = results_df[results_df[circle_column] != 'UNMATCHED']
    print(f"  Found {len(matched_df)} matched participants")
    
    # Get unique circle IDs
    unique_circles = matched_df[circle_column].unique()
    print(f"  Found {len(unique_circles)} unique circles")
    
    # Extract circle information from the original circles dataframe if provided
    original_circle_info = {}
    if original_circles is not None and isinstance(original_circles, pd.DataFrame):
        for _, row in original_circles.iterrows():
            if 'circle_id' in row:
                c_id = row['circle_id']
                original_circle_info[c_id] = row.to_dict()
    
    # Group participants by circle
    for circle_id in unique_circles:
        # Skip invalid circle IDs
        if safe_isna(circle_id) or circle_id == 'UNMATCHED':
            continue
            
        # Get participants in this circle
        members_df = matched_df[matched_df[circle_column] == circle_id]
        member_ids = members_df[id_column].tolist()
        
        # Store members list
        circle_members[circle_id] = member_ids
        
        # Initialize circle metadata
        circle_metadata[circle_id] = {
            'circle_id': circle_id,
            'member_count': len(member_ids),
            'members': member_ids
        }
        
        # Try to extract region, subregion, meeting time from results
        sample_member = members_df.iloc[0]
        
        # Extract circle properties from results
        for prop, column_options in [
            ('region', ['proposed_NEW_Region', 'Current_Region', 'region']),
            ('subregion', ['proposed_NEW_Subregion', 'Current_Subregion', 'subregion']),
            ('meeting_time', ['proposed_NEW_DayTime', 'Current_Meeting_Time', 'meeting_time'])
        ]:
            # Check each possible column name
            for col in column_options:
                # Safe check for column existence and non-NA values
                if col in sample_member:
                    # Use safe_isna to handle potential array-like values
                    if not safe_isna(sample_member[col]):
                        circle_metadata[circle_id][prop] = sample_member[col]
                        break
                    
        # Check if the circle was in the original circles dataframe
        if circle_id in original_circle_info:
            # Copy properties not already set
            for prop, value in original_circle_info[circle_id].items():
                # Safe handling for DataFrame/Series truth value ambiguity
                prop_exists = prop in circle_metadata[circle_id]
                if not prop_exists:
                    circle_metadata[circle_id][prop] = value
                else:
                    # Use our safe_isna helper function to handle all types
                    val = circle_metadata[circle_id][prop]
                    if safe_isna(val):  # This handles both scalar and array-like values safely
                        circle_metadata[circle_id][prop] = value
                    
        # Count hosts if host column exists
        if 'host' in results_df.columns:
            always_hosts = len(members_df[members_df['host'] == 'Always'])
            sometimes_hosts = len(members_df[members_df['host'] == 'Sometimes'])
            circle_metadata[circle_id]['always_hosts'] = always_hosts
            circle_metadata[circle_id]['sometimes_hosts'] = sometimes_hosts
                    
        # Add is_existing flag and new_members count
        if 'Status' in results_df.columns:
            new_members = sum(1 for _, row in members_df.iterrows() if row.get('Status') == 'NEW')
            continuing_members = sum(1 for _, row in members_df.iterrows() if row.get('Status') == 'CURRENT-CONTINUING')
            
            circle_metadata[circle_id]['new_members'] = new_members
            circle_metadata[circle_id]['continuing_members'] = continuing_members
            circle_metadata[circle_id]['is_existing'] = continuing_members > 0
            circle_metadata[circle_id]['is_new_circle'] = continuing_members == 0
            
            # Calculate max_additions for continuing circles
            if continuing_members > 0:  # This is an existing circle
                # First, check if max_additions exists in original data
                if circle_id in original_circle_info and 'max_additions' in original_circle_info[circle_id]:
                    # Use the existing max_additions value from optimization
                    max_additions = original_circle_info[circle_id]['max_additions']
                    circle_metadata[circle_id]['max_additions'] = max_additions
                    print(f"  Preserved max_additions={max_additions} for circle {circle_id}")
                else:
                    # Calculate max_additions based on continuing circle rules
                    # 1. For continuing circles, never exceed a total of 8 members
                    # 2. For small circles (<5 members), add enough to reach 5 regardless of preferences
                    total_members = len(member_ids)
                    
                    if total_members < 5:
                        # Small circle - can add members to reach 5
                        max_additions = 5 - total_members
                        print(f"  Small circle {circle_id}: {total_members} members, calculated max_additions={max_additions}")
                    else:
                        # Regular continuing circle - never exceed 8 total
                        max_additions = max(0, 8 - total_members)
                        print(f"  Continuing circle {circle_id}: {total_members} members, calculated max_additions={max_additions}")
                    
                    circle_metadata[circle_id]['max_additions'] = max_additions
            else:
                # New circle - set max_additions to 0 since it's already formed
                circle_metadata[circle_id]['max_additions'] = 0
            
    # Convert circle metadata to DataFrame
    circles_df = pd.DataFrame(list(circle_metadata.values()))
    
    # Post-process the dataframe
    if not circles_df.empty:
        # Ensure numeric columns are integers
        for col in ['member_count', 'new_members', 'always_hosts', 'sometimes_hosts', 'max_additions']:
            if col in circles_df.columns:
                circles_df[col] = pd.to_numeric(circles_df[col], errors='coerce').fillna(0).astype(int)
                
        # Sort by circle ID
        if 'circle_id' in circles_df.columns:
            circles_df = circles_df.sort_values('circle_id')
    
    print(f"  Successfully created circles DataFrame with {len(circles_df)} circles")
    
    # Debug - show a few sample circles
    if not circles_df.empty and len(circles_df) > 0:
        print("  Sample circles:")
        for i, (_, row) in enumerate(circles_df.head(3).iterrows()):
            print(f"    {i+1}. {row['circle_id']}: {row['member_count']} members ({row.get('new_members', 0)} new, max_additions={row.get('max_additions', 0)})")
    
    return circles_df