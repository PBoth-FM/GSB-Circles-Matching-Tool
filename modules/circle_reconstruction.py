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
    print("🔍 ENHANCED DIAGNOSTICS: Starting comprehensive circle reconstruction")
    
    # Debug any specific test circle IDs to watch
    test_circle_ids = ['IP-WDC-01', 'IP-WDC-02', 'IP-SFO-25', 'IP-SFO-26']
    for circle_id in test_circle_ids:
        print(f"  🔎 TRACKING circle {circle_id} through reconstruction")
    
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
        
        # ENHANCED DIAGNOSTICS: Print column names for debugging
        print(f"  Results DataFrame columns: {results_df.columns.tolist()}")
        
        # Look for meeting time related columns
        meeting_columns = [col for col in results_df.columns if 'meeting' in col.lower() or 'time' in col.lower() or 'day' in col.lower()]
        print(f"  Meeting-related columns found: {meeting_columns}")
    else:
        # Already a DataFrame
        results_df = results.copy()
        print(f"  Using existing DataFrame with {len(results_df)} participants")
        # Try to determine ID column
        id_column = 'participant_id' if 'participant_id' in results_df.columns else 'Encoded ID'
        
        # ENHANCED DIAGNOSTICS: Print column names for debugging
        print(f"  Results DataFrame columns: {results_df.columns.tolist()}")
    
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
        print(f"  Loaded information for {len(original_circle_info)} circles from original data")
        
        # Debug sample of original circle data
        if len(original_circle_info) > 0:
            sample_circle_id = list(original_circle_info.keys())[0]
            sample_data = original_circle_info[sample_circle_id]
            print(f"  Sample original circle {sample_circle_id}:")
            if 'member_count' in sample_data:
                print(f"    member_count: {sample_data['member_count']}")
            if 'members' in sample_data and isinstance(sample_data['members'], list):
                print(f"    members list length: {len(sample_data['members'])}")
    
    # Group participants by circle
    for circle_id in unique_circles:
        # Skip invalid circle IDs
        if safe_isna(circle_id) or circle_id == 'UNMATCHED':
            continue
            
        # Get participants in this circle
        members_df = matched_df[matched_df[circle_column] == circle_id]
        
        # CRITICAL FIX: Remove duplicate member IDs by using a set to ensure uniqueness
        member_ids_set = set(members_df[id_column].tolist())
        member_ids = list(member_ids_set)
        
        print(f"  Circle {circle_id}: Found {len(members_df)} member records, {len(member_ids)} unique members")
        if len(members_df) > len(member_ids):
            print(f"  ⚠️ WARNING: Circle {circle_id} had {len(members_df) - len(member_ids)} duplicate member entries - removed")
        
        # Store members list
        circle_members[circle_id] = member_ids
        
        # Initialize circle metadata, prioritizing data from original_circles if available
        member_count = len(member_ids)
        has_original_data = circle_id in original_circle_info
        
        # CRITICAL FIX: For continuing circles, prioritize member counts from original data
        # This fixes the issue with continuing circles showing member_count = 1
        if has_original_data and 'member_count' in original_circle_info[circle_id]:
            original_count = original_circle_info[circle_id]['member_count']
            if not pd.isna(original_count) and original_count > member_count:
                print(f"  ✓ Using original member_count={original_count} for circle {circle_id} (instead of {member_count})")
                member_count = original_count
            
        # Initialize circle metadata 
        circle_metadata[circle_id] = {
            'circle_id': circle_id,
            'member_count': member_count,
            'members': member_ids
        }
        
        # Try to extract region, subregion, meeting time from results
        sample_member = members_df.iloc[0]
        
        # CRITICAL FIX: For new circles (starting with "IP-NEW"), use Derived_Region or Current_Region
        # This fixes the "None" region issue in Circle Composition
        is_new_circle = circle_id.startswith('IP-NEW')
        
        # Extract circle properties from results
        for prop, column_options in [
            # Use different region column priorities for new circles vs continuing circles
            ('region', ['Derived_Region', 'Current_Region', 'proposed_NEW_Region', 'region'] if is_new_circle else 
                      ['proposed_NEW_Region', 'Current_Region', 'Derived_Region', 'region']),
            ('subregion', ['proposed_NEW_Subregion', 'Current_Subregion', 'subregion']),
            # CRITICAL FIX: Expand meeting time column options to catch all possible sources
            ('meeting_time', ['proposed_NEW_DayTime', 'Current_Meeting_Time', 'Current_Meeting_Day', 
                             'Current Meeting Day', 'Current/ Continuing Meeting Day',
                             'Current_Meeting_Time', 'Current Meeting Time', 'Current/ Continuing Meeting Time',
                             'Current/ Continuing DayTime', 'meeting_time', 'Current_DayTime', 
                             'Current DayTime', 'Meeting Time', 'Meeting Day',
                             'Preferred Meeting Day', 'Preferred Meeting Time'])
        ]:
            # Check each possible column name
            for col in column_options:
                # Safe check for column existence and non-NA values
                if col in sample_member:
                    # Use safe_isna to handle potential array-like values
                    if not safe_isna(sample_member[col]):
                        # For meeting time, try to standardize format
                        if prop == 'meeting_time':
                            # If this is just a day without time, append standard format
                            if any(day in str(sample_member[col]).lower() for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
                                if '(' not in str(sample_member[col]):
                                    # Try to extract the time from other columns
                                    time_value = None
                                    for time_col in ['Current_Meeting_Time', 'Current Meeting Time', 'Current/ Continuing Meeting Time']:
                                        if time_col in sample_member and not safe_isna(sample_member[time_col]):
                                            time_value = sample_member[time_col]
                                            break
                                    
                                    if time_value:
                                        # Format as "Day (Time)"
                                        formatted_value = f"{sample_member[col]} ({time_value})"
                                        circle_metadata[circle_id][prop] = formatted_value
                                        print(f"  ✅ Set combined {prop}='{formatted_value}' for circle {circle_id} from columns {col} and time")
                                        break
                        
                        circle_metadata[circle_id][prop] = sample_member[col]
                        print(f"  ✅ Set {prop}='{sample_member[col]}' for circle {circle_id} from column {col}")
                        break
                        
            # If meeting_time is still not set, try additional strategies
            if prop == 'meeting_time' and ('meeting_time' not in circle_metadata[circle_id] or safe_isna(circle_metadata[circle_id].get('meeting_time'))):
                # Look for day and time in separate columns and combine them
                day_cols = [
                    'Current_Meeting_Day', 'Current Meeting Day', 'Current/ Continuing Meeting Day',
                    'Meeting Day', 'Preferred Meeting Day', 'Preferred Day', 'Day',
                    'Current Day', 'Current/ Continuing Day'
                ]
                time_cols = [
                    'Current_Meeting_Time', 'Current Meeting Time', 'Current/ Continuing Meeting Time',
                    'Meeting Time', 'Preferred Meeting Time', 'Preferred Time', 'Time',
                    'Current Time', 'Current/ Continuing Time'
                ]
                
                day_value = None
                time_value = None
                
                # Try to find day value
                for day_col in day_cols:
                    if day_col in sample_member and not safe_isna(sample_member[day_col]):
                        day_value = sample_member[day_col]
                        print(f"  Found day value '{day_value}' from column '{day_col}'")
                        break
                
                # Try to find time value
                for time_col in time_cols:
                    if time_col in sample_member and not safe_isna(sample_member[time_col]):
                        time_value = sample_member[time_col]
                        print(f"  Found time value '{time_value}' from column '{time_col}'")
                        break
                
                # If we found both, combine them
                if day_value and time_value:
                    combined_value = f"{day_value} ({time_value})"
                    circle_metadata[circle_id][prop] = combined_value
                    print(f"  ✅ COMBINED {prop}='{combined_value}' for circle {circle_id} from separate day and time columns")
                    
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
                # Count unique continuing members and new members for accurate sizing
                unique_continuing = sum(1 for p_id in member_ids_set if any(
                    members_df[members_df[id_column] == p_id]['Status'] == 'CURRENT-CONTINUING'))
                unique_new = sum(1 for p_id in member_ids_set if any(
                    members_df[members_df[id_column] == p_id]['Status'] == 'NEW'))
                
                print(f"  Circle {circle_id}: {unique_continuing} unique continuing members, {unique_new} unique new members")
                
                # CRITICAL FIX: Update member_count for continuing circles
                # This fixes the issue where continuing circles show 1 member in Circle Composition
                total_members = len(member_ids)
                
                # Set member_count to the actual total of unique members
                circle_metadata[circle_id]['member_count'] = total_members
                print(f"  ✅ FIXED: Set member_count to match total unique members ({total_members}) for circle {circle_id}")
                
                # CRITICAL CHECK: Enforce 8-member limit for continuing circles
                if total_members > 8:
                    print(f"  ⚠️ WARNING: Circle {circle_id} exceeds maximum size with {total_members} members!")
                    print(f"    This includes {unique_continuing} continuing members and {unique_new} new members")
                    if unique_new > 0:
                        print(f"    This circle should not have accepted new members as it already has {unique_continuing} continuing members")
                
                # First, check if max_additions exists in original data
                if circle_id in original_circle_info and 'max_additions' in original_circle_info[circle_id]:
                    # Use the existing max_additions value from optimization
                    max_additions = original_circle_info[circle_id]['max_additions']
                    
                    # CRITICAL FIX: Enforce the 8-member limit
                    # Even if optimizer allowed more, we need to correct it here
                    if total_members >= 8:
                        # Already at or over capacity, force max_additions to 0
                        print(f"  ⚠️ FIXING: Circle {circle_id} is at/over capacity. Setting max_additions to 0 (was {max_additions})")
                        max_additions = 0
                    elif total_members + max_additions > 8:
                        # Would exceed capacity, adjust max_additions
                        corrected_max = 8 - total_members
                        print(f"  ⚠️ FIXING: Circle {circle_id} would exceed capacity. Adjusting max_additions from {max_additions} to {corrected_max}")
                        max_additions = corrected_max
                    
                    circle_metadata[circle_id]['max_additions'] = max_additions
                    print(f"  Preserved max_additions={max_additions} for circle {circle_id}")
                else:
                    # Calculate max_additions based on continuing circle rules
                    # 1. For continuing circles, never exceed a total of 8 members
                    # 2. For small circles (<5 members), add enough to reach 5 regardless of preferences
                    
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
                # New circle - different max size (10)
                total_members = len(member_ids)
                # CRITICAL FIX: For new circles, set consistent max additions
                # Always set max_additions to 10 for new circles as requested
                max_additions = 10
                
                # CRITICAL FIX: For new circles, member_count should match new_members
                # These circles often show 1 (likely due to member counting issues)
                new_members_count = circle_metadata[circle_id].get('new_members', total_members)
                if new_members_count > 0:
                    circle_metadata[circle_id]['member_count'] = new_members_count
                    print(f"  ✅ FIXED: Set member_count to match new_members ({new_members_count}) for circle {circle_id}")
                
                circle_metadata[circle_id]['max_additions'] = max_additions
                print(f"  New circle {circle_id}: {new_members_count} members, max_additions set to {max_additions}")
            
    # Convert circle metadata to DataFrame
    circles_df = pd.DataFrame(list(circle_metadata.values()))
    
    # If we have results, do final verification and fix of member counts based on actual members list
    print("\n🔍 CRITICAL VERIFICATION: Double-checking member counts against actual members list")
    
    # Post-process the dataframe
    if not circles_df.empty:
        # CRITICAL FIX: Verify each circle's member count matches its actual members list length
        count_fixed = 0
        for i, row in circles_df.iterrows():
            circle_id = row['circle_id']
            members_list = row['members'] if 'members' in row else []
            
            # Get the actual member count from the members list
            actual_count = len(members_list) if isinstance(members_list, list) else 0
            
            # If member_count doesn't match the actual number of members, fix it
            if 'member_count' in row and row['member_count'] != actual_count and actual_count > 0:
                old_count = row['member_count']
                circles_df.at[i, 'member_count'] = actual_count
                count_fixed += 1
                print(f"  ✅ FIXED: Circle {circle_id} member_count corrected from {old_count} to {actual_count}")
                
                # Special tracking for test circles
                if circle_id in test_circle_ids:
                    print(f"  🔎 TEST CIRCLE {circle_id}: member_count updated to {actual_count}")
                    print(f"    Members: {members_list}")
        
        print(f"  Fixed member counts for {count_fixed} circles")
        
        # Ensure numeric columns are integers
        for col in ['member_count', 'new_members', 'always_hosts', 'sometimes_hosts', 'max_additions']:
            if col in circles_df.columns:
                circles_df[col] = pd.to_numeric(circles_df[col], errors='coerce').fillna(0).astype(int)
                
        # Sort by circle ID
        if 'circle_id' in circles_df.columns:
            circles_df = circles_df.sort_values('circle_id')
    
    print(f"  Successfully created circles DataFrame with {len(circles_df)} circles")
    
    # Critical debug for member counts - we need to make a direct fix
    print("\n🔍 CRITICAL DEBUG: MEMBER COUNT VERIFICATION - Final Fix")
    count_fixed = 0

    # Check if we have continuing/existing circles that should have multiple members but show just 1
    for i, row in circles_df.iterrows():
        circle_id = row['circle_id']
        is_continuing = row.get('is_existing', False)
        is_new = row.get('is_new_circle', False) 
        member_count = row.get('member_count', 0)
        continuing_count = row.get('continuing_members', 0)
        
        # Debug output for identified test circles
        if circle_id in test_circle_ids or is_continuing:
            print(f"  🔍 DIRECT CHECK circle {circle_id}:")
            print(f"    member_count = {member_count}, continuing_members = {continuing_count}")
            print(f"    is_existing = {is_continuing}, is_new_circle = {is_new}")
            
            # Check if we have a mismatch - existing circle with only 1 member
            if is_continuing and continuing_count > 0 and member_count == 1:
                # CRITICAL FIX: Based on continuing_members, force update the member_count
                old_count = member_count
                circles_df.at[i, 'member_count'] = max(member_count, continuing_count)
                count_fixed += 1
                print(f"  🔧 DIRECT FIX: Circle {circle_id} member_count forced from {old_count} to {circles_df.at[i, 'member_count']}")
    
    if count_fixed > 0:
        print(f"  ✅ Direct-fixed member counts for {count_fixed} continuing circles with incorrect counts")

    # Note: We leave meeting times blank if not available, per user instruction
    missing_time_count = 0
    for i, row in circles_df.iterrows():
        circle_id = row['circle_id']
        
        # Count how many meeting times are missing but don't add fallbacks
        if 'meeting_time' not in row or safe_isna(row['meeting_time']):
            missing_time_count += 1
            print(f"  ⚠️ Circle {circle_id} has missing meeting time - leaving blank")
            
    if missing_time_count > 0:
        print(f"  Found {missing_time_count} circles with missing meeting times - leaving blank as instructed")
    
    # Enhanced debugging - show all tracked test circles
    for circle_id in test_circle_ids:
        test_circle = circles_df[circles_df['circle_id'] == circle_id]
        if not test_circle.empty:
            row = test_circle.iloc[0]
            print(f"  🔍 FINAL TEST CIRCLE: {circle_id}: {row['member_count']} members "
                 f"({row.get('new_members', 0)} new, {row.get('continuing_members', 0)} continuing, "
                 f"max_additions={row.get('max_additions', 0)})")
            print(f"     Meeting time: '{row.get('meeting_time', 'None')}'")
        else:
            print(f"  ⚠️ TEST CIRCLE {circle_id} not found in final circles DataFrame")
    
    # Debug - show a few sample circles
    if not circles_df.empty and len(circles_df) > 0:
        print("  Sample circles:")
        for i, (_, row) in enumerate(circles_df.head(5).iterrows()):
            print(f"    {i+1}. {row['circle_id']}: {row['member_count']} members "
                 f"({row.get('new_members', 0)} new, {row.get('continuing_members', 0)} continuing, "
                 f"max_additions={row.get('max_additions', 0)})")
    
    return circles_df