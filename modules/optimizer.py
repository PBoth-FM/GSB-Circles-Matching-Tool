import pandas as pd
import numpy as np
import pulp
import time
from itertools import combinations
from utils.helpers import determine_unmatched_reason

def run_matching_algorithm(data, config):
    """
    Run the optimization algorithm to match participants into circles
    
    Args:
        data: DataFrame with processed participant data
        config: Dictionary with configuration parameters
        
    Returns:
        Tuple of (results DataFrame, matched_circles DataFrame, unmatched_participants DataFrame)
    """
    # Initialize optimization logs
    import streamlit as st
    if 'optimization_logs' not in st.session_state:
        st.session_state.optimization_logs = ""
    
    # Capture existing stdout to enable logging
    import sys
    from io import StringIO
    
    # Create a string buffer to capture print statements
    log_capture = StringIO()
    original_stdout = sys.stdout
    sys.stdout = log_capture
    # Extract configuration parameters
    min_circle_size = config.get('min_circle_size', 5)
    existing_circle_handling = config.get('existing_circle_handling', 'preserve')
    debug_mode = config.get('debug_mode', False)
    enable_host_requirement = config.get('enable_host_requirement', True)
    
    # Copy data to avoid modifying original
    df = data.copy()
    
    # Print the status counts for debugging
    if debug_mode:
        status_counts = df['Status'].value_counts().to_dict()
        print(f"Input data status counts: {status_counts}")
        print(f"Available columns: {df.columns.tolist()}")
    
    # STEP 0: DIRECT HANDLING OF CURRENT-CONTINUING PARTICIPANTS
    # This ensures existing circles are maintained exactly as they were
    
    # First, identify the current circle ID column
    current_circle_col = None
    potential_columns = ['current_circles_id', 'Current_Circle_ID', 'Current Circle ID']
    
    # Try direct matches first
    for col in potential_columns:
        if col in df.columns:
            current_circle_col = col
            break
            
    # If not found, try case-insensitive matching
    if current_circle_col is None:
        for col in df.columns:
            if col.lower() in [c.lower() for c in potential_columns]:
                current_circle_col = col
                break
                
    if debug_mode:
        if current_circle_col:
            print(f"DIRECT CONTINUATION: Found Current Circle ID column: '{current_circle_col}'")
        else:
            print("DIRECT CONTINUATION: ERROR - Could not find Current Circle ID column!")
    
    # Initialize tracking containers for direct circle continuation
    directly_continued_circles = {}  # Maps circle_id to circle data
    processed_participants = set()   # Track participants already assigned
    direct_circles_list = []         # Track circle metadata
    direct_results = []              # Track direct assignments

    # Only proceed if we found the current circle column and are in preserve mode
    if current_circle_col and existing_circle_handling == 'preserve':
        # Find all CURRENT-CONTINUING participants with circle IDs
        continuing_df = df[(df['Status'] == 'CURRENT-CONTINUING') & df[current_circle_col].notna()]
        
        if debug_mode:
            print(f"DIRECT CONTINUATION: Found {len(continuing_df)} CURRENT-CONTINUING participants with circle IDs")
        
        if not continuing_df.empty:
            # Group participants by their current circle ID
            circle_groups = continuing_df.groupby(current_circle_col)
            
            if debug_mode:
                print(f"DIRECT CONTINUATION: Found {len(circle_groups)} distinct circles")
            
            # Process each circle group
            for circle_id, group in circle_groups:
                # Skip empty or NaN circle IDs
                if not circle_id or pd.isna(circle_id):
                    continue
                    
                # Convert to string and standardize
                circle_id = str(circle_id).strip()
                
                if debug_mode:
                    print(f"DIRECT CONTINUATION: Processing circle {circle_id} with {len(group)} members")
                
                # Get the current region, subregion and meeting time from the first member
                first_member = group.iloc[0]
                current_region = first_member.get('Current_Region', '')
                current_subregion = first_member.get('Current_Subregion', '')
                
                # Check for column names that might contain the meeting day/time information
                day_column = None
                time_column = None
                
                # Try different potential column names for meeting day
                for col_name in ['Current_Meeting_Day', 'Current Meeting Day', 'Current/ Continuing Meeting Day']:
                    if col_name in first_member:
                        day_column = col_name
                        break
                
                # Try different potential column names for meeting time
                for col_name in ['Current_Meeting_Time', 'Current Meeting Time', 'Current/ Continuing Meeting Time']:
                    if col_name in first_member:
                        time_column = col_name
                        break
                
                # Get meeting day and time, defaulting to 'Varies' if not found
                current_meeting_day = first_member.get(day_column, 'Varies') if day_column else 'Varies'
                current_meeting_time = first_member.get(time_column, 'Varies') if time_column else 'Varies'
                
                # Ensure non-empty strings for day and time
                current_meeting_day = current_meeting_day if pd.notna(current_meeting_day) and current_meeting_day else 'Varies'
                current_meeting_time = current_meeting_time if pd.notna(current_meeting_time) and current_meeting_time else 'Varies'
                
                # Format the day/time combination for proposed_NEW_DayTime using the standard format
                formatted_day_time = f"{current_meeting_day} ({current_meeting_time})"
                
                if debug_mode:
                    print(f"For circle {circle_id}, using day '{current_meeting_day}' and time '{current_meeting_time}'")
                    print(f"Formatted meeting time: '{formatted_day_time}'")
                
                # Check if it's an in-person or virtual circle
                is_in_person = circle_id.startswith('IP-') and not circle_id.startswith('IP-NEW-')
                is_virtual = circle_id.startswith('V-') and not circle_id.startswith('V-NEW-')
                
                # Count hosts in the circle
                always_hosts = sum(1 for _, row in group.iterrows() if str(row.get('host', '')).lower() in ['always', 'always host'])
                sometimes_hosts = sum(1 for _, row in group.iterrows() if str(row.get('host', '')).lower() in ['sometimes', 'sometimes host'])
                
                # Skip host requirement check for now - keep all circles regardless
                
                # Create a direct circle assignment for all members
                member_ids = group['Encoded ID'].tolist()
                
                if debug_mode:
                    print(f"DIRECT CONTINUATION: Directly assigning {len(member_ids)} members to continuing circle {circle_id}")
                
                # Calculate max_additions for this circle based on co-leader preferences
                max_additions = None  # Start with None to indicate no limit specified yet
                has_none_preference = False
                has_co_leader = False
                
                for _, member in group.iterrows():
                    # Only consider preferences from current co-leaders
                    # Try to find the co-leader column with the correct name (could be 'Current Co-Leader?' or 'Current_Co_Leader')
                    co_leader_value = ''
                    if 'Current Co-Leader?' in member:
                        co_leader_value = str(member.get('Current Co-Leader?', ''))
                    elif 'Current_Co_Leader' in member:
                        co_leader_value = str(member.get('Current_Co_Leader', ''))
                    
                    is_current_co_leader = co_leader_value.strip().lower() == 'yes'
                    
                    if debug_mode and circle_id in ['IP-SIN-01', 'IP-LON-04']:
                        print(f"  Co-Leader check: value='{co_leader_value}', is_co_leader={is_current_co_leader}")
                    
                    # Skip non-co-leaders
                    if not is_current_co_leader:
                        continue
                    
                    # Mark that at least one co-leader was found
                    has_co_leader = True
                    
                    # Get the max new members value if present
                    max_value = member.get('co_leader_max_new_members', None)
                    
                    if debug_mode and pd.notna(max_value):
                        print(f"  Co-Leader {member['Encoded ID']} specified max new members: {max_value}")
                    
                    # Check for "None" literal string preference
                    if isinstance(max_value, str) and max_value.lower() == "none":
                        has_none_preference = True
                        if debug_mode:
                            print(f"  Co-Leader {member['Encoded ID']} specified 'None' - no new members allowed")
                        break
                    
                    # Check for 0 which should be treated like "None"
                    elif pd.notna(max_value) and (
                        (isinstance(max_value, (int, float)) and max_value == 0) or
                        (isinstance(max_value, str) and max_value.strip() == "0")
                    ):
                        has_none_preference = True
                        if debug_mode:
                            print(f"  Co-Leader {member['Encoded ID']} specified '0' - no new members allowed")
                        break
                    
                    # Process numeric values
                    elif pd.notna(max_value):
                        try:
                            int_value = int(max_value)
                            # If first valid value or smaller than previous minimum
                            if max_additions is None or int_value < max_additions:
                                max_additions = int_value
                                if debug_mode and circle_id in ['IP-SIN-01', 'IP-LON-04']:
                                    print(f"  Set max_additions to {int_value} based on co-leader preference for circle {circle_id}")
                        except (ValueError, TypeError):
                            # Not a valid number, ignore
                            if debug_mode:
                                print(f"  Invalid max new members value: {max_value}")
                
                # Set max_additions based on rules
                if has_none_preference:
                    # Any co-leader saying "None" means no new members
                    final_max_additions = 0
                    if debug_mode:
                        print(f"  Circle {circle_id} has 'None' preference from co-leader - not accepting new members")
                elif max_additions is not None:
                    # Use the minimum valid value provided by a co-leader
                    final_max_additions = max_additions
                    if debug_mode:
                        print(f"  Circle {circle_id} can accept up to {final_max_additions} new members (co-leader preference)")
                else:
                    # Default to 8 total if no co-leader specified a value or no co-leaders exist
                    final_max_additions = max(0, 8 - len(member_ids))
                    if debug_mode:
                        message = "No co-leader preference specified" if has_co_leader else "No co-leaders found"
                        print(f"  {message} for circle {circle_id} - using default max total of 8")
                        print(f"  Currently has {len(member_ids)} members, can accept {final_max_additions} more")
                
                # Create circle metadata
                circle_data = {
                    'circle_id': circle_id,
                    'region': current_region,
                    'subregion': current_subregion,
                    'meeting_time': formatted_day_time,
                    'member_count': len(member_ids),
                    'new_members': 0,  # No new members in directly continued circles
                    'always_hosts': always_hosts,
                    'sometimes_hosts': sometimes_hosts,
                    'members': member_ids,
                    'is_direct_continuation': True,
                    'max_additions': final_max_additions
                }
                direct_circles_list.append(circle_data)
                
                # Process each member of this circle
                for _, member in group.iterrows():
                    member_dict = member.to_dict()
                    member_id = member['Encoded ID']
                    
                    # Direct assignment - skip optimization
                    member_dict['proposed_NEW_circles_id'] = circle_id
                    member_dict['proposed_NEW_Subregion'] = current_subregion
                    member_dict['proposed_NEW_DayTime'] = formatted_day_time
                    
                    # Handle host status
                    if str(member.get('host', '')).lower() in ['always', 'always host']:
                        member_dict['proposed_NEW_host'] = "Yes"
                    elif str(member.get('host', '')).lower() in ['sometimes', 'sometimes host']:
                        member_dict['proposed_NEW_host'] = "Maybe"
                    else:
                        member_dict['proposed_NEW_host'] = "No"
                    
                    # Set co-leader status (first always host or sometimes host)
                    if member_dict['proposed_NEW_host'] == "Yes" and not any(r.get('proposed_NEW_co_leader') == "Yes" for r in direct_results if r.get('proposed_NEW_circles_id') == circle_id):
                        member_dict['proposed_NEW_co_leader'] = "Yes"
                    else:
                        member_dict['proposed_NEW_co_leader'] = "No"
                    
                    # Add max_additions value to individual participants
                    member_dict['max_additions'] = final_max_additions
                    
                    # Mark as processed and add to results
                    direct_results.append(member_dict)
                    processed_participants.add(member_id)
    
    if debug_mode and current_circle_col:
        print(f"DIRECT CONTINUATION: Directly assigned {len(processed_participants)} participants to {len(direct_circles_list)} circles")
    
    # Create a filtered copy of the dataframe excluding directly processed participants
    remaining_df = df[~df['Encoded ID'].isin(processed_participants)].copy()
    
    if debug_mode:
        print(f"DIRECT CONTINUATION: {len(remaining_df)} participants remaining for regular optimization")
        if 'Status' in remaining_df.columns:
            status_counts = remaining_df['Status'].value_counts().to_dict()
            print(f"DIRECT CONTINUATION: Remaining status counts: {status_counts}")
    
    # Group participants by derived region (Current_Region for CURRENT-CONTINUING, Requested_Region for others)
    # If Derived_Region exists (added in data_processor.normalize_data), use it
    region_column = 'Derived_Region' if 'Derived_Region' in df.columns else 'Requested_Region'
    
    if debug_mode:
        print(f"Using {region_column} for region grouping according to PRD 4.3.2")
    
    regions = remaining_df[region_column].unique()
    
    # Initialize results containers
    all_results = direct_results.copy()  # Start with directly continued participants
    all_circles = direct_circles_list.copy()  # Start with directly continued circles
    all_unmatched = []
    
    # Process each region separately
    for region in regions:
        if debug_mode:
            print(f"Processing region: {region}")
        
        region_df = remaining_df[remaining_df[region_column] == region]
        
        if debug_mode:
            status_counts = region_df['Status'].value_counts().to_dict() if 'Status' in region_df.columns else {}
            print(f"Region {region} has {len(region_df)} participants: {status_counts}")
        
        # Skip regions with too few participants
        if len(region_df) < min_circle_size:
            # Mark all as unmatched due to insufficient participants
            for _, participant in region_df.iterrows():
                participant_dict = participant.to_dict()
                participant_dict['proposed_NEW_circles_id'] = "UNMATCHED"
                
                # Set scores to 0 for unmatched participants
                participant_dict['location_score'] = 0
                participant_dict['time_score'] = 0
                participant_dict['total_score'] = 0
                
                # Use our enhanced reason determination with appropriate context
                reason_context = {"insufficient_regional_participants": True}
                participant_dict['unmatched_reason'] = determine_unmatched_reason(participant, reason_context)
                
                all_unmatched.append(participant_dict)
                all_results.append(participant_dict)
            continue
        
        # Run optimization for this region
        region_results, region_circles, region_unmatched = optimize_region(
            region, region_df, min_circle_size, enable_host_requirement, existing_circle_handling, debug_mode
        )
        
        # Add to overall results
        all_results.extend(region_results)
        all_circles.extend(region_circles)
        all_unmatched.extend(region_unmatched)
    
    # Convert results to DataFrames
    results_df = pd.DataFrame(all_results)
    circles_df = pd.DataFrame(all_circles) if all_circles else pd.DataFrame()
    unmatched_df = pd.DataFrame(all_unmatched) if all_unmatched else pd.DataFrame()
    
    # Ensure Class_Vintage is properly preserved in results
    if not results_df.empty and 'Class_Vintage' not in results_df.columns and 'Class_Vintage' in data.columns:
        print(f"Adding Class_Vintage from original data to results")
        # Create a mapping from Encoded ID to Class_Vintage
        id_to_vintage = data[['Encoded ID', 'Class_Vintage']].set_index('Encoded ID').to_dict()['Class_Vintage']
        # Add Class_Vintage to results using the mapping
        results_df['Class_Vintage'] = results_df['Encoded ID'].map(id_to_vintage)
        print(f"Added Class_Vintage to {results_df['Class_Vintage'].notna().sum()} rows")
    
    # Ensure numeric columns are properly typed to avoid comparison issues
    if not circles_df.empty:
        # Convert numeric columns to int to avoid string/float comparison issues
        for col in ['member_count', 'new_members', 'always_hosts', 'sometimes_hosts']:
            if col in circles_df.columns:
                circles_df[col] = pd.to_numeric(circles_df[col], errors='coerce').fillna(0).astype(int)
    
    # Calculate final metrics
    if not results_df.empty:
        matched_count = len(results_df[results_df['proposed_NEW_circles_id'] != "UNMATCHED"])
        total_count = len(results_df)
        matched_percentage = (matched_count / total_count) * 100 if total_count > 0 else 0
        
        if debug_mode:
            print(f"Matched {matched_count} out of {total_count} participants ({matched_percentage:.2f}%)")
    
    # Restore original stdout
    sys.stdout = original_stdout
    
    # Capture logs
    logs = log_capture.getvalue()
    
    # Add logs to session state for display in the Debug tab
    if 'optimization_logs' in st.session_state:
        st.session_state.optimization_logs = logs
    
    return results_df, circles_df, unmatched_df

def optimize_region(region, region_df, min_circle_size, enable_host_requirement, existing_circle_handling, debug_mode=False):
    """
    Optimize matching within a single region
    
    Args:
        region: Region name
        region_df: DataFrame with participants from this region
        min_circle_size: Minimum number of participants per circle
        enable_host_requirement: Whether to enforce host requirements
        existing_circle_handling: How to handle existing circles ('preserve', 'dissolve', 'optimize')
        debug_mode: Whether to print debug information
        
    Returns:
        Tuple of (results list, circles list, unmatched list)
    """
    # Context information for enhanced unmatched reason determination
    optimization_context = {
        'existing_circles': [],
        'similar_participants': {},
        'full_circles': [],
        'circles_needing_hosts': [],
        'host_counts': {},
        'compatibility_matrix': {},  # Track compatibility between participants and circle options
        'participant_compatible_options': {},  # Count of compatible options per participant
        'location_time_pairs': []  # All possible location-time combinations
    }
    # Initialize containers for results
    results = []
    circles = []
    unmatched = []
    
    # Track timing for performance analysis
    import time
    start_time = time.time()
    
    # Initialize containers for existing circle handling
    existing_circles = {}  # Maps circle_id to circle data for viable circles (>= min_circle_size)
    small_circles = {}     # Maps circle_id to circle data for small circles (2-4 members)
    current_circle_members = {}  # Maps circle_id to list of members
    
    # Step 1: Identify existing circles if we're preserving them
    if existing_circle_handling == 'preserve':
        # Check for circle ID column (case-insensitive to handle column mapping issues)
        # In our column mapping, it's now 'Current_Circle_ID' 
        current_col = None
        potential_columns = ['current_circles_id', 'Current_Circle_ID', 'Current Circle ID']
        
        # Print potential column names for debugging
        if debug_mode:
            print(f"Looking for circle ID column. Options: {potential_columns}")
            print(f"Available columns: {region_df.columns.tolist()}")
        
        # First try direct matches
        for col in potential_columns:
            if col in region_df.columns:
                current_col = col
                break
                
        # If not found, try case-insensitive matching
        if current_col is None:
            for col in region_df.columns:
                if col.lower() in [c.lower() for c in potential_columns]:
                    current_col = col
                    break
                
        if current_col is None and debug_mode:
            print(f"CRITICAL ERROR: Could not find current circles ID column. Available columns: {region_df.columns.tolist()}")
            return [], [], []  # Return empty results if we can't find the critical column
            
        if current_col is not None:
            if debug_mode:
                print(f"Using column '{current_col}' for current circle IDs")
                continuing_count = len(region_df[region_df['Status'] == 'CURRENT-CONTINUING'])
                circles_count = region_df[region_df['Status'] == 'CURRENT-CONTINUING'][current_col].notna().sum()
                print(f"Found {continuing_count} CURRENT-CONTINUING participants, {circles_count} with circle IDs")
                
            # Group participants by their current circle
            for _, row in region_df.iterrows():
                # First, check if it's a CURRENT-CONTINUING participant
                if row.get('Status') == 'CURRENT-CONTINUING':
                    # They must have a current circle ID - this is required for CURRENT-CONTINUING
                    if pd.notna(row.get(current_col)):
                        circle_id = str(row[current_col]).strip()
                        if circle_id:
                            if circle_id not in current_circle_members:
                                current_circle_members[circle_id] = []
                            current_circle_members[circle_id].append(row)
                        else:
                            # They're CURRENT-CONTINUING but have an empty circle ID
                            # This shouldn't happen per the spec, but log for debugging
                            if debug_mode:
                                print(f"WARNING: CURRENT-CONTINUING participant {row['Encoded ID']} has empty circle ID")
                    else:
                        # They're CURRENT-CONTINUING but circle ID is null
                        # This shouldn't happen per the spec, but log for debugging
                        if debug_mode:
                            print(f"WARNING: CURRENT-CONTINUING participant {row['Encoded ID']} has null circle ID")
        
        # Evaluate each existing circle in the region
        # Note: By this point, direct continuation has already been done in the main function
        # so we only need to handle edge cases here
        for circle_id, members in current_circle_members.items():
            # Per PRD: An existing circle is maintained if it has at least 2 CURRENT-CONTINUING members
            # and meets host requirements (for in-person circles)
            if len(members) >= 2:
                # Check if it's an in-person circle (IP prefix) or virtual circle (V prefix)
                is_in_person = circle_id.startswith('IP-') and not circle_id.startswith('IP-NEW-')
                is_virtual = circle_id.startswith('V-') and not circle_id.startswith('V-NEW-')
                
                # For in-person circles, check host requirements
                host_requirement_met = True
                if is_in_person and enable_host_requirement:
                    has_host = any(m.get('host', '').lower() in ['always', 'always host', 'sometimes', 'sometimes host'] for m in members)
                    host_requirement_met = has_host
                
                if host_requirement_met:
                    # Get subregion and time if available
                    subregion = members[0].get('Current_Subregion', '')
                    
                    # Check for column names that might contain the meeting day/time information
                    day_column = None
                    time_column = None
                    
                    # Try different potential column names for meeting day
                    for col_name in ['Current_Meeting_Day', 'Current Meeting Day', 'Current/ Continuing Meeting Day']:
                        if col_name in members[0]:
                            day_column = col_name
                            break
                    
                    # Try different potential column names for meeting time
                    for col_name in ['Current_Meeting_Time', 'Current Meeting Time', 'Current/ Continuing Meeting Time']:
                        if col_name in members[0]:
                            time_column = col_name
                            break
                    
                    # Get meeting day and time, defaulting to 'Varies' if not found
                    meeting_day = members[0].get(day_column, 'Varies') if day_column else 'Varies'
                    meeting_time = members[0].get(time_column, 'Varies') if time_column else 'Varies'
                    
                    # Ensure non-empty strings for day and time
                    meeting_day = meeting_day if pd.notna(meeting_day) and meeting_day else 'Varies'
                    meeting_time = meeting_time if pd.notna(meeting_time) and meeting_time else 'Varies'
                    
                    # Standardize time format to plural form (e.g., "Evening" -> "Evenings", "Day" -> "Days")
                    # This ensures consistency between existing circles and new participant preferences
                    if meeting_time.lower() == 'evening':
                        meeting_time = 'Evenings'
                    elif meeting_time.lower() == 'day':
                        meeting_time = 'Days'
                    
                    # Format the day/time combination for proposed_NEW_DayTime using the standard format
                    formatted_meeting_time = f"{meeting_day} ({meeting_time})"
                    
                    if debug_mode:
                        print(f"For existing circle_id {circle_id}, using day '{meeting_day}' and time '{meeting_time}'")
                        print(f"Formatted meeting time: '{formatted_meeting_time}'")
                    
                    # Calculate max_additions for this circle based on co-leader preferences
                    max_additions = None  # Start with None to indicate no limit specified yet
                    has_none_preference = False
                    has_co_leader = False
                    
                    for member in members:
                        # Only consider preferences from current co-leaders
                        # Try to find the co-leader column with the correct name
                        co_leader_value = ''
                        if 'Current Co-Leader?' in member:
                            co_leader_value = str(member.get('Current Co-Leader?', ''))
                        elif 'Current_Co_Leader' in member:
                            co_leader_value = str(member.get('Current_Co_Leader', ''))
                        
                        is_current_co_leader = co_leader_value.strip().lower() == 'yes'
                        
                        if debug_mode and circle_id in ['IP-SIN-01', 'IP-LON-04']:
                            print(f"  Co-Leader check in existing circle preservation: value='{co_leader_value}', is_co_leader={is_current_co_leader}")
                        
                        # Skip non-co-leaders
                        if not is_current_co_leader:
                            continue
                        
                        # Mark that at least one co-leader was found
                        has_co_leader = True
                        
                        # Get the max new members value if present
                        max_value = member.get('co_leader_max_new_members', None)
                        
                        if debug_mode and pd.notna(max_value):
                            print(f"  Co-Leader {member['Encoded ID']} specified max new members: {max_value}")
                        
                        # Check for "None" literal string preference
                        if isinstance(max_value, str) and max_value.lower() == "none":
                            has_none_preference = True
                            if debug_mode:
                                print(f"  Co-Leader {member['Encoded ID']} specified 'None' - no new members allowed")
                            break
                        
                        # Check for 0 which should be treated like "None"
                        elif pd.notna(max_value) and (
                            (isinstance(max_value, (int, float)) and max_value == 0) or
                            (isinstance(max_value, str) and max_value.strip() == "0")
                        ):
                            has_none_preference = True
                            if debug_mode:
                                print(f"  Co-Leader {member['Encoded ID']} specified '0' - no new members allowed")
                            break
                        
                        # Process numeric values
                        elif pd.notna(max_value):
                            try:
                                int_value = int(max_value)
                                # If first valid value or smaller than previous minimum
                                if max_additions is None or int_value < max_additions:
                                    max_additions = int_value
                            except (ValueError, TypeError):
                                # Not a valid number, ignore
                                if debug_mode:
                                    print(f"  Invalid max new members value: {max_value}")
                    
                    # Set max_additions based on rules
                    if has_none_preference:
                        # Any co-leader saying "None" means no new members
                        final_max_additions = 0
                        if debug_mode:
                            print(f"  Circle {circle_id} has 'None' preference from co-leader - not accepting new members")
                    elif max_additions is not None:
                        # Use the minimum valid value provided by co-leaders
                        final_max_additions = max_additions
                        if debug_mode:
                            print(f"  Circle {circle_id} can accept up to {final_max_additions} new members (co-leader preference)")
                    else:
                        # Default to 8 total if no co-leader specified a value or no co-leaders exist
                        final_max_additions = max(0, 8 - len(members))
                        if debug_mode:
                            message = "No co-leader preference specified" if has_co_leader else "No co-leaders found"
                            print(f"  {message} for circle {circle_id} - using default max total of 8")
                            print(f"  Currently has {len(members)} members, can accept {final_max_additions} more")
                    
                    # Create circle data with member list and metadata
                    circle_data = {
                        'members': [m['Encoded ID'] for m in members],
                        'region': region,  # Add region to ensure we can filter properly by region
                        'subregion': subregion,
                        'meeting_time': formatted_meeting_time,
                        'always_hosts': sum(1 for m in members if m.get('host', '').lower() in ['always', 'always host']),
                        'sometimes_hosts': sum(1 for m in members if m.get('host', '').lower() in ['sometimes', 'sometimes host']),
                        'is_in_person': is_in_person,
                        'is_virtual': is_virtual,
                        'max_additions': final_max_additions
                    }
                    
                    # Special extensive debug for our example circles
                    if circle_id in ['IP-SIN-01', 'IP-LON-04']:
                        print(f"\n🔍 CRITICAL DEBUG FOR CIRCLE {circle_id}:")
                        print(f"  Current members: {len(members)}")
                        print(f"  Max additions set to: {final_max_additions}")
                        print(f"  Meeting time: {formatted_meeting_time}")
                        print(f"  Is determining max_additions - has co-leader: {has_co_leader}")
                        print(f"  Is determining max_additions - has 'None' preference: {has_none_preference}")
                        print(f"  Is determining max_additions - explicit max_additions value: {max_additions}")
                        print(f"  Rules applied: {'Co-leader None preference' if has_none_preference else 'Co-leader explicit value' if max_additions is not None else 'Default max total of 8'}")
                    
                    # Per PRD: Small circles (2-4 members) need to grow to be viable
                    if len(members) < min_circle_size:
                        small_circles[circle_id] = circle_data
                        if debug_mode:
                            print(f"Small existing circle {circle_id} has {len(members)} members, needs {min_circle_size - len(members)} more")
                    else:
                        # Viable circle (>= min_circle_size) - add to existing circles
                        existing_circles[circle_id] = circle_data
                        if debug_mode:
                            print(f"Preserving viable existing circle {circle_id} with {len(members)} members")
    
    # Step 2: Process participants in existing viable circles
    processed_ids = set()
    for circle_id, circle_data in existing_circles.items():
        # Process each member of this circle
        for encoded_id in circle_data['members']:
            # Find the participant in the region data
            participant = region_df[region_df['Encoded ID'] == encoded_id].iloc[0].to_dict()
            
            # Add the participant to results list with their circle assignment
            participant['proposed_NEW_circles_id'] = circle_id
            participant['proposed_NEW_Subregion'] = circle_data['subregion']
            participant['proposed_NEW_DayTime'] = circle_data['meeting_time']
            
            # Calculate preference match scores for existing circle assignment
            subregion = circle_data['subregion']
            time_slot = circle_data['meeting_time']
            
            # Calculate location score
            loc_score = 0
            if participant.get('first_choice_location') == subregion:
                loc_score = 3
            elif participant.get('second_choice_location') == subregion:
                loc_score = 2
            elif participant.get('third_choice_location') == subregion:
                loc_score = 1
                
            # Calculate time score
            time_score = 0
            if participant.get('first_choice_time') == time_slot:
                time_score = 3
            elif participant.get('second_choice_time') == time_slot:
                time_score = 2
            elif participant.get('third_choice_time') == time_slot:
                time_score = 1
                
            # Update scores
            participant['location_score'] = loc_score
            participant['time_score'] = time_score
            participant['total_score'] = loc_score + time_score
            
            # Handle host status
            if participant.get('host', '').lower() in ['always', 'always host']:
                participant['proposed_NEW_host'] = "Yes"
            elif participant.get('host', '').lower() in ['sometimes', 'sometimes host']:
                participant['proposed_NEW_host'] = "Maybe"
            else:
                participant['proposed_NEW_host'] = "No"
            
            # Add max_additions value to individual participants
            participant['max_additions'] = circle_data.get('max_additions', 0)
            
            # Add to results and mark as processed
            results.append(participant)
            processed_ids.add(encoded_id)
        
        # Add the circle to circles list
        circle_dict = {
            'circle_id': circle_id,
            'region': region,
            'subregion': circle_data['subregion'],
            'meeting_time': circle_data['meeting_time'],
            'member_count': len(circle_data['members']),
            'new_members': 0,  # No new members in continuing circles
            'always_hosts': circle_data['always_hosts'],
            'sometimes_hosts': circle_data['sometimes_hosts'],
            'members': circle_data['members'],
            'max_additions': circle_data.get('max_additions', 0)  # Include max_additions from the circle_data
        }
        circles.append(circle_dict)
    
    # Step 2.5: Handle small existing circles (2-4 members)
    # Per PRD: Try to add participants to reach 5 members. If not possible, return to general pool.
    if small_circles and debug_mode:
        print(f"Found {len(small_circles)} small circles with 2-4 members that need to grow")
        
    # Get the pool of participants we can add to small circles (exclude already processed)
    available_df = region_df[~region_df['Encoded ID'].isin(processed_ids)]
    
    # Try to grow each small circle to viable size
    grown_small_circles = {}
    small_circle_members_to_keep = set()
    
    for circle_id, circle_data in small_circles.items():
        current_members = circle_data['members']
        current_size = len(current_members)
        
        # Check if max_additions is 0 (meaning "None" preference)
        if 'max_additions' in circle_data and circle_data['max_additions'] == 0:
            if debug_mode:
                print(f"Circle {circle_id} has max_additions=0, not growing (co-leader specified 'None')")
            continue
        
        # Calculate actual growth limit based on max_additions
        max_growth = circle_data.get('max_additions', min_circle_size - current_size)
        needed_size = min(min_circle_size - current_size, max_growth)
        
        if debug_mode:
            print(f"Small circle {circle_id} has {current_size} members, needs {needed_size} more (max_additions={circle_data.get('max_additions', 'default')})")
        
        # Skip if no growth is needed or allowed
        if needed_size <= 0:
            if debug_mode:
                print(f"Circle {circle_id} already at or over capacity, not growing")
            continue
            
        # Skip circles that are impossible to grow given available pool
        if needed_size > len(available_df):
            if debug_mode:
                print(f"Circle {circle_id} needs {needed_size} members but only {len(available_df)} available - cannot grow")
            continue
        
        # Find compatible participants for this circle
        # Use the subregion and meeting time from the circle
        subregion = circle_data['subregion']
        meeting_time = circle_data['meeting_time']
        
        # Score each available participant based on compatibility
        compatibility_scores = []
        for _, row in available_df.iterrows():
            participant_id = row['Encoded ID']
            # Calculate preference score for this subregion and time slot
            score = calculate_preference_score(row, subregion, meeting_time)
            compatibility_scores.append((participant_id, score))
        
        # Sort by compatibility score (highest first)
        compatibility_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top N needed participants
        best_matches = compatibility_scores[:needed_size]
        
        # Check if we have enough good matches (score > 0)
        good_matches = [p for p, score in best_matches if score > 0]
        
        if len(good_matches) >= needed_size:
            # We have enough good matches to grow this circle
            new_members = [p for p, _ in best_matches[:needed_size]]
            
            # Update the circle data
            circle_data['members'].extend(new_members)
            circle_data['member_count'] = len(circle_data['members'])
            circle_data['new_members'] = needed_size
            
            # Count the hosts in the combined group
            all_member_rows = []
            for encoded_id in circle_data['members']:
                member_rows = region_df[region_df['Encoded ID'] == encoded_id]
                if not member_rows.empty:
                    all_member_rows.append(member_rows.iloc[0])
            
            circle_data['always_hosts'] = sum(1 for m in all_member_rows if m.get('host', '').lower() in ['always', 'always host'])
            circle_data['sometimes_hosts'] = sum(1 for m in all_member_rows if m.get('host', '').lower() in ['sometimes', 'sometimes host'])
            
            # Check host requirements
            host_requirement_met = True
            if circle_data['is_in_person'] and enable_host_requirement:
                has_host = circle_data['always_hosts'] > 0 or circle_data['sometimes_hosts'] >= 2
                host_requirement_met = has_host
            
            if host_requirement_met:
                # This small circle has grown into a viable circle
                grown_small_circles[circle_id] = circle_data
                small_circle_members_to_keep.update(current_members)
                small_circle_members_to_keep.update(new_members)
                
                if debug_mode:
                    print(f"Successfully grew circle {circle_id} by adding {needed_size} members")
            else:
                if debug_mode:
                    print(f"Circle {circle_id} would have enough members but fails host requirements")
    
    # Process the grown small circles
    for circle_id, circle_data in grown_small_circles.items():
        # Process each member of this circle
        for encoded_id in circle_data['members']:
            if encoded_id not in processed_ids:  # Only process members we haven't processed yet
                # Find the participant in the region data
                participant = region_df[region_df['Encoded ID'] == encoded_id].iloc[0].to_dict()
                
                # Add the participant to results list with their circle assignment
                participant['proposed_NEW_circles_id'] = circle_id
                participant['proposed_NEW_Subregion'] = circle_data['subregion']
                participant['proposed_NEW_DayTime'] = circle_data['meeting_time']
                
                # Calculate preference match scores for assigned location and time
                subregion = circle_data['subregion']
                time_slot = circle_data['meeting_time']
                
                # Calculate location score
                loc_score = 0
                if participant.get('first_choice_location') == subregion:
                    loc_score = 3
                elif participant.get('second_choice_location') == subregion:
                    loc_score = 2
                elif participant.get('third_choice_location') == subregion:
                    loc_score = 1
                    
                # Calculate time score
                time_score = 0
                if participant.get('first_choice_time') == time_slot:
                    time_score = 3
                elif participant.get('second_choice_time') == time_slot:
                    time_score = 2
                elif participant.get('third_choice_time') == time_slot:
                    time_score = 1
                    
                # Update scores
                participant['location_score'] = loc_score
                participant['time_score'] = time_score
                participant['total_score'] = loc_score + time_score
                
                # Handle host status
                if participant.get('host', '').lower() in ['always', 'always host']:
                    participant['proposed_NEW_host'] = "Yes"
                elif participant.get('host', '').lower() in ['sometimes', 'sometimes host']:
                    participant['proposed_NEW_host'] = "Maybe"
                else:
                    participant['proposed_NEW_host'] = "No"
                
                # Add max_additions value to individual participants
                participant['max_additions'] = circle_data.get('max_additions', 0)
                
                # Add to results and mark as processed
                results.append(participant)
                processed_ids.add(encoded_id)
        
        # Add the circle to circles list
        circle_dict = {
            'circle_id': circle_id,
            'region': region,
            'subregion': circle_data['subregion'],
            'meeting_time': circle_data['meeting_time'],
            'member_count': len(circle_data['members']),
            'new_members': circle_data['new_members'],  # Number of newly added members
            'always_hosts': circle_data['always_hosts'],
            'sometimes_hosts': circle_data['sometimes_hosts'],
            'members': circle_data['members'],
            'max_additions': circle_data.get('max_additions', 0)  # Include max_additions from the circle_data
        }
        circles.append(circle_dict)
    
    # For small circles that couldn't grow, set preferences from circle data
    # Per PRD: Members of non-viable circles return to general pool with preferences from their circle
    non_viable_circle_members = {}  # Track members from non-viable circles by ID
    
    # Skip single-member circle handling - this is now done directly in the main function
    
    # Process non-viable small circles (2-4 members that couldn't grow)
    single_member_circles = {}  # This is a placeholder to avoid reference errors
    for circle_id, circle_data in small_circles.items():
        if circle_id not in grown_small_circles:
            if debug_mode:
                print(f"Small circle {circle_id} could not grow, returning {len(circle_data['members'])} members to general pool")
            
            # Store the circle data for these members so we can use it for preference setting
            for member_id in circle_data['members']:
                non_viable_circle_members[member_id] = {
                    'format_prefix': 'IP-' if circle_data['is_in_person'] else 'V-',
                    'subregion': circle_data['subregion'],
                    'meeting_time': circle_data['meeting_time']
                }
                
            # Note: These members will be processed in the remaining_df optimization
    
    # Step 3: Run optimization for remaining participants (both NEW and CURRENT-CONTINUING without a current circle)
    remaining_df = region_df[~region_df['Encoded ID'].isin(processed_ids)]
    
    if debug_mode:
        remaining_status_counts = remaining_df['Status'].value_counts().to_dict() if 'Status' in remaining_df.columns else {}
        print(f"Remaining participants status counts: {remaining_status_counts}")
        print(f"Total remaining participants: {len(remaining_df)}")
    
    # Determine all possible subregions and time slots for the remaining participants
    subregions = get_unique_preferences(remaining_df, ['first_choice_location', 'second_choice_location', 'third_choice_location'])
    time_slots = get_unique_preferences(remaining_df, ['first_choice_time', 'second_choice_time', 'third_choice_time'])
    
    # Remove empty values
    subregions = [s for s in subregions if s]
    time_slots = [t for t in time_slots if t]
    
    # Collect information about similar participants at each location-time
    for subregion in subregions:
        for time_slot in time_slots:
            # Count participants who prefer this location and time
            compatible_count = 0
            host_count = 0
            
            for _, participant in remaining_df.iterrows():
                participant_locations = [
                    participant.get('first_choice_location', ''),
                    participant.get('second_choice_location', ''),
                    participant.get('third_choice_location', '')
                ]
                participant_times = [
                    participant.get('first_choice_time', ''),
                    participant.get('second_choice_time', ''),
                    participant.get('third_choice_time', '')
                ]
                
                # Check if this location and time match the participant's preferences
                if (subregion in participant_locations and time_slot in participant_times):
                    compatible_count += 1
                    
                    # Track host availability
                    host_value = str(participant.get('host', '')).lower()
                    if host_value in ['always', 'always host', 'sometimes', 'sometimes host']:
                        host_count += 1
            
            # Add to context
            loc_time_key = (subregion, time_slot)
            optimization_context['similar_participants'][loc_time_key] = compatible_count
            optimization_context['host_counts'][loc_time_key] = host_count
    
    # Add existing circles to context
    # Use the existing_circles dictionary rather than the circles list
    # Filter to only include circles that:
    # 1. Can accept new members (max_additions > 0)
    # 2. Belong to the current region
    viable_circles = [circle for circle_id, circle in existing_circles.items() 
                     if circle.get('max_additions', 0) > 0 and
                        circle.get('region', '') == region]
    
    optimization_context['existing_circles'] = viable_circles
    
    # More detailed debug output to help diagnose issues
    if debug_mode:
        all_regions = set(circle.get('region', '') for circle in existing_circles.values())
        circles_in_region = [circle_id for circle_id, circle in existing_circles.items() 
                           if circle.get('region', '') == region]
        viable_circle_count = len(viable_circles)
        
        print(f"All circles span {len(all_regions)} regions: {all_regions}")
        print(f"Found {len(circles_in_region)} total circles in region {region}")
        print(f"Of those, {viable_circle_count} have capacity (max_additions > 0)")
        
        if viable_circle_count > 0:
            print(f"Viable circles in region {region}:")
            for circle in viable_circles:
                print(f"  {circle.get('circle_id')}: {circle.get('member_count')} members, can add {circle.get('max_additions')} more")
    
    if debug_mode:
        print(f"Found {len(existing_circles)} total existing circles")
        print(f"Adding {len(viable_circles)} circles with capacity (max_additions > 0) to optimization context")
    
    # Track circles at capacity (10 members)
    for circle in circles:
        if circle.get('member_count', 0) >= 10:
            optimization_context['full_circles'].append(circle.get('circle_id'))
    
    # Track circles needing hosts (in a separate loop to avoid LSP confusion)
    for circle in circles:
        if (circle.get('always_hosts', 0) == 0 and 
            circle.get('sometimes_hosts', 0) < 2 and
            circle.get('circle_id', '').startswith('IP-')):
            optimization_context['circles_needing_hosts'].append(circle)
            
    # Removed redundant debug log since we now provide more detailed information above
    
    if debug_mode:
        print(f"Region: {region}, Subregions: {subregions}, Time slots: {time_slots}")
    
    # Handle case where no preferences exist
    if not subregions or not time_slots:
        results = []
        unmatched = []
        
        for _, participant in region_df.iterrows():
            participant_dict = participant.to_dict()
            participant_dict['proposed_NEW_circles_id'] = "UNMATCHED"
            
            # Set scores to 0 for unmatched participants
            participant_dict['location_score'] = 0
            participant_dict['time_score'] = 0
            participant_dict['total_score'] = 0
            
            # Use our enhanced determine_unmatched_reason function with appropriate reason
            if not subregions and not time_slots:
                reason_context = {"no_preferences": True}
                participant_dict['unmatched_reason'] = determine_unmatched_reason(participant, reason_context)
            elif not subregions:
                reason_context = {"no_location_preferences": True} 
                participant_dict['unmatched_reason'] = determine_unmatched_reason(participant, reason_context)
            elif not time_slots:
                reason_context = {"no_time_preferences": True}
                participant_dict['unmatched_reason'] = determine_unmatched_reason(participant, reason_context)
                
            results.append(participant_dict)
            unmatched.append(participant_dict)
            
        return results, [], unmatched
    
    # Generate all possible circle combinations (subregion + time slot)
    circle_options = [(subregion, time_slot) for subregion in subregions for time_slot in time_slots]
    
    # Set up the optimization problem
    prob = pulp.LpProblem(f"CircleMatching_{region}", pulp.LpMaximize)
    
    # Debug: look for specific examples in this region
    example_participants = ['73177784103', '50625303450']
    example_circles = ['IP-SIN-01', 'IP-LON-04']
    
    has_examples = False
    for p_id in example_participants:
        if p_id in remaining_df['Encoded ID'].values:
            has_examples = True
            p_row = remaining_df[remaining_df['Encoded ID'] == p_id].iloc[0]
            print(f"\n==================== DEBUG EXAMPLE ====================")
            print(f"Found example participant {p_id} in region {region}")
            print(f"  Status: {p_row.get('Status', 'Unknown')}")
            print(f"  Circle status: {p_row.get('Circle Status', 'Unknown')}")
            print(f"  Location preferences: 1={p_row['first_choice_location']}, 2={p_row['second_choice_location']}, 3={p_row['third_choice_location']}")
            print(f"  Time preferences: 1={p_row['first_choice_time']}, 2={p_row['second_choice_time']}, 3={p_row['third_choice_time']}")
            print(f"  Host preference: {p_row.get('host', 'Unknown')}")
            print(f"=======================================================")
    
    for c_id in example_circles:
        if c_id in existing_circles:
            has_examples = True
            circle_data = existing_circles[c_id]
            print(f"\n==================== DEBUG EXAMPLE ====================")
            print(f"Found example circle {c_id} in region {region}")
            print(f"  Subregion: {circle_data['subregion']}")
            print(f"  Meeting time: {circle_data['meeting_time']}")
            print(f"  Member count: {circle_data['member_count']}")
            print(f"  Max additions: {circle_data.get('max_additions', 0)}")
            print(f"  Circle members: {circle_data.get('members', [])}")
            print(f"=======================================================")
    
    if has_examples:
        print("\nTHIS IS THE REGION WHERE OUR TEST EXAMPLES ARE LOCATED!")
    
    # Create decision variables: x[i, j] = 1 if participant i is assigned to circle j
    # Must use remaining_df here, not region_df to avoid including already processed participants
    participants = remaining_df['Encoded ID'].tolist()
    
    if debug_mode:
        print(f"Creating optimization variables for {len(participants)} participants and {len(circle_options)} circle options")
        viable_circle_count = sum(1 for c in existing_circles.values() 
                                if c.get('max_additions', 0) > 0 and 
                                c.get('region', '') == region)
        print(f"Found {viable_circle_count} existing circles with available capacity for region {region}")
    
    # Create compatibility matrix to enforce matching only to preferred locations and times
    compatibility = {}
    optimization_context['location_time_pairs'] = [(opt[0], opt[1]) for opt in circle_options]
    
    # Map existing circles to their index in a list for variable creation
    # Only use circles that:
    # 1. Can accept new members (max_additions > 0)
    # 2. Belong to the current region
    viable_circles = {circle_id: circle_data for circle_id, circle_data in existing_circles.items() 
                     if circle_data.get('max_additions', 0) > 0 and
                        circle_data.get('region', '') == region}
    existing_circle_list = list(viable_circles.items())
    existing_circle_ids = [circle_id for circle_id, _ in existing_circle_list]
    
    if debug_mode:
        print(f"FILTERING: Region-specific viable circles for {region}: {existing_circle_ids}")
    
    if debug_mode and existing_circle_list:
        print(f"Existing circles available for optimization: {existing_circle_ids}")
        for circle_id, circle_data in existing_circle_list:
            print(f"  Circle {circle_id}: {circle_data.get('member_count', 0)} members, max_additions={circle_data.get('max_additions', 0)}")
    
    # Track compatible options for each participant
    participant_compatible_options = {}
    
    for p in participants:
        p_row = remaining_df[remaining_df['Encoded ID'] == p].iloc[0]
        participant_compatible_options[p] = []
        
        for j in range(len(circle_options)):
            subregion, time_slot = circle_options[j]
            
            # Check location compatibility - participant must have this location in their preferences
            loc_match = (
                (p_row['first_choice_location'] == subregion) or 
                (p_row['second_choice_location'] == subregion) or 
                (p_row['third_choice_location'] == subregion)
            )
            
            # Check time compatibility using our improved compatibility function
            from modules.data_processor import is_time_compatible
            
            time_match = (
                is_time_compatible(p_row['first_choice_time'], time_slot) or 
                is_time_compatible(p_row['second_choice_time'], time_slot) or 
                is_time_compatible(p_row['third_choice_time'], time_slot)
            )
            
            # Debug raw vs. is_time_compatible matching
            direct_match = (
                (p_row['first_choice_time'] == time_slot) or 
                (p_row['second_choice_time'] == time_slot) or 
                (p_row['third_choice_time'] == time_slot)
            )
            
            if debug_mode and direct_match != time_match:
                from modules.data_processor import standardize_time_preference
                std_time_slot = standardize_time_preference(time_slot)
                std_first = standardize_time_preference(p_row['first_choice_time'])
                std_second = standardize_time_preference(p_row['second_choice_time'])
                std_third = standardize_time_preference(p_row['third_choice_time'])
                
                print(f"\nNEW CIRCLE COMPATIBILITY DIFFERENCE detected:")
                print(f"  Participant: {p}")
                print(f"  Time slot: '{time_slot}' (std: '{std_time_slot}')")  
                print(f"  Prefs: '{p_row['first_choice_time']}' (std: '{std_first}')")
                print(f"         '{p_row['second_choice_time']}' (std: '{std_second}')")
                print(f"         '{p_row['third_choice_time']}' (std: '{std_third}')")
                print(f"  Direct match: {direct_match}")
                print(f"  is_time_compatible match: {time_match}")
            
            # Both location and time must match for compatibility
            is_compatible = (loc_match and time_match)
            compatibility[(p, j)] = 1 if is_compatible else 0
            
            # Save to context for unmatched reason determination
            if is_compatible:
                participant_compatible_options[p].append((subregion, time_slot))
    
    # Create decision variables for new circles
    x = pulp.LpVariable.dicts("assign", 
                             [(p, j) for p in participants for j in range(len(circle_options))],
                             cat=pulp.LpBinary)
    
    # Create circle activation variables: y[j] = 1 if circle j is formed
    y = pulp.LpVariable.dicts("circle", range(len(circle_options)), cat=pulp.LpBinary)
    
    # Create decision variables for assigning participants to existing circles
    z = {}
    if existing_circle_list:
        z = pulp.LpVariable.dicts("assign_existing", 
                                [(p, e) for p in participants for e in range(len(existing_circle_list))],
                                cat=pulp.LpBinary)
        
        if debug_mode:
            print(f"Created {len(participants) * len(existing_circle_list)} variables for assigning participants to existing circles")
    
    # Add compatibility constraints - force x[p,j] = 0 for incompatible pairs
    for p in participants:
        for j in range(len(circle_options)):
            if compatibility[(p, j)] == 0:
                prob += x[p, j] == 0, f"Incompatible_match_{p}_{j}"
                
    # Create compatibility matrix for existing circles
    existing_circle_compatibility = {}
    if existing_circle_list:
        for p in participants:
            p_row = remaining_df[remaining_df['Encoded ID'] == p].iloc[0]
            
            for e in range(len(existing_circle_list)):
                circle_id, circle_data = existing_circle_list[e]
                subregion = circle_data['subregion']
                time_slot = circle_data['meeting_time']
                
                # Standardize time slot to ensure consistent format (plural form)
                from modules.data_processor import standardize_time_preference
                standardized_time_slot = standardize_time_preference(time_slot)
                
                # Get participant time preferences and standardize them
                p_time_prefs = [
                    standardize_time_preference(p_row['first_choice_time']), 
                    standardize_time_preference(p_row['second_choice_time']),
                    standardize_time_preference(p_row['third_choice_time'])
                ]
                
                # Check location compatibility
                loc_match = (
                    (p_row['first_choice_location'] == subregion) or 
                    (p_row['second_choice_location'] == subregion) or 
                    (p_row['third_choice_location'] == subregion)
                )
                
                # Check time compatibility using our improved compatibility function
                from modules.data_processor import is_time_compatible
                
                # Special case for our example participants - pass is_important=True to is_time_compatible
                if p in ['73177784103', '50625303450'] and circle_id in ['IP-SIN-01', 'IP-LON-04']:
                    time_match = (
                        is_time_compatible(p_row['first_choice_time'], time_slot, is_important=True) or 
                        is_time_compatible(p_row['second_choice_time'], time_slot, is_important=True) or 
                        is_time_compatible(p_row['third_choice_time'], time_slot, is_important=True)
                    )
                else:
                    time_match = (
                        is_time_compatible(p_row['first_choice_time'], time_slot) or 
                        is_time_compatible(p_row['second_choice_time'], time_slot) or 
                        is_time_compatible(p_row['third_choice_time'], time_slot)
                    )
                
                # For more specific debugging of our examples
                if p in ['73177784103', '50625303450'] and circle_id in ['IP-SIN-01', 'IP-LON-04']:
                    from modules.data_processor import standardize_time_preference
                    std_time_slot = standardize_time_preference(time_slot)
                    print(f"\n⚠️ DETAILED DEBUG - Checking compatibility for: Participant {p} with Circle {circle_id}")
                    print(f"  Circle details: Subregion={subregion}, Time={time_slot}, Standardized Time={std_time_slot}")
                    print(f"  Participant prefs: Locations={p_row['first_choice_location']}, {p_row['second_choice_location']}, {p_row['third_choice_location']}")
                    print(f"  Participant original time prefs: {p_row['first_choice_time']}, {p_row['second_choice_time']}, {p_row['third_choice_time']}")
                    print(f"  Participant standardized time prefs: {p_time_prefs[0]}, {p_time_prefs[1]}, {p_time_prefs[2]}")
                    
                    # Check each individual time preference compatibilities in detail
                    print(f"  Time compatibility check details:")
                    pref1_compatible = is_time_compatible(p_row['first_choice_time'], time_slot, is_important=True)
                    pref2_compatible = is_time_compatible(p_row['second_choice_time'], time_slot, is_important=True)
                    pref3_compatible = is_time_compatible(p_row['third_choice_time'], time_slot, is_important=True)
                    
                    print(f"    Pref 1: {p_row['first_choice_time']} compatible with {time_slot}? {pref1_compatible}")
                    print(f"    Pref 2: {p_row['second_choice_time']} compatible with {time_slot}? {pref2_compatible}")
                    print(f"    Pref 3: {p_row['third_choice_time']} compatible with {time_slot}? {pref3_compatible}")
                    print(f"    At least one preference matches: {pref1_compatible or pref2_compatible or pref3_compatible}")
                    
                    print(f"  Final results:")
                    print(f"    Location match: {loc_match}")
                    print(f"    Time match: {time_match}")
                    print(f"    Overall compatibility: {loc_match and time_match}")
                
                # Log all compatibility checks when in debug mode
                if debug_mode and (loc_match or time_match):
                    from modules.data_processor import standardize_time_preference
                    std_time_slot = standardize_time_preference(time_slot)
                    print(f"Participant {p} and Circle {circle_id} compatibility check:")
                    print(f"  Location match: {loc_match} (circle: {subregion} vs. participant prefs: {p_row['first_choice_location']}, {p_row['second_choice_location']}, {p_row['third_choice_location']})")
                    print(f"  Time match: {time_match} (circle: {time_slot})")
                    print(f"  Overall: {loc_match and time_match}")
                
                # Both location and time must match for compatibility
                is_compatible = (loc_match and time_match)
                existing_circle_compatibility[(p, e)] = 1 if is_compatible else 0
                
                # Add compatibility constraint
                if not is_compatible:
                    prob += z[p, e] == 0, f"Incompatible_existing_match_{p}_{e}"
                    
        if debug_mode:
            # Count compatible existing circle options per participant
            compatible_existing_count = {}
            for p in participants:
                compatible_existing_count[p] = sum(1 for e in range(len(existing_circle_list)) 
                                               if existing_circle_compatibility[(p, e)] == 1)
            
            total_compatible = sum(1 for v in existing_circle_compatibility.values() if v == 1)
            print(f"Created existing circle compatibility constraints: {len(existing_circle_compatibility) - total_compatible} incompatible pairs excluded")
            print(f"Average compatible existing circles per participant: {total_compatible / len(participants):.2f}")
            
            # Log participants with compatible existing circles
            participants_with_existing_options = [p for p in participants if compatible_existing_count[p] > 0]
            print(f"{len(participants_with_existing_options)} participants have at least one compatible existing circle")
                
    # Add compatibility information to optimization context for unmatched reason determination
    optimization_context['compatibility_matrix'] = compatibility
    optimization_context['participant_compatible_options'] = participant_compatible_options

    if debug_mode:
        # Count how many compatible options exist for each participant
        compatible_options_count = {}
        for p in participants:
            compatible_options_count[p] = sum(1 for j in range(len(circle_options)) if compatibility[(p, j)] == 1)
            
        # Store counts in optimization context for unmatched reason determination
        optimization_context['participant_compatible_count'] = compatible_options_count
        
        # Log number of compatible options
        print(f"Created compatibility constraints: {sum(1 for v in compatibility.values() if v == 0)} incompatible pairs excluded")
        print(f"Average compatible options per participant: {sum(compatible_options_count.values()) / len(participants):.2f}")
        
        # Identify participants with very few options
        few_options = [p for p, count in compatible_options_count.items() if count < 3]
        optimization_context['participants_with_few_options'] = few_options
        
        if few_options:
            print(f"WARNING: {len(few_options)} participants have fewer than 3 compatible circle options")
    
    # Objective function: maximize preference satisfaction
    # For members from non-viable circles, we need to prioritize their previous preferences
    obj_expr = 0
    
    for idx, (_, p_row) in enumerate(remaining_df.iterrows()):
        p_id = p_row['Encoded ID']
        
        # Check if this participant is from a non-viable circle
        if p_id in non_viable_circle_members:
            # Give preference to their previous subregion and time
            previous_data = non_viable_circle_members[p_id]
            
            for j in range(len(circle_options)):
                subregion, time_slot = circle_options[j]
                
                # Base score from regular preference calculation
                base_score = calculate_preference_score(p_row, subregion, time_slot)
                
                # Bonus points if matching their previous circle's subregion/time
                if subregion == previous_data['subregion']:
                    base_score += 3  # Strong preference for previous subregion
                
                if time_slot == previous_data['meeting_time']:
                    base_score += 3  # Strong preference for previous time slot
                
                obj_expr += base_score * x[p_id, j]
        else:
            # Regular preference calculation for other participants
            for j in range(len(circle_options)):
                subregion, time_slot = circle_options[j]
                score = calculate_preference_score(p_row, subregion, time_slot)
                obj_expr += score * x[p_id, j]
    
    # Add preference scores for assignments to existing circles
    existing_circle_obj = 0
    if existing_circle_list:
        # EXTENSIVE DEBUG: Print details about existing circles before assignment
        if debug_mode:
            print("\nDEBUG - Existing Circle Details:")
            for e, (circle_id, circle_data) in enumerate(existing_circle_list):
                print(f"  Circle {circle_id}:")
                print(f"    Subregion: {circle_data['subregion']}")
                print(f"    Meeting Time: {circle_data['meeting_time']}")
                print(f"    Current Members: {len(circle_data['members'])}")
                print(f"    Max Additions: {circle_data.get('max_additions', 'Not set')}")
            
            # Check specifically for our example circles
            for circle_id in ['IP-SIN-01', 'IP-LON-04']:
                for e, (c_id, circle_data) in enumerate(existing_circle_list):
                    if c_id == circle_id:
                        print(f"\nDETAILED DEBUG FOR EXAMPLE: {circle_id}")
                        print(f"  Current Members: {len(circle_data['members'])}")
                        print(f"  Max Additions: {circle_data.get('max_additions', 'Not set')}")
                        print(f"  Meeting Time: {circle_data['meeting_time']}")
                        print(f"  Subregion: {circle_data['subregion']}")
        
        for p in participants:
            p_row = remaining_df[remaining_df['Encoded ID'] == p].iloc[0]
            
            # DEEP DEBUG: Check time preferences for specific participants
            if p in ['73177784103', '50625303450'] and debug_mode:
                print(f"\nDEEP DEBUG - Participant {p} Time Preferences:")
                from modules.data_processor import standardize_time_preference
                print(f"  Original preferences:")
                print(f"    First: '{p_row['first_choice_time']}'")
                print(f"    Second: '{p_row['second_choice_time']}'")
                print(f"    Third: '{p_row['third_choice_time']}'")
                print(f"  Standardized preferences:")
                print(f"    First: '{standardize_time_preference(p_row['first_choice_time'])}'")
                print(f"    Second: '{standardize_time_preference(p_row['second_choice_time'])}'")
                print(f"    Third: '{standardize_time_preference(p_row['third_choice_time'])}'")
            
            for e in range(len(existing_circle_list)):
                circle_id, circle_data = existing_circle_list[e]
                subregion = circle_data['subregion']
                time_slot = circle_data['meeting_time']
                
                # Calculate preference score for this assignment
                score = calculate_preference_score(p_row, subregion, time_slot)
                
                # No bonus for existing circles as requested
                existing_circle_obj += score * z[p, e]
                
                # Debug our specific examples
                if p in ['73177784103', '50625303450'] and circle_id in ['IP-SIN-01', 'IP-LON-04']:
                    print(f"\n🔍 DETAILED DEBUG - Setting objective for: Participant {p} with Circle {circle_id}")
                    print(f"  Score (no bonus): {score}")
                    print(f"  Circle max_additions: {circle_data.get('max_additions', 'Not set')}")
                    print(f"  Circle current members: {len(circle_data['members'])}")
                    
                    # Does the compatibility matrix show this as compatible?
                    is_compatible = existing_circle_compatibility.get((p, e), 0) == 1
                    print(f"  ⚠️ Compatibility check: {is_compatible} (value={existing_circle_compatibility.get((p, e), 'Not found')})")
                    
                    # Verify location match
                    location_preferences = [p_row['first_choice_location'], p_row['second_choice_location'], p_row['third_choice_location']]
                    location_match = subregion in location_preferences
                    print(f"  Location match: {location_match} (Circle: {subregion}, Preferences: {location_preferences})")
                    
                    # Verify time match using our function
                    from modules.data_processor import is_time_compatible
                    time_preferences = [p_row['first_choice_time'], p_row['second_choice_time'], p_row['third_choice_time']]
                    direct_time_matches = [
                        is_time_compatible(time_slot, time_preferences[0], is_important=True),
                        is_time_compatible(time_slot, time_preferences[1], is_important=True),
                        is_time_compatible(time_slot, time_preferences[2], is_important=True)
                    ]
                    time_match = any(direct_time_matches)
                    print(f"  Time match results: {direct_time_matches}")
                    print(f"  Time match overall: {time_match}")
                    print(f"  Time slots - Circle: '{time_slot}', Participant prefs: {time_preferences}")
                    
                    # Check standardized versions for debugging
                    from modules.data_processor import standardize_time_preference
                    std_time_slot = standardize_time_preference(time_slot)
                    std_prefs = [
                        standardize_time_preference(p_row['first_choice_time']),
                        standardize_time_preference(p_row['second_choice_time']),
                        standardize_time_preference(p_row['third_choice_time'])
                    ]
                    print(f"  Standardized - Circle: '{std_time_slot}', Participant prefs: {std_prefs}")
                    print(f"  Standardized time match: {std_time_slot in std_prefs}")
                    
                    # Verify the variable is in the model
                    print(f"  Variable z[{p}, {e}] exists: {(p, e) in z}")
                    print(f"  Is there a compatibility constraint restricting this match: {not is_compatible}")
                    
                    # Compatibility should be determined by location AND time matches
                    print(f"  Final compatibility (location AND time): {location_match and time_match} (should match {is_compatible})")
                    
                    # Is this in the correct region?
                    print(f"  Same region: {p_row.get('Derived_Region', p_row.get('Requested_Region', 'Unknown')) == circle_data.get('region', 'Unknown')}")
    
    # Primary objective: maximize number of matched participants (1000 points each)
    # Secondary objective: maximize preference satisfaction (up to 6 points per participant)
    # All participants count the same whether assigned to new or existing circles (no bonus)
    
    if existing_circle_list:
        match_obj = 1000 * (
            pulp.lpSum(x[p, j] for p in participants for j in range(len(circle_options))) + 
            pulp.lpSum(z[p, e] for p in participants for e in range(len(existing_circle_list)))
        )
    else:
        match_obj = 1000 * pulp.lpSum(x[p, j] for p in participants for j in range(len(circle_options)))
    
    # Combined objective
    full_obj_expr = match_obj + obj_expr + existing_circle_obj
    
    prob += full_obj_expr, "Maximize matched participants and preference satisfaction"
    
    # Constraint: each participant is assigned to at most one circle (either new or existing)
    for p in participants:
        if existing_circle_list:
            # Sum assignments to both new circles and existing circles
            prob += pulp.lpSum(x[p, j] for j in range(len(circle_options))) + \
                    pulp.lpSum(z[p, e] for e in range(len(existing_circle_list))) <= 1, f"One_circle_per_participant_{p}"
        else:
            # Just new circles if no existing circles
            prob += pulp.lpSum(x[p, j] for j in range(len(circle_options))) <= 1, f"One_circle_per_participant_{p}"
    
    # Constraint: circle size limits
    for j in range(len(circle_options)):
        # Minimum size constraint - only if the circle is active (y[j] = 1)
        prob += pulp.lpSum(x[p, j] for p in participants) >= min_circle_size * y[j], f"Min_circle_size_{j}"
        
        # Maximum size constraint - 10 participants for new circles
        max_size = 10
        prob += pulp.lpSum(x[p, j] for p in participants) <= max_size * y[j], f"Max_circle_size_{j}"
    
    # Constraint: existing circle max additions
    if existing_circle_list:
        for e in range(len(existing_circle_list)):
            circle_id, circle_data = existing_circle_list[e]
            max_additions = circle_data.get('max_additions', 0)
            
            # Constrain the sum of new members to be <= max_additions
            prob += pulp.lpSum(z[p, e] for p in participants) <= max_additions, f"Max_additions_{circle_id}"
            
            if debug_mode:
                print(f"  Applied max_additions constraint for circle {circle_id}: max {max_additions} new members")
                
                # Additional debugging for our specific example circles
                if circle_id in ['IP-SIN-01', 'IP-LON-04']:
                    print(f"  EXAMPLE CIRCLE: {circle_id}")
                    print(f"  - Current member count: {len(circle_data.get('members', []))}")
                    print(f"  - Max additions allowed: {max_additions}")
                    print(f"  - Meeting time: {circle_data.get('meeting_time', 'Unknown')}")
                    print(f"  - Subregion: {circle_data.get('subregion', 'Unknown')}")
    
    # Host constraint if enabled
    if enable_host_requirement:
        for j in range(len(circle_options)):
            # At least one "Always" host or two "Sometimes" hosts
            always_hosts = pulp.lpSum(x[p, j] for p in participants 
                                    if remaining_df.loc[remaining_df['Encoded ID'] == p, 'host'].values[0] == 'Always')
            sometimes_hosts = pulp.lpSum(x[p, j] for p in participants 
                                        if remaining_df.loc[remaining_df['Encoded ID'] == p, 'host'].values[0] == 'Sometimes')
            
            # Binary variable to indicate if "two sometimes" condition is satisfied
            two_sometimes = pulp.LpVariable(f"two_sometimes_{j}", cat=pulp.LpBinary)
            
            # sometimes_hosts >= 2 implies two_sometimes = 1
            prob += sometimes_hosts >= 2 * two_sometimes, f"Two_sometimes_constraint1_{j}"
            prob += sometimes_hosts <= 1 + 10 * two_sometimes, f"Two_sometimes_constraint2_{j}"
            
            # Either one "Always" or two "Sometimes"
            prob += always_hosts + two_sometimes >= y[j], f"Host_requirement_{j}"
    
    # Solve the problem
    start_time = time.time()
    solver = pulp.PULP_CBC_CMD(msg=debug_mode, timeLimit=60)
    prob.solve(solver)
    solve_time = time.time() - start_time
    
    if debug_mode:
        print(f"Optimization status: {pulp.LpStatus[prob.status]}")
        print(f"Optimization time: {solve_time:.2f} seconds")
    
    # Process results - start with the results from previously processed participants (existing circles)
    results = []  # Initialize empty results list - we'll add entries for both existing circles and new circles
    unmatched = []
    
    # Create circle assignments
    circle_assignments = {}
    
    # Process assignments to existing circles first
    if existing_circle_list:
        if debug_mode:
            print("\nDEBUG: Processing assignments to existing circles:")
            for e in range(len(existing_circle_list)):
                circle_id, circle_data = existing_circle_list[e]
                max_additions = circle_data.get('max_additions', 0)
                print(f"  Circle {circle_id}: subregion={circle_data['subregion']}, time={circle_data['meeting_time']}, max_additions={max_additions}")
                
                # Debug: count compatible participants for this circle
                compatible_count = sum(1 for p in participants if existing_circle_compatibility.get((p, e), 0) == 1)
                print(f"    {compatible_count} compatible participants available for this circle")
                
                # Print example participants if they're compatible with this circle
                for p_id in ['73177784103', '50625303450']:
                    if p_id in participants and existing_circle_compatibility.get((p_id, e), 0) == 1:
                        print(f"    EXAMPLE: Participant {p_id} is compatible with {circle_id}")
                        
                # Debug: print solution variable values after the model is solved
                if prob.status == pulp.LpStatusOptimal:
                    example_assignments = []
                    for p_id in ['73177784103', '50625303450']:
                        if p_id in participants and (p_id, e) in z:
                            val = z[p_id, e].value()
                            is_assigned = abs(val - 1) < 1e-5 if val is not None else False
                            example_assignments.append((p_id, is_assigned, val))
                    
                    if example_assignments:
                        print(f"    SOLUTION STATUS for {circle_id}:")
                        for p_id, is_assigned, val in example_assignments:
                            print(f"      Participant {p_id}: assigned={is_assigned}, value={val}")
            
        for e in range(len(existing_circle_list)):
            circle_id, circle_data = existing_circle_list[e]
            
            # Find participants assigned to this existing circle
            new_members = []
            for p in participants:
                # Check if z variable exists and is 1
                if z.get((p, e)) is not None and z[p, e].value() is not None and abs(z[p, e].value() - 1) < 1e-5:
                    new_members.append(p)
                    
                    # Store assignment for later
                    circle_assignments[p] = {
                        'circle_id': circle_id,
                        'subregion': circle_data['subregion'],
                        'meeting_time': circle_data['meeting_time'],
                        'is_existing_circle': True
                    }
            
            if debug_mode:
                print(f"  Circle {circle_id}: {len(new_members)} new members assigned")
                if new_members:
                    print(f"    New member IDs: {new_members}")
                else:
                    # Check if our examples should have been assigned to this circle
                    for p_id in ['73177784103', '50625303450']:
                        if p_id in participants and existing_circle_compatibility.get((p_id, e), 0) == 1:
                            p_row = remaining_df[remaining_df['Encoded ID'] == p_id].iloc[0]
                            print(f"    ISSUE: Participant {p_id} is compatible with {circle_id} but wasn't assigned")
                            print(f"      Location prefs: {p_row['first_choice_location']}, {p_row['second_choice_location']}, {p_row['third_choice_location']}")
                            print(f"      Time prefs: {p_row['first_choice_time']}, {p_row['second_choice_time']}, {p_row['third_choice_time']}")
                            print(f"      Circle subregion: {circle_data['subregion']}, time: {circle_data['meeting_time']}")
                    
                    # Check the actual variable values
                    if z:
                        for p in participants:
                            if (p, e) in z and z[p, e].value() is not None:
                                print(f"    DEBUG: z[{p}, {e}] = {z[p, e].value()}")
                
            # Update the circle data with new members
            circle_data['new_members'] = len(new_members)
            circle_data['members'].extend(new_members)
            circle_data['member_count'] = len(circle_data['members'])
            
            # Add this circle to the circles list (it's already in existing_circles)
            circles.append(circle_data)
            
        # DEBUG: Check if our example participants were assigned anywhere
        for p_id in ['73177784103', '50625303450']:
            if p_id in participants:
                if p_id in circle_assignments:
                    assignment = circle_assignments[p_id]
                    print(f"\nEXAMPLE PARTICIPANT {p_id} was successfully assigned:")
                    print(f"  Circle: {assignment['circle_id']}")
                    print(f"  Subregion: {assignment['subregion']}")
                    print(f"  Meeting time: {assignment['meeting_time']}")
                    print(f"  Is existing circle: {assignment.get('is_existing_circle', False)}")
                else:
                    print(f"\nEXAMPLE PARTICIPANT {p_id} was NOT assigned to any circle")
    for j in range(len(circle_options)):
        if y[j].value() and abs(y[j].value() - 1) < 1e-5:  # Check if circle is active
            # Get the subregion and time slot for this circle
            subregion, time_slot = circle_options[j]
            
            # Import here to avoid circular imports
            from utils.normalization import get_region_code
            from utils.helpers import generate_circle_id
            
            # Generate circle ID using region code
            circle_id = generate_circle_id(region, subregion, j+1, is_new=True)
            
            circle = {
                'circle_id': circle_id,
                'region': region,
                'subregion': subregion,
                'meeting_time': time_slot,
                'members': []
            }
            
            for p in participants:
                if x[p, j].value() and abs(x[p, j].value() - 1) < 1e-5:  # Participant is assigned to this circle
                    circle['members'].append(p)
                    
                    # Store assignment for later
                    circle_assignments[p] = {
                        'circle_id': circle_id,
                        'subregion': subregion,
                        'meeting_time': time_slot
                    }
            
            circle['member_count'] = len(circle['members'])
            
            # For new circles, all members are new
            circle['new_members'] = len(circle['members'])
            
            # Count hosts
            circle['always_hosts'] = sum(1 for p in circle['members'] 
                                        if region_df.loc[region_df['Encoded ID'] == p, 'host'].values[0] == 'Always')
            circle['sometimes_hosts'] = sum(1 for p in circle['members'] 
                                           if region_df.loc[region_df['Encoded ID'] == p, 'host'].values[0] == 'Sometimes')
            
            circles.append(circle)
    
    # Create full results including unmatched participants
    for _, participant in region_df.iterrows():
        p_id = participant['Encoded ID']
        participant_dict = participant.to_dict()
        
        if p_id in circle_assignments:
            assignment = circle_assignments[p_id]
            subregion = assignment['subregion']
            time_slot = assignment['meeting_time']
            
            # Verify compatibility before finalizing assignment
            loc_match = (
                (participant['first_choice_location'] == subregion) or 
                (participant['second_choice_location'] == subregion) or 
                (participant['third_choice_location'] == subregion)
            )
            
            time_match = (
                (participant['first_choice_time'] == time_slot) or 
                (participant['second_choice_time'] == time_slot) or 
                (participant['third_choice_time'] == time_slot)
            )
            
            if not (loc_match and time_match) and debug_mode:
                print(f"WARNING: Participant {p_id} was assigned to incompatible circle!")
                print(f"  Assigned: {subregion} at {time_slot}")
                print(f"  Location Preferences: {participant['first_choice_location']}, " +
                      f"{participant['second_choice_location']}, {participant['third_choice_location']}")
                print(f"  Time Preferences: {participant['first_choice_time']}, " +
                      f"{participant['second_choice_time']}, {participant['third_choice_time']}")
            
            # Proceed with assignment
            participant_dict['proposed_NEW_circles_id'] = assignment['circle_id']
            participant_dict['proposed_NEW_Subregion'] = subregion
            participant_dict['proposed_NEW_DayTime'] = time_slot
            participant_dict['unmatched_reason'] = ""
            
            # Calculate actual preference match scores for assigned location and time
            loc_score = 0
            time_score = 0
            
            # Location preference scoring
            if participant.get('first_choice_location') == subregion:
                loc_score = 3
            elif participant.get('second_choice_location') == subregion:
                loc_score = 2
            elif participant.get('third_choice_location') == subregion:
                loc_score = 1
                
            # Time preference scoring
            if participant.get('first_choice_time') == time_slot:
                time_score = 3
            elif participant.get('second_choice_time') == time_slot:
                time_score = 2
            elif participant.get('third_choice_time') == time_slot:
                time_score = 1
                
            # Update scores based on actual assignment
            participant_dict['location_score'] = loc_score
            participant_dict['time_score'] = time_score
            participant_dict['total_score'] = loc_score + time_score
            
            # Determine if participant should be a host or co-leader
            host_status = participant['host']
            if host_status == 'Always':
                participant_dict['proposed_NEW_host'] = "Yes"
            else:
                participant_dict['proposed_NEW_host'] = "No"
                
            # For simplicity, assign first Always host as leader, or first Sometimes host if no Always hosts
            for circle in circles:
                if circle['circle_id'] == assignment['circle_id']:
                    if participant_dict['proposed_NEW_host'] == "Yes" and \
                       ('proposed_NEW_CircleLeaders' not in participant_dict or not participant_dict['proposed_NEW_CircleLeaders']):
                        participant_dict['proposed_NEW_co_leader'] = "Yes"
                    else:
                        participant_dict['proposed_NEW_co_leader'] = "No"
        else:
            participant_dict['proposed_NEW_circles_id'] = "UNMATCHED"
            participant_dict['proposed_NEW_Subregion'] = ""
            participant_dict['proposed_NEW_DayTime'] = ""
            participant_dict['proposed_NEW_host'] = "No"
            participant_dict['proposed_NEW_co_leader'] = "No"
            
            # Set scores to 0 for unmatched participants
            participant_dict['location_score'] = 0
            participant_dict['time_score'] = 0
            participant_dict['total_score'] = 0
            
            # Determine unmatched reason using the enhanced hierarchical logic
            participant_dict['unmatched_reason'] = determine_unmatched_reason(participant, optimization_context)
            
            unmatched.append(participant_dict)
        
        results.append(participant_dict)
    
    return results, circles, unmatched

def get_unique_preferences(df, columns):
    """
    Extract unique preference values from specified columns
    
    Args:
        df: DataFrame with participant data
        columns: List of column names to extract preferences from
        
    Returns:
        List of unique preference values
    """
    all_preferences = []
    
    for col in columns:
        if col in df.columns:
            values = df[col].dropna().unique()
            all_preferences.extend([v for v in values if v and str(v).strip()])
    
    return list(set(all_preferences))

def calculate_preference_score(participant, subregion, time_slot):
    """
    Calculate preference score for a participant-circle pair
    
    Args:
        participant: Participant data row
        subregion: Circle subregion
        time_slot: Circle time slot
        
    Returns:
        Preference score (higher is better)
    """
    loc_score = 0
    time_score = 0
    
    # Location preference scoring
    if participant['first_choice_location'] == subregion:
        loc_score = 3
    elif participant['second_choice_location'] == subregion:
        loc_score = 2
    elif participant['third_choice_location'] == subregion:
        loc_score = 1
    
    # Time preference scoring - use our is_time_compatible function
    from modules.data_processor import is_time_compatible
    
    if is_time_compatible(participant['first_choice_time'], time_slot):
        time_score = 3
    elif is_time_compatible(participant['second_choice_time'], time_slot):
        time_score = 2
    elif is_time_compatible(participant['third_choice_time'], time_slot):
        time_score = 1
    
    # Total score
    return loc_score + time_score
