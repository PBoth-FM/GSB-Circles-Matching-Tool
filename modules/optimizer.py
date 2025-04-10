import pandas as pd
import numpy as np
import pulp
import time
from itertools import combinations

def run_matching_algorithm(data, config):
    """
    Run the optimization algorithm to match participants into circles
    
    Args:
        data: DataFrame with processed participant data
        config: Dictionary with configuration parameters
        
    Returns:
        Tuple of (results DataFrame, matched_circles DataFrame, unmatched_participants DataFrame)
    """
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
                current_meeting_day = first_member.get('Current_Meeting_Day', '')
                current_meeting_time = first_member.get('Current_Meeting_Time', '')
                
                # Format the day/time combination for proposed_NEW_DayTime
                if current_meeting_day and current_meeting_time:
                    formatted_day_time = f"{current_meeting_day} ({current_meeting_time})"
                else:
                    formatted_day_time = current_meeting_day or current_meeting_time
                
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
                    'is_direct_continuation': True
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
                participant_dict['unmatched_reason'] = "Insufficient participants in region"
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
                    meeting_day = members[0].get('Current_Meeting_Day', '')
                    meeting_time = members[0].get('Current_Meeting_Time', '')
                    
                    # Format the day/time combination for proposed_NEW_DayTime
                    if meeting_day and meeting_time:
                        formatted_meeting_time = f"{meeting_day} ({meeting_time})"
                    else:
                        formatted_meeting_time = meeting_day or meeting_time
                    
                    # Create circle data with member list and metadata
                    circle_data = {
                        'members': [m['Encoded ID'] for m in members],
                        'subregion': subregion,
                        'meeting_time': formatted_meeting_time,
                        'always_hosts': sum(1 for m in members if m.get('host', '').lower() in ['always', 'always host']),
                        'sometimes_hosts': sum(1 for m in members if m.get('host', '').lower() in ['sometimes', 'sometimes host']),
                        'is_in_person': is_in_person,
                        'is_virtual': is_virtual
                    }
                    
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
            
            # Handle host status
            if participant.get('host', '').lower() in ['always', 'always host']:
                participant['proposed_NEW_host'] = "Yes"
            elif participant.get('host', '').lower() in ['sometimes', 'sometimes host']:
                participant['proposed_NEW_host'] = "Maybe"
            else:
                participant['proposed_NEW_host'] = "No"
            
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
            'members': circle_data['members']
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
        needed_size = min_circle_size - current_size
        
        if debug_mode:
            print(f"Small circle {circle_id} has {current_size} members, needs {needed_size} more")
        
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
                
                # Handle host status
                if participant.get('host', '').lower() in ['always', 'always host']:
                    participant['proposed_NEW_host'] = "Yes"
                elif participant.get('host', '').lower() in ['sometimes', 'sometimes host']:
                    participant['proposed_NEW_host'] = "Maybe"
                else:
                    participant['proposed_NEW_host'] = "No"
                
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
            'members': circle_data['members']
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
    
    if debug_mode:
        print(f"Region: {region}, Subregions: {subregions}, Time slots: {time_slots}")
    
    # Handle case where no preferences exist
    if not subregions or not time_slots:
        results = []
        unmatched = []
        
        for _, participant in region_df.iterrows():
            participant_dict = participant.to_dict()
            participant_dict['proposed_NEW_circles_id'] = "UNMATCHED"
            
            if not subregions:
                participant_dict['unmatched_reason'] = "No location preferences"
            elif not time_slots:
                participant_dict['unmatched_reason'] = "No time preferences"
            else:
                participant_dict['unmatched_reason'] = "No preferences"
                
            results.append(participant_dict)
            unmatched.append(participant_dict)
            
        return results, [], unmatched
    
    # Generate all possible circle combinations (subregion + time slot)
    circle_options = [(subregion, time_slot) for subregion in subregions for time_slot in time_slots]
    
    # Set up the optimization problem
    prob = pulp.LpProblem(f"CircleMatching_{region}", pulp.LpMaximize)
    
    # Create decision variables: x[i, j] = 1 if participant i is assigned to circle j
    # Must use remaining_df here, not region_df to avoid including already processed participants
    participants = remaining_df['Encoded ID'].tolist()
    
    if debug_mode:
        print(f"Creating optimization variables for {len(participants)} participants and {len(circle_options)} circle options")
    
    x = pulp.LpVariable.dicts("assign", 
                             [(p, j) for p in participants for j in range(len(circle_options))],
                             cat=pulp.LpBinary)
    
    # Create circle activation variables: y[j] = 1 if circle j is formed
    y = pulp.LpVariable.dicts("circle", range(len(circle_options)), cat=pulp.LpBinary)
    
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
    
    # Primary objective: maximize number of matched participants (1000 points each)
    # Secondary objective: maximize preference satisfaction (up to 6 points per participant)
    match_obj = 1000 * pulp.lpSum(x[p, j] for p in participants for j in range(len(circle_options)))
    
    # Combined objective
    full_obj_expr = match_obj + obj_expr
    
    prob += full_obj_expr, "Maximize matched participants and preference satisfaction"
    
    # Constraint: each participant is assigned to at most one circle
    for p in participants:
        prob += pulp.lpSum(x[p, j] for j in range(len(circle_options))) <= 1, f"One_circle_per_participant_{p}"
    
    # Constraint: circle size limits
    for j in range(len(circle_options)):
        # Minimum size constraint - only if the circle is active (y[j] = 1)
        prob += pulp.lpSum(x[p, j] for p in participants) >= min_circle_size * y[j], f"Min_circle_size_{j}"
        
        # Maximum size constraint - 10 participants
        max_size = 10
        prob += pulp.lpSum(x[p, j] for p in participants) <= max_size * y[j], f"Max_circle_size_{j}"
    
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
            participant_dict['proposed_NEW_circles_id'] = assignment['circle_id']
            participant_dict['proposed_NEW_Subregion'] = assignment['subregion']
            participant_dict['proposed_NEW_DayTime'] = assignment['meeting_time']
            participant_dict['unmatched_reason'] = ""
            
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
            
            # Determine unmatched reason
            if not participant['first_choice_location'] and not participant['second_choice_location'] and not participant['third_choice_location']:
                participant_dict['unmatched_reason'] = "No location preference"
            elif not participant['first_choice_time'] and not participant['second_choice_time'] and not participant['third_choice_time']:
                participant_dict['unmatched_reason'] = "No time preference"
            else:
                # Check location compatibility
                loc_compatible = False
                for loc in [participant['first_choice_location'], participant['second_choice_location'], participant['third_choice_location']]:
                    if loc in subregions:
                        loc_compatible = True
                        break
                
                # Check time compatibility
                time_compatible = False
                for time_pref in [participant['first_choice_time'], participant['second_choice_time'], participant['third_choice_time']]:
                    if time_pref in time_slots:
                        time_compatible = True
                        break
                
                if not loc_compatible:
                    participant_dict['unmatched_reason'] = "Location issues"
                elif not time_compatible:
                    participant_dict['unmatched_reason'] = "Time compatibility"
                else:
                    participant_dict['unmatched_reason'] = "Optimization trade-off"
            
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
    
    # Time preference scoring
    if participant['first_choice_time'] == time_slot:
        time_score = 3
    elif participant['second_choice_time'] == time_slot:
        time_score = 2
    elif participant['third_choice_time'] == time_slot:
        time_score = 1
    
    # Total score
    return loc_score + time_score
