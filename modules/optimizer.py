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
    
    # Group participants by region and determine subregions for each region
    regions = df['Requested_Region'].unique()
    
    # Initialize results containers
    all_results = []
    all_circles = []
    all_unmatched = []
    
    # Process each region separately
    for region in regions:
        if debug_mode:
            print(f"Processing region: {region}")
        
        region_df = df[df['Requested_Region'] == region]
        
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
    
    # Check if we need to handle existing circles
    existing_circles = {}
    current_circle_members = {}
    
    # Step 1: Identify existing circles if we're preserving them
    if existing_circle_handling == 'preserve' and 'current_circles_id' in region_df.columns:
        # Group participants by their current circle
        for _, row in region_df.iterrows():
            # First, check if it's a CURRENT-CONTINUING participant
            if row.get('Status') == 'CURRENT-CONTINUING':
                # If they have a valid circle ID, add them to that circle
                if pd.notna(row.get('current_circles_id')):
                    circle_id = str(row['current_circles_id']).strip()
                    if circle_id:
                        if circle_id not in current_circle_members:
                            current_circle_members[circle_id] = []
                        current_circle_members[circle_id].append(row)
                    else:
                        # They're CURRENT-CONTINUING but have an empty circle ID
                        # These need to be assigned to new circles
                        if debug_mode:
                            print(f"CURRENT-CONTINUING participant {row['Encoded ID']} has empty circle ID")
        
        # Evaluate each existing circle
        for circle_id, members in current_circle_members.items():
            # Check if the circle has enough members to continue
            if len(members) >= min_circle_size:
                # Check if there's at least one host
                has_host = any(m.get('host', '').lower() in ['always', 'always host', 'sometimes', 'sometimes host'] for m in members)
                
                if has_host or not enable_host_requirement:
                    # This circle can continue - get subregion and time if available
                    subregion = members[0].get('Current_Subregion', '')
                    meeting_time = members[0].get('Current_Meeting_Time', '')
                    
                    # Add to existing circles dict with member list and metadata
                    existing_circles[circle_id] = {
                        'members': [m['Encoded ID'] for m in members],
                        'subregion': subregion,
                        'meeting_time': meeting_time,
                        'always_hosts': sum(1 for m in members if m.get('host', '').lower() in ['always', 'always host']),
                        'sometimes_hosts': sum(1 for m in members if m.get('host', '').lower() in ['sometimes', 'sometimes host'])
                    }
                    
                    if debug_mode:
                        print(f"Preserving existing circle {circle_id} with {len(members)} members")
    
    # Step 2: Process participants in existing circles
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
    obj_expr = pulp.lpSum(calculate_preference_score(p_row, circle_options[j][0], circle_options[j][1]) * x[p, j]
                         for idx, (_, p_row) in enumerate(remaining_df.iterrows())
                         for j in range(len(circle_options))
                         for p in [p_row['Encoded ID']])
    
    prob += obj_expr, "Maximize preference satisfaction"
    
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
