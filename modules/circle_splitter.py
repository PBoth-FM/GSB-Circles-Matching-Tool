"""
Circle splitter module that identifies large circles (11+ members) and splits them
into smaller circles (at least 5 members each).
"""

import pandas as pd
import numpy as np
import logging
import random
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_large_circles(circles_data, participants_data):
    """
    Identifies circles with 11+ members and splits them into smaller circles.
    
    Args:
        circles_data: DataFrame or list of dictionaries containing circle information
        participants_data: DataFrame containing participant information
        
    Returns:
        tuple: (
            updated_circles: DataFrame or list with split circles replacing large ones,
            split_summary: Dictionary containing statistics and details about the splitting process
        )
    """
    print("=========================================")
    print("🔴 CIRCLE SPLITTER FUNCTION ENTRY POINT REACHED")
    print("=========================================")
    logger.info("Starting circle splitting process")
    print("\n🔴 CRITICAL DEBUG: CIRCLE SPLITTER: Processing circles to identify those with 11+ members")
    
    # Initialize statistics
    split_summary = {
        "total_circles_examined": 0,
        "total_large_circles_found": 0,
        "total_circles_successfully_split": 0,
        "total_new_circles_created": 0,
        "split_details": [],
        "status": "success"
    }
    
    # Standardize input format - convert to DataFrame if it's a list
    if isinstance(circles_data, list):
        # Create DataFrame from list of dictionaries
        print("🔴 CIRCLE SPLITTER: Converting list of dictionaries to DataFrame")
        circles_df = pd.DataFrame(circles_data)
    else:
        # Already a DataFrame, make a copy to avoid modifying the original
        circles_df = circles_data.copy()
    
    # Ensure participant_data is a DataFrame
    if not isinstance(participants_data, pd.DataFrame):
        print("⚠️ CIRCLE SPLITTER: participants_data must be a DataFrame")
        split_summary["status"] = "error"
        split_summary["error_message"] = "participants_data must be a DataFrame"
        return circles_data, split_summary
    
    print(f"🔴 CIRCLE SPLITTER: Processing {len(circles_df)} circles")
    split_summary["total_circles_examined"] = len(circles_df)
    
    # Find circles with 11+ members
    large_circles = []
    for _, circle in circles_df.iterrows():
        circle_id = circle.get('circle_id', None)
        
        # Check for member_count field or calculate from members list
        if 'member_count' in circle:
            member_count = circle['member_count']
        elif 'members' in circle and isinstance(circle['members'], list):
            member_count = len(circle['members'])
        else:
            member_count = 0
            
        # Skip circles without an ID or with fewer than 11 members
        if not circle_id or member_count < 11:
            continue
            
        print(f"🔴 CIRCLE SPLITTER: Found large circle {circle_id} with {member_count} members")
        large_circles.append(circle_id)
    
    # Update statistics
    split_summary["total_large_circles_found"] = len(large_circles)
    print(f"🔴 CIRCLE SPLITTER: Found {len(large_circles)} large circles to split")
    
    if not large_circles:
        print("🔴 CIRCLE SPLITTER: No large circles found that need splitting")
        return circles_df, split_summary
    
    # For each large circle, split into smaller circles
    new_circles = []
    circles_to_remove = []
    
    for large_circle_id in large_circles:
        print(f"\n🔴 CIRCLE SPLITTER: Processing large circle {large_circle_id}")
        
        # Get the circle data
        circle_row = circles_df[circles_df['circle_id'] == large_circle_id].iloc[0]
        
        # Extract circle metadata
        circle_members = []
        if 'members' in circle_row and isinstance(circle_row['members'], list):
            circle_members = circle_row['members']
        else:
            # Try to find members from participants data
            print(f"🔴 CIRCLE SPLITTER: Finding members of {large_circle_id} from participants data")
            
            # Determine which column contains circle assignments
            circle_id_col = None
            possible_columns = ['Circle', 'circle_id', 'current_circle_id', 'Current_Circle_ID']
            
            for col in possible_columns:
                if col in participants_data.columns:
                    circle_id_col = col
                    break
            
            if circle_id_col:
                # Get all participants assigned to this circle
                circle_members = participants_data[participants_data[circle_id_col] == large_circle_id]['Encoded ID'].tolist()
                print(f"🔴 CIRCLE SPLITTER: Found {len(circle_members)} members in {large_circle_id} from participants data")
            else:
                print(f"⚠️ CIRCLE SPLITTER: Could not find circle ID column in participants data")
                continue
        
        # Only proceed if we found members
        if not circle_members:
            print(f"⚠️ CIRCLE SPLITTER: No members found for circle {large_circle_id}")
            continue
        
        # Extract circle properties
        circle_region = circle_row.get('region', '')
        circle_subregion = circle_row.get('subregion', '')
        meeting_time = circle_row.get('meeting_time', '')
        is_in_person = large_circle_id.startswith('IP-')
        format_prefix = 'IP-' if is_in_person else 'V-'
        
        # Get circle number (the last part after the last dash)
        circle_parts = large_circle_id.split('-')
        if len(circle_parts) > 1:
            circle_number = circle_parts[-1]
        else:
            circle_number = '00'  # Default if we can't extract
        
        # Analyze members to identify hosts and co-leaders
        member_roles = get_member_roles(participants_data, circle_members)
        
        # Split circle into smaller ones with balanced host distribution
        split_result = split_circle_with_balanced_hosts(
            circle_id=large_circle_id,
            members=circle_members,
            member_roles=member_roles,
            format_prefix=format_prefix,
            region=circle_region,
            circle_number=circle_number
        )
        
        if not split_result['success']:
            print(f"⚠️ CIRCLE SPLITTER: Failed to split circle {large_circle_id}: {split_result['error']}")
            continue
            
        # Mark original circle for removal
        circles_to_remove.append(large_circle_id)
        
        # Create new circle entries for each split
        new_circle_ids = []
        for idx, split_data in enumerate(split_result['splits']):
            suffix = chr(65 + idx)  # A, B, C, etc.
            new_circle_id = f"{format_prefix}{circle_region}-SPLIT-{circle_number}-{suffix}"
            new_circle_ids.append(new_circle_id)
            
            # Create new circle data (inheriting properties from original)
            new_circle = circle_row.to_dict()
            new_circle['circle_id'] = new_circle_id
            new_circle['members'] = split_data['members']
            new_circle['member_count'] = len(split_data['members'])
            new_circle['always_hosts'] = split_data['always_hosts']
            new_circle['sometimes_hosts'] = split_data['sometimes_hosts']
            new_circle['co_leaders'] = split_data.get('co_leaders', 0)
            new_circle['is_split_circle'] = True
            new_circle['original_circle_id'] = large_circle_id
            
            # Set maximum additions (split circles can grow to 8 members)
            max_new_members = max(0, 8 - new_circle['member_count'])
            new_circle['max_additions'] = max_new_members
            
            print(f"🔴 CIRCLE SPLITTER: Created new circle {new_circle_id} with {new_circle['member_count']} members, {new_circle['always_hosts']} always hosts, {new_circle['sometimes_hosts']} sometimes hosts, can add {max_new_members} more")
            
            new_circles.append(new_circle)
        
        # Update summary with details of this split
        split_summary["split_details"].append({
            "original_circle_id": large_circle_id,
            "new_circle_ids": new_circle_ids,
            "member_counts": [len(split['members']) for split in split_result['splits']],
            "always_hosts": [split['always_hosts'] for split in split_result['splits']],
            "sometimes_hosts": [split['sometimes_hosts'] for split in split_result['splits']]
        })
        
        # Update statistics
        split_summary["total_circles_successfully_split"] += 1
        split_summary["total_new_circles_created"] += len(new_circle_ids)
        
        print(f"🔴 CIRCLE SPLITTER: Successfully split circle {large_circle_id} into {len(new_circle_ids)} new circles: {', '.join(new_circle_ids)}")
    
    # Remove original large circles and add new split circles
    if circles_to_remove:
        # Filter out the original large circles
        circles_df = circles_df[~circles_df['circle_id'].isin(circles_to_remove)]
        
        # Convert new circles to DataFrame and concatenate
        new_circles_df = pd.DataFrame(new_circles)
        updated_circles = pd.concat([circles_df, new_circles_df], ignore_index=True)
        
        print(f"🔴 CIRCLE SPLITTER: Final result: Removed {len(circles_to_remove)} large circles, added {len(new_circles)} split circles")
        
        return updated_circles, split_summary
    else:
        # No changes were made
        print("🔴 CIRCLE SPLITTER: No circles were split")
        return circles_df, split_summary

def get_member_roles(participants_data, member_ids):
    """
    Analyze the roles of members in a circle.
    
    Args:
        participants_data: DataFrame with participant information
        member_ids: List of member IDs
        
    Returns:
        dict: Dictionary with member roles information
    """
    roles = {
        'always_hosts': [],
        'sometimes_hosts': [],
        'never_hosts': [],
        'co_leaders': []
    }
    
    # Find the right host column (standardized or original)
    host_column = None
    possible_columns = ['host', 'Host', 'Host Status', 'host_status']
    
    for col in possible_columns:
        if col in participants_data.columns:
            host_column = col
            break
    
    # Find the co-leader column if it exists
    coleader_column = None
    possible_columns = ['co_leader', 'Co-Leader', 'Is Co-Leader']
    
    for col in possible_columns:
        if col in participants_data.columns:
            coleader_column = col
            break
    
    if not host_column:
        print("⚠️ CIRCLE SPLITTER: Could not find host column in participants data")
        return roles
    
    # Process each member
    for member_id in member_ids:
        # Find member in participants data
        member_rows = participants_data[participants_data['Encoded ID'] == member_id]
        
        if len(member_rows) == 0:
            print(f"⚠️ CIRCLE SPLITTER: Member {member_id} not found in participants data")
            continue
            
        member_row = member_rows.iloc[0]
        
        # Get host status
        host_status = str(member_row[host_column]).lower() if host_column in member_row else ''
        
        # Categorize host status
        if 'always' in host_status or 'yes' in host_status:
            roles['always_hosts'].append(member_id)
        elif 'sometimes' in host_status or 'maybe' in host_status:
            roles['sometimes_hosts'].append(member_id)
        else:
            roles['never_hosts'].append(member_id)
        
        # Check if member is a co-leader
        if coleader_column and coleader_column in member_row:
            coleader_value = str(member_row[coleader_column]).lower()
            if coleader_value in ['yes', 'true', '1']:
                roles['co_leaders'].append(member_id)
    
    print(f"🔴 CIRCLE SPLITTER: Member role analysis - {len(roles['always_hosts'])} always hosts, {len(roles['sometimes_hosts'])} sometimes hosts, {len(roles['never_hosts'])} never hosts, {len(roles['co_leaders'])} co-leaders")
    
    return roles

def split_circle_with_balanced_hosts(circle_id, members, member_roles, format_prefix, region, circle_number):
    """
    Split a circle into smaller circles with balanced host distribution.
    Each split circle must have at least one "Always Host" or two "Sometimes Host" members.
    Co-leaders should be distributed evenly across split circles.
    
    Args:
        circle_id: Original circle ID
        members: List of all member IDs
        member_roles: Dictionary mapping roles to list of member IDs
        format_prefix: Format prefix (IP- or V-)
        region: Region code
        circle_number: Original circle number
        
    Returns:
        dict: Result of splitting containing success status and split data
    """
    # Count total members
    total_members = len(members)
    
    # Determine optimal number of splits (each with at least 5 members)
    max_splits = total_members // 5
    if max_splits < 1:
        return {'success': False, 'error': 'Not enough members to split into groups of at least 5'}
    
    # We don't want more than 3 splits (A, B, C) to avoid getting too many small circles
    num_splits = min(max_splits, 3)
    
    # Calculate target size for each split
    target_size = total_members // num_splits
    
    print(f"🔴 CIRCLE SPLITTER: Splitting circle {circle_id} with {total_members} members into {num_splits} groups of approximately {target_size} members each")
    
    # Extract member role lists
    always_hosts = member_roles['always_hosts']
    sometimes_hosts = member_roles['sometimes_hosts']
    never_hosts = member_roles['never_hosts']
    co_leaders = member_roles['co_leaders']
    
    # Verify that we can meet the host requirements with the available hosts
    total_always = len(always_hosts)
    total_sometimes = len(sometimes_hosts)
    
    # Check if we have enough hosts to distribute
    # Each split needs either 1 always host or 2 sometimes hosts
    host_coverage = total_always + (total_sometimes // 2)
    
    if host_coverage < num_splits:
        print(f"⚠️ CIRCLE SPLITTER: Not enough hosts to distribute. Need {num_splits} hosts, have {host_coverage} (always={total_always}, sometimes={total_sometimes//2})")
        
        # If we have at least one always host or two sometimes hosts, we can still do a single split
        if total_always > 0 or total_sometimes >= 2:
            print(f"🔴 CIRCLE SPLITTER: Adjusting to a single split with all hosts")
            num_splits = 1
        else:
            return {'success': False, 'error': 'Not enough hosts to meet requirements for any split'}
    
    # Initialize splits with empty lists
    splits = [{'members': [], 'always_hosts': 0, 'sometimes_hosts': 0, 'co_leaders': 0} for _ in range(num_splits)]
    
    # First, distribute always hosts as evenly as possible
    for i, member_id in enumerate(always_hosts):
        split_idx = i % num_splits
        splits[split_idx]['members'].append(member_id)
        splits[split_idx]['always_hosts'] += 1
    
    # Then, distribute sometimes hosts, ensuring each split has at least two if there's no always host
    for i, member_id in enumerate(sometimes_hosts):
        # Find the split with the fewest sometimes hosts, prioritizing those without always hosts
        target_idx = find_optimal_split_for_sometimes_host(splits)
        
        splits[target_idx]['members'].append(member_id)
        splits[target_idx]['sometimes_hosts'] += 1
    
    # Next, distribute co-leaders evenly across splits, if not already distributed as hosts
    for i, member_id in enumerate(co_leaders):
        if member_id in always_hosts or member_id in sometimes_hosts:
            # Skip co-leaders that are also hosts (they're already distributed)
            continue
            
        # Find the split with the fewest co-leaders
        target_idx = min(range(num_splits), key=lambda idx: splits[idx]['co_leaders'])
        
        if member_id not in splits[target_idx]['members']:
            splits[target_idx]['members'].append(member_id)
            splits[target_idx]['co_leaders'] += 1
        
    # Finally, distribute remaining members (never hosts that aren't co-leaders)
    remaining_members = [m for m in never_hosts if m not in co_leaders]
    
    # Shuffle the remaining members to ensure random distribution
    random.shuffle(remaining_members)
    
    for i, member_id in enumerate(remaining_members):
        # Find the split with the fewest members
        target_idx = min(range(num_splits), key=lambda idx: len(splits[idx]['members']))
        
        if member_id not in splits[target_idx]['members']:
            splits[target_idx]['members'].append(member_id)
    
    # Final verification step to ensure all members are assigned
    assigned_members = [m for split in splits for m in split['members']]
    if len(assigned_members) != total_members:
        # Verify there are no duplicates
        if len(assigned_members) != len(set(assigned_members)):
            print(f"⚠️ CIRCLE SPLITTER: Warning - some members were assigned to multiple splits")
            
        missing_members = [m for m in members if m not in assigned_members]
        if missing_members:
            print(f"⚠️ CIRCLE SPLITTER: Warning - {len(missing_members)} members were not assigned to any split")
            
            # Assign missing members to the smallest split
            for member_id in missing_members:
                target_idx = min(range(num_splits), key=lambda idx: len(splits[idx]['members']))
                splits[target_idx]['members'].append(member_id)
    
    # Final check to ensure host requirements are met
    for i, split in enumerate(splits):
        always_count = split['always_hosts']
        sometimes_count = split['sometimes_hosts']
        
        if always_count == 0 and sometimes_count < 2:
            print(f"⚠️ CIRCLE SPLITTER: Split {i} does not meet host requirements")
            
            # Try to move an always host from another split if possible
            if total_always > 0:
                for j, other_split in enumerate(splits):
                    if j != i and other_split['always_hosts'] > 0:
                        # Find an always host to move
                        for member in other_split['members']:
                            if member in always_hosts:
                                # Move the host
                                other_split['members'].remove(member)
                                other_split['always_hosts'] -= 1
                                split['members'].append(member)
                                split['always_hosts'] += 1
                                print(f"🔴 CIRCLE SPLITTER: Moved always host {member} from split {j} to split {i}")
                                break
                        break
            
            # If we still don't have enough hosts, try to move sometimes hosts
            if split['always_hosts'] == 0 and split['sometimes_hosts'] < 2:
                # Find another split with excess sometimes hosts
                for j, other_split in enumerate(splits):
                    if j != i and other_split['sometimes_hosts'] > 2:
                        # Count how many we can take
                        to_take = min(2 - split['sometimes_hosts'], other_split['sometimes_hosts'] - 2)
                        
                        if to_take > 0:
                            # Find sometimes hosts to move
                            moved = 0
                            for member in list(other_split['members']):  # Create a copy to avoid modification during iteration
                                if member in sometimes_hosts and moved < to_take:
                                    # Move the host
                                    other_split['members'].remove(member)
                                    other_split['sometimes_hosts'] -= 1
                                    split['members'].append(member)
                                    split['sometimes_hosts'] += 1
                                    moved += 1
                                    print(f"🔴 CIRCLE SPLITTER: Moved sometimes host {member} from split {j} to split {i}")
                            
                            if moved > 0:
                                break
    
    # Final verification of host requirements
    all_requirements_met = True
    for i, split in enumerate(splits):
        host_requirement_met = split['always_hosts'] > 0 or split['sometimes_hosts'] >= 2
        if not host_requirement_met:
            print(f"⚠️ CIRCLE SPLITTER: Split {i} still does not meet host requirements after adjustment")
            all_requirements_met = False
    
    if not all_requirements_met:
        return {'success': False, 'error': 'Could not satisfy host requirements for all splits'}
    
    # Print summary of each split
    for i, split in enumerate(splits):
        print(f"🔴 CIRCLE SPLITTER: Split {i}: {len(split['members'])} members, {split['always_hosts']} always hosts, {split['sometimes_hosts']} sometimes hosts, {split['co_leaders']} co-leaders")
    
    return {
        'success': True,
        'splits': splits
    }

def find_optimal_split_for_sometimes_host(splits):
    """
    Find the optimal split to add a sometimes host to.
    Prioritize splits without always hosts and with fewer sometimes hosts.
    
    Args:
        splits: List of split data dictionaries
        
    Returns:
        int: Index of the optimal split
    """
    # First priority: splits with no always hosts and insufficient sometimes hosts
    needy_splits = [i for i, split in enumerate(splits) 
                    if split['always_hosts'] == 0 and split['sometimes_hosts'] < 2]
    
    if needy_splits:
        # Among the needy splits, choose the one with the fewest sometimes hosts
        return min(needy_splits, key=lambda idx: splits[idx]['sometimes_hosts'])
    
    # Second priority: the split with the fewest sometimes hosts overall
    return min(range(len(splits)), key=lambda idx: splits[idx]['sometimes_hosts'])