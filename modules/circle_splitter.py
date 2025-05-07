"""
Circle Splitter Module

Handles the logic for splitting large circles (11+ members) into smaller, more manageable circles
while ensuring each split circle meets host requirements.
"""

import pandas as pd
import numpy as np
import logging
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
    logger.info("Starting circle splitting process")
    print("\n🔄 CIRCLE SPLITTER: Processing circles to identify those with 11+ members")
    
    # Convert to DataFrame if it's a list of dictionaries
    if isinstance(circles_data, list):
        print(f"🔄 CIRCLE SPLITTER: Received {len(circles_data)} circles as list")
        circles_df = pd.DataFrame(circles_data)
    else:
        print(f"🔄 CIRCLE SPLITTER: Received DataFrame with {len(circles_data)} circles")
        circles_df = circles_data.copy()
        
    # Quick check of the data
    print(f"🔄 CIRCLE SPLITTER: DataFrame has columns: {list(circles_df.columns)}")
    if 'members' in circles_df.columns:
        print(f"🔄 CIRCLE SPLITTER: Sample members format: {str(circles_df['members'].iloc[0])[:100]}...")
    
    # Initialize tracking data
    split_summary = {
        'total_circles_eligible_for_splitting': 0,
        'total_circles_successfully_split': 0,
        'total_new_circles_created': 0,
        'circles_unable_to_split': [],
        'split_details': []
    }
    
    # Identify large circles (11+ members)
    large_circles = []
    
    # Determine how to count members based on data structure
    for idx, circle in circles_df.iterrows():
        circle_id = circle.get('circle_id')
        
        # Skip if this is already a split circle
        if 'SPLIT' in str(circle_id):
            continue
            
        # Get member count - handle different data structures
        member_count = 0
        if 'member_count' in circle:
            member_count = circle['member_count']
            print(f"🔍 CIRCLE SPLITTER: Found member_count {member_count} for circle {circle_id}")
        elif 'members' in circle:
            if isinstance(circle['members'], list):
                member_count = len(circle['members'])
                print(f"🔍 CIRCLE SPLITTER: Counted {member_count} members from list for circle {circle_id}")
            elif isinstance(circle['members'], str) and circle['members'].startswith('['):
                # Handle string representation of list
                try:
                    member_list = eval(circle['members'])
                    member_count = len(member_list)
                    print(f"🔍 CIRCLE SPLITTER: Parsed string list with {member_count} members for circle {circle_id}")
                except Exception as e:
                    # If eval fails, count commas + 1 as a fallback
                    member_count = circle['members'].count(',') + 1 if ',' in circle['members'] else 1
                    print(f"🔍 CIRCLE SPLITTER: Used fallback count of {member_count} for circle {circle_id} due to error: {str(e)}")
            else:
                print(f"⚠️ CIRCLE SPLITTER: Unexpected members format for circle {circle_id}: {type(circle['members'])}")
        else:
            print(f"⚠️ CIRCLE SPLITTER: No member count or members list found for circle {circle_id}")
        
        if member_count >= 11:
            logger.info(f"Circle {circle_id} has {member_count} members - eligible for splitting")
            large_circles.append((idx, circle, member_count))
            split_summary['total_circles_eligible_for_splitting'] += 1
    
    if not large_circles:
        logger.info("No large circles found for splitting")
        return circles_df, split_summary
    
    # Process each large circle
    new_circles = []
    indices_to_drop = []
    
    for idx, circle, member_count in large_circles:
        circle_id = circle['circle_id']
        logger.info(f"Processing large circle {circle_id} with {member_count} members")
        
        # Extract circle components for new ID generation
        circle_format = "IP"  # Default format
        circle_region = "UNK"  # Default region
        circle_number = "00"   # Default number
        
        # Parse the circle ID to extract components
        if '-' in circle_id:
            parts = circle_id.split('-')
            if len(parts) >= 3:
                circle_format = parts[0]
                circle_region = parts[1]
                circle_number = parts[2]
        
        # Get member information including host status and co-leader status
        members = get_circle_members(circle, participants_data)
        
        # Calculate optimal number of split circles
        num_splits = calculate_optimal_splits(member_count)
        logger.info(f"Optimal number of splits for {circle_id}: {num_splits}")
        
        # Analyze member roles
        members_analysis = analyze_member_roles(members)
        
        # Determine if splitting is possible based on host requirements
        can_split, split_explanation = can_meet_host_requirements(members_analysis, num_splits)
        
        if not can_split:
            logger.warning(f"Cannot split circle {circle_id}: {split_explanation}")
            split_summary['circles_unable_to_split'].append({
                'circle_id': circle_id,
                'member_count': member_count,
                'reason': split_explanation
            })
            continue
        
        # Split the circle into optimal groups
        split_groups = create_optimal_groups(members, members_analysis, num_splits)
        
        # Generate new circle records
        split_detail = {
            'original_circle_id': circle_id,
            'member_count': member_count,
            'num_splits': num_splits,
            'new_circle_ids': []
        }
        
        for i, group in enumerate(split_groups):
            # Generate split circle ID with letter suffix (A, B, C, etc.)
            suffix = chr(65 + i)  # A=65, B=66, etc. in ASCII
            new_circle_id = f"{circle_format}-{circle_region}-SPLIT-{circle_number}-{suffix}"
            
            # Create new circle record by copying and updating the original
            new_circle = circle.copy()
            new_circle['circle_id'] = new_circle_id
            new_circle['original_circle_id'] = circle_id
            new_circle['is_split_circle'] = True
            new_circle['split_letter'] = suffix
            new_circle['split_total'] = num_splits
            
            # Update members and counts
            new_circle['members'] = [m['id'] for m in group]
            new_circle['member_count'] = len(group)
            
            # Update host counts
            new_circle['always_hosts'] = sum(1 for m in group if m['host_status'].lower() == 'always host')
            new_circle['sometimes_hosts'] = sum(1 for m in group if m['host_status'].lower() == 'sometimes host')
            
            # Calculate max_additions (8 - current size)
            new_circle['max_additions'] = max(0, 8 - len(group))
            
            new_circles.append(new_circle)
            split_detail['new_circle_ids'].append(new_circle_id)
            
            logger.info(f"Created split circle {new_circle_id} with {len(group)} members")
        
        # Track this circle for removal from the original dataframe
        indices_to_drop.append(idx)
        
        # Add to split summary
        split_summary['total_circles_successfully_split'] += 1
        split_summary['total_new_circles_created'] += num_splits
        split_summary['split_details'].append(split_detail)
    
    # Remove original large circles
    circles_df = circles_df.drop(indices_to_drop)
    
    # Add new split circles
    new_circles_df = pd.DataFrame(new_circles)
    updated_circles = pd.concat([circles_df, new_circles_df], ignore_index=True)
    
    logger.info(f"Circle splitting complete. Created {len(new_circles)} new circles from {len(indices_to_drop)} large circles.")
    
    return updated_circles, split_summary

def get_circle_members(circle, participants_data):
    """
    Extract detailed member information for a circle.
    
    Args:
        circle: Circle data (dict or Series)
        participants_data: DataFrame with all participant data
        
    Returns:
        list: Members with their details (id, host status, co-leader status)
    """
    circle_id = circle.get('circle_id', 'unknown')
    print(f"🔍 CIRCLE SPLITTER: Getting members for circle {circle_id}")
    members_list = []
    
    # Extract member IDs from circle
    member_ids = []
    if 'members' in circle:
        if isinstance(circle['members'], list):
            member_ids = circle['members']
            print(f"🔍 CIRCLE SPLITTER: Found {len(member_ids)} members as list for circle {circle_id}")
        elif isinstance(circle['members'], str) and circle['members'].startswith('['):
            try:
                member_ids = eval(circle['members'])
                print(f"🔍 CIRCLE SPLITTER: Parsed {len(member_ids)} members from string list for circle {circle_id}")
            except Exception as e:
                # Fallback: parse comma-separated string
                member_ids = [m.strip(" '\"") for m in circle['members'].strip('[]').split(',')]
                print(f"🔍 CIRCLE SPLITTER: Used fallback parsing for {len(member_ids)} members for circle {circle_id}")
        else:
            print(f"⚠️ CIRCLE SPLITTER: Unexpected members format in get_circle_members: {type(circle['members'])}")
    else:
        print(f"⚠️ CIRCLE SPLITTER: No members field found for circle {circle_id}")
    
    # Get detailed information for each member
    for member_id in member_ids:
        # Find participant in data
        participant_rows = participants_data[participants_data['Encoded ID'] == member_id]
        
        if participant_rows.empty:
            # Member not found in participant data
            members_list.append({
                'id': member_id,
                'host_status': 'unknown',
                'is_co_leader': False
            })
            continue
        
        participant = participant_rows.iloc[0]
        
        # Determine host status
        host_status = 'not host'
        if 'host' in participant:
            host_status = str(participant['host']).lower()
        elif 'Host?' in participant:
            host_status = str(participant['Host?']).lower()
        
        # Try to apply standardization if possible
        try:
            from utils.data_standardization import normalize_host_status
            normalized_status = normalize_host_status(host_status)
            
            # Map to the format needed for splitting
            if normalized_status == 'ALWAYS':
                host_status = 'always host'
            elif normalized_status == 'SOMETIMES':
                host_status = 'sometimes host'
            else:
                host_status = 'not host'
            
            print(f"🔍 CIRCLE SPLITTER: Standardized host status from '{str(participant.get('host'))}' to '{host_status}'")
        except Exception as e:
            print(f"⚠️ CIRCLE SPLITTER: Couldn't standardize host status: {str(e)}")
        
        # Determine co-leader status
        is_co_leader = False
        co_leader_fields = ['Current Co-Leader', 'Co-Leader', 'Is Co-Leader', 'co_leader']
        
        for field in co_leader_fields:
            if field in participant:
                value = str(participant[field]).lower()
                if value in ['yes', 'true', '1']:
                    is_co_leader = True
                    break
        
        members_list.append({
            'id': member_id,
            'host_status': host_status,
            'is_co_leader': is_co_leader
        })
    
    return members_list

def analyze_member_roles(members):
    """
    Analyze the roles of members in a circle.
    
    Args:
        members: List of member dictionaries with role information
        
    Returns:
        dict: Analysis of member roles
    """
    analysis = {
        'total_members': len(members),
        'always_hosts': [],
        'sometimes_hosts': [],
        'co_leaders': [],
        'regular_members': []
    }
    
    for member in members:
        # Track by role
        if member['is_co_leader']:
            analysis['co_leaders'].append(member)
        
        if 'always' in member['host_status'].lower():
            analysis['always_hosts'].append(member)
        elif 'sometimes' in member['host_status'].lower():
            analysis['sometimes_hosts'].append(member)
        else:
            analysis['regular_members'].append(member)
    
    return analysis

def can_meet_host_requirements(members_analysis, num_splits):
    """
    Determine if the circle can be split while meeting host requirements.
    
    Args:
        members_analysis: Dictionary with member role analysis
        num_splits: Number of splits planned
        
    Returns:
        tuple: (can_split, explanation)
    """
    # Host requirements: at least one "Always Host" or at least two "Sometimes Host" per split circle
    always_hosts = len(members_analysis['always_hosts'])
    sometimes_hosts = len(members_analysis['sometimes_hosts'])
    
    # Calculate if we have enough hosts to meet requirements
    hosts_needed = num_splits  # Need at least one always host per split
    
    if always_hosts >= hosts_needed:
        return True, "Sufficient 'Always Host' members for all split circles"
    
    # If we don't have enough always hosts, check if sometimes hosts can fill the gap
    remaining_splits = hosts_needed - always_hosts
    sometimes_hosts_needed = remaining_splits * 2  # Need at least two sometimes hosts per remaining split
    
    if sometimes_hosts >= sometimes_hosts_needed:
        return True, "Combined 'Always Host' and 'Sometimes Host' members are sufficient"
    
    return False, f"Insufficient hosts: have {always_hosts} Always, {sometimes_hosts} Sometimes, need {hosts_needed} Always or {sometimes_hosts_needed} Sometimes"

def calculate_optimal_splits(member_count):
    """
    Calculate the optimal number of split circles.
    
    Args:
        member_count: Number of members in the circle
        
    Returns:
        int: Optimal number of splits
    """
    # Each split circle must have at least 5 members
    max_possible_splits = member_count // 5
    
    # Start with the maximum possible and work backward to find best fit
    for num_splits in range(max_possible_splits, 0, -1):
        # Calculate the size of each split group
        base_size = member_count // num_splits
        remainder = member_count % num_splits
        
        # Check if this split configuration would work
        smallest_group_size = base_size + (1 if remainder > 0 else 0)
        
        if smallest_group_size >= 5:
            return num_splits
    
    # Fallback (should never reach here if member_count >= 11)
    return 1

def create_optimal_groups(members, members_analysis, num_splits):
    """
    Create optimal groups for splitting the circle.
    
    Args:
        members: List of member dictionaries
        members_analysis: Analysis of member roles
        num_splits: Number of splits to create
        
    Returns:
        list: List of member groups for each split circle
    """
    # Calculate base size and remainder
    total_members = len(members)
    base_size = total_members // num_splits
    remainder = total_members % num_splits
    
    # Create empty groups
    groups = [[] for _ in range(num_splits)]
    
    # First, distribute co-leaders as evenly as possible
    co_leaders = members_analysis['co_leaders'].copy()
    for i, co_leader in enumerate(co_leaders):
        group_index = i % num_splits
        groups[group_index].append(co_leader)
        # Remove from other lists to avoid duplication
        if co_leader in members_analysis['always_hosts']:
            members_analysis['always_hosts'].remove(co_leader)
        elif co_leader in members_analysis['sometimes_hosts']:
            members_analysis['sometimes_hosts'].remove(co_leader)
        elif co_leader in members_analysis['regular_members']:
            members_analysis['regular_members'].remove(co_leader)
    
    # Next, distribute always hosts
    always_hosts = members_analysis['always_hosts'].copy()
    while always_hosts:
        # Find the group with the fewest members
        min_group_index = min(range(num_splits), key=lambda i: len(groups[i]))
        if always_hosts:
            groups[min_group_index].append(always_hosts.pop(0))
    
    # Next, distribute sometimes hosts in pairs if possible
    sometimes_hosts = members_analysis['sometimes_hosts'].copy()
    while sometimes_hosts:
        # Find the group with the fewest members
        min_group_index = min(range(num_splits), key=lambda i: len(groups[i]))
        if sometimes_hosts:
            groups[min_group_index].append(sometimes_hosts.pop(0))
            # Try to add a second sometimes host to the same group if available
            if sometimes_hosts:
                groups[min_group_index].append(sometimes_hosts.pop(0))
    
    # Finally, distribute remaining regular members
    regular_members = members_analysis['regular_members'].copy()
    while regular_members:
        # Find the group with the fewest members
        min_group_index = min(range(num_splits), key=lambda i: len(groups[i]))
        if regular_members:
            groups[min_group_index].append(regular_members.pop(0))
    
    # Verify the distribution is relatively even
    group_sizes = [len(g) for g in groups]
    min_size = min(group_sizes)
    max_size = max(group_sizes)
    
    if max_size - min_size > 1:
        # If the distribution is uneven, redistribute
        # Sort groups by size (ascending)
        sorted_indices = sorted(range(num_splits), key=lambda i: len(groups[i]))
        
        # Move members from largest to smallest groups until balanced
        while max_size - min_size > 1:
            # Get largest and smallest groups
            largest_idx = sorted_indices[-1]
            smallest_idx = sorted_indices[0]
            
            # Move a non-critical member (not a co-leader if possible)
            for i, member in enumerate(groups[largest_idx]):
                if not member.get('is_co_leader', False):
                    # Found a non-co-leader, safe to move
                    groups[smallest_idx].append(groups[largest_idx].pop(i))
                    break
            else:
                # If all are co-leaders, move the last one
                groups[smallest_idx].append(groups[largest_idx].pop())
            
            # Recalculate group sizes and sort indices
            group_sizes = [len(g) for g in groups]
            min_size = min(group_sizes)
            max_size = max(group_sizes)
            sorted_indices = sorted(range(num_splits), key=lambda i: len(groups[i]))
    
    return groups