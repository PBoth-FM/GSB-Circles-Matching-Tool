"""
Circle splitter module that identifies large circles (11+ members) and splits them
into smaller circles (at least 5 members each).
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Union, Optional

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
    # Initialize summary stats
    split_summary = {
        "total_circles_examined": 0,
        "total_large_circles_found": 0,
        "total_circles_successfully_split": 0,
        "total_new_circles_created": 0,
        "split_details": [],
        "circles_unable_to_split": []
    }
    
    # Convert circles_data to list of dictionaries if it's a DataFrame
    if isinstance(circles_data, pd.DataFrame):
        circles_list = circles_data.to_dict('records')
    else:
        circles_list = circles_data
    
    # Create a new list for updated circles
    updated_circles = []
    
    # Identify large circles (11+ members)
    for circle in circles_list:
        split_summary["total_circles_examined"] += 1
        
        # Get circle ID and member count
        circle_id = circle.get('circle_id', '')
        
        # Skip circles without IDs
        if not circle_id:
            updated_circles.append(circle)
            continue
        
        # Get member IDs
        members = circle.get('members', [])
        
        # Add detailed debugging for member data structure
        print(f"🔍 DEBUG: Circle {circle_id} members type: {type(members)}")
        print(f"🔍 DEBUG: Circle {circle_id} members sample: {str(members)[:200]}...")
        
        # Enhanced member handling using standardization module
        if not isinstance(members, list) or (members and not isinstance(members[0], str)):
            # Import standardization functionality
            from utils.data_standardization import normalize_member_list
            members = normalize_member_list(members)
            print(f"🔍 DEBUG: After normalization, circle {circle_id} has {len(members)} members")
        
        # Count members
        member_count = len(members) if members else circle.get('member_count', 0)
        
        # Skip if not a large circle (11+ members)
        if member_count < 11:
            updated_circles.append(circle)
            continue
            
        print(f"🔍 Found large circle {circle_id} with {member_count} members")
        split_summary["total_large_circles_found"] += 1
        
        # Get member roles to ensure proper host distribution
        member_roles = get_member_roles(participants_data, members)
        
        # Extract format prefix and region from circle ID
        format_prefix = circle_id.split('-')[0] if '-' in circle_id else "IP"
        
        parts = circle_id.split('-')
        region = parts[1] if len(parts) > 1 else ""
        
        # Extract circle number
        circle_number = ""
        if len(parts) > 2:
            # Handle numeric and alphanumeric circle numbers
            circle_number = parts[2]
        
        # Try to split the circle
        split_result = split_circle_with_balanced_hosts(
            circle_id=circle_id,
            members=members,
            member_roles=member_roles,
            format_prefix=format_prefix,
            region=region,
            circle_number=circle_number
        )
        
        if split_result["success"]:
            # Successfully split the circle
            split_summary["total_circles_successfully_split"] += 1
            split_summary["total_new_circles_created"] += len(split_result["new_circles"])
            
            # Add each new circle to the updated list
            for new_circle in split_result["new_circles"]:
                updated_circles.append(new_circle)
                
            # Add details to the summary
            split_detail = {
                "original_circle_id": circle_id,
                "member_count": member_count,
                "new_circle_ids": [c["circle_id"] for c in split_result["new_circles"]],
                "member_counts": [len(c["members"]) for c in split_result["new_circles"]],
                "always_hosts": split_result.get("always_hosts", []),
                "sometimes_hosts": split_result.get("sometimes_hosts", []),
                "region": region,
                "subregion": circle.get("subregion", ""),
                "meeting_time": circle.get("meeting_time", ""),
                "members": [c["members"] for c in split_result["new_circles"]]
            }
            split_summary["split_details"].append(split_detail)
            
            print(f"✅ Successfully split {circle_id} into {len(split_result['new_circles'])} circles")
            
        else:
            # Unable to split the circle
            updated_circles.append(circle)
            
            # Add to circles unable to split
            split_summary["circles_unable_to_split"].append({
                "circle_id": circle_id,
                "member_count": member_count,
                "reason": split_result.get("reason", "Unknown reason")
            })
            
            print(f"❌ Could not split {circle_id}: {split_result.get('reason', 'Unknown reason')}")
    
    return updated_circles, split_summary

def get_member_roles(participants_data, member_ids):
    """
    Analyze the roles of members in a circle.
    
    Args:
        participants_data: DataFrame with participant information
        member_ids: List of member IDs
        
    Returns:
        dict: Dictionary with member roles information
    """
    # Initialize role categories
    roles = {
        "always_host": [],
        "sometimes_host": [],
        "never_host": [],
        "co_leader": []
    }
    
    # Skip if no participant data
    if participants_data is None or len(member_ids) == 0:
        print(f"🔍 DEBUG: No participants or member_ids for role analysis")
        return roles
    
    # Log member_ids for debugging
    print(f"🔍 DEBUG: get_member_roles - Analyzing {len(member_ids)} members")
    print(f"🔍 DEBUG: get_member_roles - First few members: {str(member_ids[:5])}")
    
    # Ensure we have an Encoded ID column
    id_col = "Encoded ID"
    if id_col not in participants_data.columns:
        print(f"⚠️ WARNING: Could not find '{id_col}' column in participants data")
        return roles
    
    # Look for host column with different possible names
    host_col = None
    for col_name in ["host", "Host", "willing_to_host"]:
        if col_name in participants_data.columns:
            host_col = col_name
            break
    
    # Look for co-leader column
    co_leader_col = None
    for col_name in ["co_leader", "Co_Leader", "co-leader", "Co-Leader", "co_lead", "Co_Lead"]:
        if col_name in participants_data.columns:
            co_leader_col = col_name
            break
    
    # Process each member
    for member_id in member_ids:
        # Find this member in the DataFrame
        member_rows = participants_data[participants_data[id_col] == member_id]
        
        if not member_rows.empty:
            # Process host status
            if host_col:
                host_status = str(member_rows.iloc[0][host_col]).lower()
                
                if "always" in host_status or host_status == "yes":
                    roles["always_host"].append(member_id)
                elif "sometimes" in host_status or host_status == "maybe":
                    roles["sometimes_host"].append(member_id)
                else:
                    roles["never_host"].append(member_id)
            
            # Process co-leader status
            if co_leader_col and not pd.isna(member_rows.iloc[0][co_leader_col]):
                co_leader_status = str(member_rows.iloc[0][co_leader_col]).lower()
                
                if co_leader_status == "yes" or co_leader_status == "true" or co_leader_status == "1":
                    roles["co_leader"].append(member_id)
    
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
    # Ensure we have at least 11 members
    if len(members) < 11:
        return {
            "success": False,
            "reason": f"Not enough members to split (need 11+, found {len(members)})"
        }
    
    # Check if we have enough hosts
    always_hosts = member_roles.get("always_host", [])
    sometimes_hosts = member_roles.get("sometimes_host", [])
    co_leaders = member_roles.get("co_leader", [])
    
    # Calculate minimum required hosts for splitting
    min_splits = 2  # Minimum 2 splits for a large circle
    
    # For a valid split, each new circle needs either:
    # - At least 1 Always Host, or
    # - At least 2 Sometimes Hosts
    
    # Check if we have enough hosts for the required splits
    always_host_count = len(always_hosts)
    sometimes_host_count = len(sometimes_hosts)
    
    # Calculate how many circles we can create based on host distribution
    possible_circles_from_always = always_host_count
    possible_circles_from_sometimes = sometimes_host_count // 2
    
    max_possible_circles = possible_circles_from_always + possible_circles_from_sometimes
    
    if max_possible_circles < min_splits:
        return {
            "success": False,
            "reason": f"Not enough hosts for valid splitting. Need at least {min_splits} circles, " 
                     f"but can only create {max_possible_circles} with available hosts."
        }
    
    # Determine how many splits to create
    # Try to make all circles at least 5 members
    member_count = len(members)
    max_splits = member_count // 5
    
    # Choose number of splits, capped by host availability
    num_splits = min(max_splits, max_possible_circles)
    
    # Shuffle members to randomize, but keep a copy of original order
    shuffled_members = members.copy()
    
    # Split co-leaders evenly
    random.shuffle(co_leaders)
    
    # Create empty splits
    splits = []
    for i in range(num_splits):
        splits.append({
            "members": [],
            "always_hosts": 0,
            "sometimes_hosts": 0,
            "co_leaders": 0
        })
    
    # First, add always hosts, giving priority to circles without hosts
    random.shuffle(always_hosts)
    for host in always_hosts:
        # Find the circle with the fewest always hosts
        min_always_hosts = min(splits, key=lambda s: s["always_hosts"])
        min_always_hosts["members"].append(host)
        min_always_hosts["always_hosts"] += 1
        shuffled_members.remove(host)
    
    # Next, add sometimes hosts, prioritizing circles without always hosts
    random.shuffle(sometimes_hosts)
    for host in sometimes_hosts:
        if host not in shuffled_members:  # Skip if already assigned as an always host
            continue
            
        # Find optimal split to add this sometimes host
        target_split_index = find_optimal_split_for_sometimes_host(splits)
        splits[target_split_index]["members"].append(host)
        splits[target_split_index]["sometimes_hosts"] += 1
        shuffled_members.remove(host)
    
    # Next, add co-leaders, distributed evenly
    for leader in co_leaders:
        if leader not in shuffled_members:  # Skip if already assigned
            continue
            
        # Find the split with fewest co-leaders
        min_leaders = min(splits, key=lambda s: s["co_leaders"])
        min_leaders["members"].append(leader)
        min_leaders["co_leaders"] += 1
        shuffled_members.remove(leader)
    
    # Finally, distribute remaining members evenly
    random.shuffle(shuffled_members)
    
    # Calculate target size for each circle
    total_remaining = len(shuffled_members)
    base_size = total_remaining // num_splits
    extra = total_remaining % num_splits
    
    # Assign extra members to some circles
    for i in range(num_splits):
        target_size = base_size + (1 if i < extra else 0)
        
        # Add remaining members up to target size
        while len(splits[i]["members"]) < target_size and shuffled_members:
            splits[i]["members"].append(shuffled_members.pop(0))
    
    # Verify all splits meet minimum requirements
    for split in splits:
        if split["always_hosts"] == 0 and split["sometimes_hosts"] < 2:
            return {
                "success": False,
                "reason": "Could not create balanced splits that meet host requirements"
            }
    
    # Create the new circle objects
    new_circles = []
    always_hosts_counts = []
    sometimes_hosts_counts = []
    
    for i, split in enumerate(splits):
        # Generate new circle ID with suffix A, B, C, etc.
        suffix = chr(65 + i)  # A, B, C, ...
        
        # If original circle_id has a SPLIT section, it's already been split, so we handle differently
        if "SPLIT" in circle_id:
            # For already split circles being split again, we append another level
            # Example: IP-NAP-SPLIT-01-A becomes IP-NAP-SPLIT-01-A-A, IP-NAP-SPLIT-01-A-B, etc.
            new_circle_id = f"{circle_id}-{suffix}"
        else:
            # For first-time splits
            # Example: IP-NAP-01 becomes IP-NAP-SPLIT-01-A, IP-NAP-SPLIT-01-B, etc.
            new_circle_id = f"{format_prefix}-{region}-SPLIT-{circle_number}-{suffix}"
        
        # Create the new circle object with all necessary metadata
        new_circle = {
            "circle_id": new_circle_id,
            "members": split["members"],
            "member_count": len(split["members"]),
            "always_hosts": split["always_hosts"],
            "sometimes_hosts": split["sometimes_hosts"],
            "max_additions": max(0, 8 - len(split["members"])),  # Can grow to 8 members max
            "is_split_circle": True,
            "original_circle_id": circle_id,
            # Inherit other metadata from original circle
            "region": region,
            # These would be inherited from the original circle's data
            # through the CircleMetadataManager
        }
        
        new_circles.append(new_circle)
        always_hosts_counts.append(split["always_hosts"])
        sometimes_hosts_counts.append(split["sometimes_hosts"])
    
    # Return the result
    return {
        "success": True,
        "new_circles": new_circles,
        "always_hosts": always_hosts_counts,
        "sometimes_hosts": sometimes_hosts_counts
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
    # First, prioritize splits with no always hosts
    no_always_host_splits = [i for i, s in enumerate(splits) if s["always_hosts"] == 0]
    
    if no_always_host_splits:
        # Among splits with no always hosts, find the one with the fewest sometimes hosts
        return min(no_always_host_splits, key=lambda i: splits[i]["sometimes_hosts"])
    
    # If all splits have always hosts, just find the one with the fewest sometimes hosts
    return min(range(len(splits)), key=lambda i: splits[i]["sometimes_hosts"])