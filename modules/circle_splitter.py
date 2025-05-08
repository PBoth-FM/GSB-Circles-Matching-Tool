"""
Circle splitter module that identifies large circles (11+ members) and splits them
into smaller circles (at least 5 members each).
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Union, Optional

def rebuild_circle_member_lists(circles_data, participants_data):
    """
    Reconstruct complete member lists for all circles directly from participant data.
    This function addresses the disconnect between member_count and actual members lists.
    
    Args:
        circles_data: DataFrame or list of dictionaries containing circle data
        participants_data: DataFrame containing participant data with circle assignments
        
    Returns:
        List or DataFrame with rebuilt member lists for each circle
    """
    # Convert to DataFrame if needed for consistent processing
    input_was_list = not isinstance(circles_data, pd.DataFrame)
    if input_was_list:
        circles_df = pd.DataFrame(circles_data)
    else:
        circles_df = circles_data.copy()
    
    # Find the column that contains circle assignments
    # Use Current_Circle_ID as primary (only for CONTINUING participants), then fall back
    circle_col = None
    current_circle_col = None
    
    # First, check for Current_Circle_ID column which applies to continuing participants
    if 'Current_Circle_ID' in participants_data.columns:
        current_circle_col = 'Current_Circle_ID'
        print(f"✅ Found Current_Circle_ID column for CONTINUING participants")
        
        # Check how many participants have assignments
        if 'Status' in participants_data.columns:
            continuing_count = participants_data[
                (participants_data['Status'] == 'CONTINUING') & 
                (~participants_data['Current_Circle_ID'].isna())
            ].shape[0]
            print(f"✅ Found {continuing_count} CONTINUING participants with circle assignments")
    
    # Also look for a column for general/new assignments
    for col in ['assigned_circle', 'circle_id', 'Circle ID', 'proposed_NEW_circles_id']:
        if col in participants_data.columns:
            circle_col = col
            print(f"✅ Found general circle assignment column: '{circle_col}'")
            # Check how many participants have assignments
            assigned_count = participants_data[~participants_data[circle_col].isna()].shape[0]
            print(f"✅ Found {assigned_count} participants with assignments in column '{circle_col}'")
            break
    
    if not current_circle_col and not circle_col:
        print("⚠️ WARNING: Could not find circle assignment column in participants data")
        print(f"⚠️ Available columns: {participants_data.columns.tolist()}")
        return circles_data  # Return original data if we can't find assignments
    
    # Track which circles were updated
    circles_updated = 0
    all_circle_members = 0
    
    # For each circle, find all participants assigned to it
    for idx, circle in circles_df.iterrows():
        try:
            circle_id = circle.get('circle_id', '')
            if not circle_id:
                print(f"⚠️ WARNING: Circle at index {idx} has no circle_id, skipping")
                continue
            
            # Get members using both approaches
            members_list = []
            
            try:
                # Method 1: Check CONTINUING participants using Current_Circle_ID
                if current_circle_col:
                    try:
                        # Select CONTINUING participants in this circle
                        if 'Status' in participants_data.columns:
                            continuing_members = participants_data[
                                (participants_data[current_circle_col] == circle_id) & 
                                (participants_data['Status'] == 'CONTINUING')
                            ]
                        else:
                            continuing_members = participants_data[participants_data[current_circle_col] == circle_id]
                        
                        if 'Encoded ID' in participants_data.columns and not continuing_members.empty:
                            continuing_ids = continuing_members['Encoded ID'].dropna().tolist()
                            # Add to our member list
                            members_list.extend(continuing_ids)
                            print(f"👥 Circle {circle_id}: Found {len(continuing_ids)} CONTINUING members")
                    except Exception as e:
                        print(f"⚠️ ERROR processing CONTINUING members for {circle_id}: {str(e)}")
                
                # Method 2: Check general assignments using the other circle column
                if circle_col:
                    try:
                        general_members = participants_data[participants_data[circle_col] == circle_id]
                        if 'Encoded ID' in participants_data.columns and not general_members.empty:
                            general_ids = general_members['Encoded ID'].dropna().tolist()
                            # Only add IDs not already in the list
                            new_ids = [id for id in general_ids if id not in members_list]
                            if new_ids:
                                members_list.extend(new_ids)
                                print(f"👥 Circle {circle_id}: Found {len(new_ids)} additional members from '{circle_col}'")
                    except Exception as e:
                        print(f"⚠️ ERROR processing general members for {circle_id}: {str(e)}")
            except Exception as e:
                print(f"⚠️ ERROR finding members for circle {circle_id}: {str(e)}")
                # Continue to next circle if we can't process members for this one
                continue
            
            # If we found any members, update the circle
            if members_list:
                try:
                    # Filter out any None or NaN values and convert to strings
                    member_ids = [str(m) for m in members_list if m is not None and not pd.isna(m)]
                    
                    # Debug for specific test circles
                    if circle_id in ['IP-ATL-1', 'IP-NAP-01', 'IP-SHA-01']:
                        print(f"🔍 TEST CIRCLE {circle_id}: Found {len(member_ids)} total members")
                        if len(member_ids) > 0:
                            print(f"🔍 First few members: {member_ids[:3]}")
                        
                        # Check for member count mismatch
                        if 'member_count' in circles_df.columns:
                            current_count = circles_df.at[idx, 'member_count']
                            if current_count != len(member_ids):
                                print(f"⚠️ Member count mismatch for {circle_id}: stored={current_count}, found={len(member_ids)}")
                    
                    # Update the circle
                    circles_df.at[idx, 'members'] = member_ids
                    
                    # Only update member_count if it doesn't match the actual count
                    if 'member_count' in circles_df.columns:
                        current_count = circles_df.at[idx, 'member_count']
                        if current_count != len(member_ids):
                            print(f"🔍 Fixed member count mismatch for {circle_id}: {current_count} → {len(member_ids)}")
                        circles_df.at[idx, 'member_count'] = len(member_ids)
                    
                    circles_updated += 1
                    all_circle_members += len(member_ids)
                except Exception as e:
                    print(f"⚠️ ERROR updating circle {circle_id} with members: {str(e)}")
        except Exception as e:
            print(f"⚠️ ERROR processing circle at index {idx}: {str(e)}")
            continue
    
    print(f"✅ Successfully rebuilt member lists for {circles_updated} circles with a total of {all_circle_members} members")
    
    # Return in the same format as the input
    if input_was_list:
        return circles_df.to_dict('records')
    return circles_df

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
    
    print("🔍 STEP 1: Rebuilding circle member lists from participant data")
    try:
        # First, ensure all circles have proper member lists by rebuilding them
        circles_with_members = rebuild_circle_member_lists(circles_data, participants_data)
        
        # Convert to list of dictionaries if it's a DataFrame
        if isinstance(circles_with_members, pd.DataFrame):
            circles_list = circles_with_members.to_dict('records')
        else:
            circles_list = circles_with_members
            
        # Validate the result
        if not circles_list or len(circles_list) == 0:
            print("⚠️ WARNING: Circle rebuilding returned empty result!")
            # Return original data as fallback
            if isinstance(circles_data, pd.DataFrame):
                circles_list = circles_data.to_dict('records')
            else:
                circles_list = circles_data
    except Exception as e:
        print(f"⚠️ ERROR in rebuilding circle member lists: {str(e)}")
        # Fall back to original data
        if isinstance(circles_data, pd.DataFrame):
            circles_list = circles_data.to_dict('records')
        else:
            circles_list = circles_data
    
    # Create a new list for updated circles
    updated_circles = []
    
    print("🔍 STEP 2: Identifying and splitting large circles")
    # Identify large circles (11+ members)
    for circle in circles_list:
        try:
            split_summary["total_circles_examined"] += 1
            
            # Get circle ID and member count
            circle_id = circle.get('circle_id', '')
            
            # Skip circles without IDs
            if not circle_id:
                print(f"⚠️ WARNING: Circle without ID found, skipping")
                updated_circles.append(circle)
                continue
            
            # Get member list - should now be properly populated
            members = circle.get('members', [])
            
            # Ensure members is a valid list
            if members is None:
                print(f"⚠️ WARNING: Circle {circle_id} has None for members list")
                members = []
            elif not isinstance(members, list):
                try:
                    # Try to convert to list if possible
                    print(f"⚠️ WARNING: Circle {circle_id} members is not a list, attempting conversion")
                    if isinstance(members, str):
                        # Handle string representation of list
                        if members.startswith('[') and members.endswith(']'):
                            try:
                                import ast
                                members = ast.literal_eval(members)
                            except Exception as e:
                                print(f"⚠️ WARNING: Failed to parse member string as list: {str(e)}")
                                members = []
                        else:
                            members = [members]  # Single string item
                    else:
                        members = [members]  # Wrap in list
                except Exception as e:
                    print(f"⚠️ ERROR converting members to list for {circle_id}: {str(e)}")
                    members = []
            
            # Log details about the members
            print(f"🔍 Circle {circle_id}: {len(members)} members, member_count={circle.get('member_count', 0)}")
            if members and len(members) > 0:
                print(f"🔍 Sample members: {str(members[:3])}...")
            else:
                print(f"⚠️ No members found for circle {circle_id}")
            
            # Count members - use actual list length
            member_count = len(members)
            
            # Skip if not a large circle (11+ members)
            if member_count < 11:
                updated_circles.append(circle)
                continue
                
            print(f"🔍 Found large circle {circle_id} with {member_count} members")
            split_summary["total_large_circles_found"] += 1
            
            # Initialize split_result with a default failure state
            split_result = {
                "success": False,
                "reason": "Not processed yet"
            }
            
            try:
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
            except Exception as e:
                print(f"⚠️ ERROR preparing for circle split of {circle_id}: {str(e)}")
                # Skip splitting and keep original circle
                updated_circles.append(circle)
                split_summary["circles_unable_to_split"].append({
                    "circle_id": circle_id,
                    "member_count": member_count,
                    "reason": f"Error during split: {str(e)}"
                })
                continue
            
            # Process the split result
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
        except Exception as e:
            print(f"⚠️ CRITICAL ERROR processing circle: {str(e)}")
            # If we get a critical error, keep the original circle and continue
            try:
                updated_circles.append(circle)
            except Exception:
                pass  # If we can't even add the circle, just skip it
            continue
    
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
        try:
            # Find this member in the DataFrame
            member_rows = participants_data[participants_data[id_col] == member_id]
            
            if not member_rows.empty:
                # Process host status
                if host_col:
                    try:
                        host_value = member_rows.iloc[0][host_col]
                        # Handle potential NaN values
                        if pd.isna(host_value):
                            print(f"⚠️ WARNING: NaN host value for member {member_id}")
                            roles["never_host"].append(member_id)
                            continue
                            
                        host_status = str(host_value).lower()
                        
                        if "always" in host_status or host_status == "yes":
                            roles["always_host"].append(member_id)
                        elif "sometimes" in host_status or host_status == "maybe":
                            roles["sometimes_host"].append(member_id)
                        else:
                            roles["never_host"].append(member_id)
                    except Exception as e:
                        print(f"⚠️ WARNING: Error processing host status for member {member_id}: {str(e)}")
                        roles["never_host"].append(member_id)  # Default to never host on error
                else:
                    # No host column, default to never host
                    roles["never_host"].append(member_id)
                
                # Process co-leader status
                if co_leader_col:
                    try:
                        if not pd.isna(member_rows.iloc[0][co_leader_col]):
                            co_leader_status = str(member_rows.iloc[0][co_leader_col]).lower()
                            
                            if co_leader_status == "yes" or co_leader_status == "true" or co_leader_status == "1":
                                roles["co_leader"].append(member_id)
                    except Exception as e:
                        print(f"⚠️ WARNING: Error processing co-leader status for member {member_id}: {str(e)}")
        except Exception as e:
            print(f"⚠️ WARNING: Error processing member {member_id}: {str(e)}")
            # Default to never host on error
            roles["never_host"].append(member_id)
    
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
    # ENHANCED DEBUGGING - Log original member count
    original_member_count = len(members)
    print(f"🔍 DEBUG: Starting split of circle {circle_id} with {original_member_count} members")
    print(f"🔍 DEBUG: Member IDs: {members}")
    
    # Ensure we have at least 11 members
    if original_member_count < 11:
        return {
            "success": False,
            "reason": f"Not enough members to split (need 11+, found {original_member_count})"
        }
    
    # Check if we have enough hosts
    always_hosts = member_roles.get("always_host", [])
    sometimes_hosts = member_roles.get("sometimes_host", [])
    co_leaders = member_roles.get("co_leader", [])
    never_hosts = member_roles.get("never_host", [])
    
    print(f"🔍 DEBUG: Role counts - Always Hosts: {len(always_hosts)}, " 
          f"Sometimes Hosts: {len(sometimes_hosts)}, "
          f"Co-Leaders: {len(co_leaders)}, "
          f"Never Hosts: {len(never_hosts)}")
    
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
    
    # IMPROVED: Determine optimal number of splits based on member count
    # Try to make all circles roughly equal size with minimum 5 members
    member_count = original_member_count
    
    # Calculate ideal number of splits for relatively equal distribution
    # For 11 members: 2 splits (5-6)
    # For 12-15 members: 2-3 splits
    # For 16+ members: 3+ splits
    
    # Start with simple calculation: divide by target size (5-8)
    target_min_size = 5
    max_splits = member_count // target_min_size
    
    # Simple heuristic: prefer smaller number of larger circles
    # If member_count is just over a multiple of 5, use fewer splits
    if member_count % target_min_size <= 2 and max_splits > 2:
        max_splits -= 1
    
    # Hard cap at 3 splits to prevent too many small circles
    max_splits = min(max_splits, 3)
    
    # Choose number of splits, capped by host availability and minimum 2 splits
    num_splits = min(max_splits, max_possible_circles)
    num_splits = max(num_splits, 2)  # Ensure at least 2 splits
    
    print(f"🔍 DEBUG: Creating {num_splits} splits for {member_count} members")
    
    # Create a copy of members to track assignments
    unassigned_members = members.copy()
    
    # Track member assignments for verification
    assigned_members = []
    
    # Create empty splits
    splits = []
    for i in range(num_splits):
        splits.append({
            "members": [],
            "always_hosts": 0,
            "sometimes_hosts": 0,
            "co_leaders": 0
        })
    
    # IMPROVED: Calculate ideal target size for each circle before any assignments
    ideal_size = member_count // num_splits
    remainder = member_count % num_splits
    
    target_sizes = [ideal_size + (1 if i < remainder else 0) for i in range(num_splits)]
    print(f"🔍 DEBUG: Target sizes for splits: {target_sizes}")
    
    # IMPROVED FUNCTION: Find optimal split based on both host needs and size balancing
    def find_optimal_split(member_id, is_always_host=False, is_sometimes_host=False, is_co_leader=False):
        # Calculate a score for each split combining:
        # 1. Host needs (always hosts or sometimes hosts)
        # 2. Current size vs. target size
        # 3. Co-leader distribution
        
        best_score = float('-inf')
        best_index = 0
        
        for i, split in enumerate(splits):
            score = 0
            
            # Host factors
            if is_always_host:
                # Prefer circles with no always hosts
                if split["always_hosts"] == 0:
                    score += 50
                else:
                    score -= split["always_hosts"] * 10
            
            if is_sometimes_host:
                # Prefer circles with no always hosts and fewer sometimes hosts
                if split["always_hosts"] == 0:
                    score += 30
                    
                # Penalize for existing sometimes hosts
                score -= split["sometimes_hosts"] * 5
            
            if is_co_leader:
                # Prefer even distribution of co-leaders
                score -= split["co_leaders"] * 8
            
            # Size factor - how close to target size?
            # Strongly prefer smaller circles until they reach target size
            remaining_capacity = target_sizes[i] - len(split["members"])
            
            if remaining_capacity > 0:
                # Proportional bonus for having space remaining
                score += 20 * (remaining_capacity / target_sizes[i])
            else:
                # Heavy penalty for exceeding target
                score -= 50 * abs(remaining_capacity)
            
            if score > best_score:
                best_score = score
                best_index = i
        
        return best_index
    
    # Process member assignment strategy:
    # 1. First pass: Distribute hosts with balanced approach
    # 2. Second pass: Distribute remaining members to balance sizes
    
    # First, handle always hosts
    print(f"🔍 DEBUG: Assigning {len(always_hosts)} Always Hosts")
    random.shuffle(always_hosts)
    for host in always_hosts:
        if host in unassigned_members:
            target_split_index = find_optimal_split(host, is_always_host=True)
            splits[target_split_index]["members"].append(host)
            splits[target_split_index]["always_hosts"] += 1
            assigned_members.append(host)
            unassigned_members.remove(host)
            print(f"  - Always Host {host} assigned to split {target_split_index}")
    
    # Next, handle sometimes hosts
    print(f"🔍 DEBUG: Assigning {len(sometimes_hosts)} Sometimes Hosts")
    random.shuffle(sometimes_hosts)
    for host in sometimes_hosts:
        if host in unassigned_members:
            target_split_index = find_optimal_split(host, is_sometimes_host=True)
            splits[target_split_index]["members"].append(host)
            splits[target_split_index]["sometimes_hosts"] += 1
            assigned_members.append(host)
            unassigned_members.remove(host)
            print(f"  - Sometimes Host {host} assigned to split {target_split_index}")
    
    # Next, handle co-leaders
    print(f"🔍 DEBUG: Assigning {len(co_leaders)} Co-Leaders")
    random.shuffle(co_leaders)
    for leader in co_leaders:
        if leader in unassigned_members:
            target_split_index = find_optimal_split(leader, is_co_leader=True)
            splits[target_split_index]["members"].append(leader)
            splits[target_split_index]["co_leaders"] += 1
            assigned_members.append(leader)
            unassigned_members.remove(leader)
            print(f"  - Co-Leader {leader} assigned to split {target_split_index}")
    
    # Finally, distribute remaining members to balance sizes
    print(f"🔍 DEBUG: Assigning {len(unassigned_members)} remaining members")
    random.shuffle(unassigned_members)
    
    # Assign remaining members prioritizing the smaller circles
    for member in unassigned_members:
        # Find the split furthest from its target size (smallest relative to target)
        smallest_index = min(range(num_splits), 
                           key=lambda i: len(splits[i]["members"]) / target_sizes[i])
        
        splits[smallest_index]["members"].append(member)
        assigned_members.append(member)
        print(f"  - Regular member {member} assigned to split {smallest_index}")
    
    # VALIDATION: Check if all members were assigned
    if len(assigned_members) != original_member_count:
        print(f"⚠️ WARNING: Member count mismatch! Original: {original_member_count}, Assigned: {len(assigned_members)}")
        
        # Find missing members
        missing = set(members) - set(assigned_members)
        if missing:
            print(f"⚠️ WARNING: Missing members: {missing}")
        
        # Find extra members
        extra = set(assigned_members) - set(members)
        if extra:
            print(f"⚠️ WARNING: Extra members: {extra}")
    
    # Verify each split has a valid host composition (at least 1 always or 2 sometimes)
    valid_splits = True
    for i, split in enumerate(splits):
        has_valid_hosts = (
            split["always_hosts"] >= 1 or 
            split["sometimes_hosts"] >= 2
        )
        
        if not has_valid_hosts:
            print(f"⚠️ WARNING: Split {i} does not have valid host composition")
            print(f"    Always Hosts: {split['always_hosts']}, Sometimes Hosts: {split['sometimes_hosts']}")
            valid_splits = False
    
    if not valid_splits:
        return {
            "success": False,
            "reason": "Unable to create splits with valid host composition"
        }
    
    # Verify all splits have at least 5 members
    for i, split in enumerate(splits):
        if len(split["members"]) < 5:
            print(f"⚠️ WARNING: Split {i} has fewer than 5 members: {len(split['members'])}")
            return {
                "success": False,
                "reason": f"Split {i} has only {len(split['members'])} members (minimum 5 required)"
            }
    
    # Create the new circles based on the original circle's metadata
    # Use naming convention: [FORMAT]-[REGION]-SPLIT-[OLD NUMBER]-A/B/C
    new_circles = []
    for i, split in enumerate(splits):
        # Use letters for suffixes: A, B, C, etc.
        suffix = chr(65 + i)  # ASCII 65 = 'A', 66 = 'B', etc.
        
        # Create new circle ID
        new_circle_id = f"{format_prefix}-{region}-SPLIT-{circle_number}-{suffix}"
        
        # Create new circle with metadata from original
        new_circle = {
            "circle_id": new_circle_id,
            "original_circle_id": circle_id,
            "members": split["members"],
            "member_count": len(split["members"]),
            "format": format_prefix,
            "region": region,
            "is_split_circle": True,
            "split_source": circle_id,
            "split_index": i,
            "max_additions": max(0, 8 - len(split["members"])),  # Allow growth up to 8 total
            "eligible_for_new_members": True,  # Split circles should be eligible for new members
            "count_always_hosts": split["always_hosts"],
            "count_sometimes_hosts": split["sometimes_hosts"],
            "count_co_leaders": split["co_leaders"]
        }
        
        new_circles.append(new_circle)
    
    # Return success result with new circles
    return {
        "success": True,
        "original_circle_id": circle_id,
        "new_circles": new_circles,
        "always_hosts": always_hosts,
        "sometimes_hosts": sometimes_hosts,
        "co_leaders": co_leaders
    }

# This function is replaced by the improved find_optimal_split function in the split_circle_with_balanced_hosts function
def find_optimal_split_for_sometimes_host(splits):
    """
    Find the optimal split to add a sometimes host to.
    Prioritize splits without always hosts and with fewer sometimes hosts.
    This is a legacy function, replaced by the improved find_optimal_split function.
    
    Args:
        splits: List of split data dictionaries
        
    Returns:
        int: Index of the optimal split
    """
    best_split_index = 0
    best_score = float('inf')  # Lower score is better
    
    for i, split in enumerate(splits):
        # Calculate a score based on host composition
        # We want to prioritize:
        # 1. Splits without always hosts
        # 2. Splits with fewer sometimes hosts
        
        # Start with the number of sometimes hosts as the base score
        score = split['sometimes_hosts']
        
        # Add a large penalty if the split already has an always host
        if split['always_hosts'] > 0:
            score += 100
        
        # Choose the split with the lowest score
        if score < best_score:
            best_score = score
            best_split_index = i
    
    return best_split_index

# Additional utility functions can be added here as needed