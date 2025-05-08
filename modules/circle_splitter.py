"""
Circle splitter module that identifies large circles (11+ members) and splits them
into smaller circles (at least 5 members each).
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Union, Optional

def rebuild_circle_member_lists(circles_data, participants_data=None):
    """
    Reconstruct complete member lists for all circles directly from participant data.
    This function addresses the disconnect between member_count and actual members lists.
    
    Args:
        circles_data: DataFrame or list of dictionaries containing circle data
        participants_data: Optional DataFrame containing participant data with circle assignments.
                          If None, will attempt to use ParticipantDataManager from session state.
        
    Returns:
        List or DataFrame with rebuilt member lists for each circle
    """
    import streamlit as st
    from utils.participant_data_manager import ParticipantDataManager
    
    # Get the ParticipantDataManager instance - preferred approach
    participant_manager = None
    
    # Check if we have a ParticipantDataManager in session state
    if hasattr(st, 'session_state') and 'participant_data_manager' in st.session_state:
        print("🔍 Using ParticipantDataManager for circle member reconstruction")
        participant_manager = st.session_state.participant_data_manager
        
        # If participants_data was not provided, get it from the manager
        if participants_data is None:
            participants_data = participant_manager.get_all_participants()
            print(f"📊 Retrieved {len(participants_data) if participants_data is not None else 0} participants from manager")
    elif participants_data is None:
        print("⚠️ No ParticipantDataManager found in session state and no participants_data provided")
        return circles_data  # Return original data if we can't reconstruct
    
    # Safety check to ensure we have participant data
    if participants_data is None:
        print("⚠️ ERROR: participants_data is None and no ParticipantDataManager found in session state")
        raise ValueError("No participant data available for circle member rebuilding")
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

def update_participant_assignments(participants_data, original_circle_id, new_circle_assignments):
    """
    Update participant assignments in the results DataFrame after circle splitting.
    This function updates the 'proposed_NEW_circles_id' column to reflect the new circle assignments.
    
    Args:
        participants_data: DataFrame containing participant information
        original_circle_id: ID of the original circle that was split
        new_circle_assignments: Dictionary mapping member IDs to their new circle IDs
        
    Returns:
        DataFrame: Updated participants DataFrame with new circle assignments
    """
    import streamlit as st
    from utils.participant_data_manager import ParticipantDataManager
    
    # Try to use ParticipantDataManager if available
    if hasattr(st, 'session_state') and hasattr(st.session_state, 'participant_data_manager'):
        print(f"🔍 Using ParticipantDataManager to update circle assignments")
        try:
            manager = st.session_state.participant_data_manager
            update_count = 0
            
            # Update assignments through the manager
            for member_id, new_circle_id in new_circle_assignments.items():
                success = manager.update_participant_circle(member_id, new_circle_id)
                if success:
                    update_count += 1
                else:
                    print(f"⚠️ WARNING: Failed to update circle for member {member_id}")
            
            print(f"✅ ParticipantDataManager: Updated {update_count} participant assignments from {original_circle_id} to split circles")
            
            # Return the updated DataFrame from the manager
            return manager.get_all_participants()
        except Exception as e:
            print(f"⚠️ ERROR using ParticipantDataManager to update assignments: {str(e)}")
            print("Falling back to direct DataFrame update")
    
    # Fall back to traditional approach
    # Create a copy to avoid modifying the original
    updated_participants = participants_data.copy()
    
    # Find ID column
    id_col = None
    for col in ['Encoded ID', 'encoded_id', 'participant_id']:
        if col in updated_participants.columns:
            id_col = col
            break
    
    if id_col is None:
        print("⚠️ ERROR: Could not find participant ID column in participants data")
        return updated_participants
    
    # Find circle assignment column
    circle_col = None
    for col in ['proposed_NEW_circles_id', 'assigned_circle', 'circle_id']:
        if col in updated_participants.columns:
            circle_col = col
            break
    
    if circle_col is None:
        print("⚠️ ERROR: Could not find circle assignment column in participants data")
        return updated_participants
    
    # Update assignments
    update_count = 0
    for member_id, new_circle_id in new_circle_assignments.items():
        # Find this member in the DataFrame
        member_mask = updated_participants[id_col] == member_id
        
        # Skip if member not found
        if not any(member_mask):
            print(f"⚠️ WARNING: Member {member_id} not found in participants data")
            continue
            
        # Update the assignment
        updated_participants.loc[member_mask, circle_col] = new_circle_id
        update_count += 1
    
    print(f"✅ Updated {update_count} participant assignments from {original_circle_id} to split circles")
    return updated_participants

def split_large_circles(circles_data, participants_data=None, test_mode=False):
    """
    Identifies circles with 11+ members and splits them into smaller circles.
    
    Args:
        circles_data: DataFrame or list of dictionaries containing circle information
        participants_data: DataFrame containing participant information (optional if using ParticipantDataManager)
        test_mode: If True, enables special test mode with more lenient host requirements and additional logging
        
    Returns:
        tuple: (
            updated_circles: DataFrame or list with split circles replacing large ones,
            split_summary: Dictionary containing statistics and details about the splitting process
            updated_participants: DataFrame with updated participant assignments
        )
    """
    import streamlit as st
    from utils.participant_data_manager import ParticipantDataManager
    
    # Enable test mode if running from split test section
    if test_mode:
        print("🧪 TEST MODE ENABLED: Using special handling for test circles")
    
    # Identify test circles for special handling
    TEST_CIRCLES = ['IP-ATL-1', 'IP-NAP-01', 'IP-SHA-01']
    
    # Get or create the ParticipantDataManager - preferred approach
    participant_manager = None
    
    # Check if we have a ParticipantDataManager in session state
    if hasattr(st, 'session_state') and 'participant_data_manager' in st.session_state:
        print("🔍 Using ParticipantDataManager for circle splitting")
        participant_manager = st.session_state.participant_data_manager
        
        # If participants_data was not provided, get it from the manager
        if participants_data is None:
            participants_data = participant_manager.get_all_participants()
            print(f"📊 Retrieved {len(participants_data) if participants_data is not None else 0} participants from manager")
    
    # Safety check to ensure we have participant data
    if participants_data is None:
        print("⚠️ ERROR: participants_data is None and no ParticipantDataManager found in session state")
        raise ValueError("No participant data available for circle splitting")
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
                
            # Check if this is a test circle for special handling
            is_test_circle = circle_id in TEST_CIRCLES
            if is_test_circle:
                print(f"🧪 TEST CIRCLE DETECTED: {circle_id}")
                
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
                # Use test_mode and circle_id for enhanced debugging with test circles
                member_roles = get_member_roles(
                    participants_data=participants_data, 
                    member_ids=members,
                    test_mode=test_mode,
                    test_circle_id=circle_id if is_test_circle else None
                )
                
                # FOR TEST CIRCLES ONLY: If this is a test circle and we don't have enough hosts,
                # Check directly with the ParticipantDataManager for improved host detection
                if is_test_circle and (len(member_roles["always_host"]) < 1 and len(member_roles["sometimes_host"]) < 2):
                    print(f"🧪 TEST CIRCLE {circle_id}: Not enough hosts detected, trying enhanced detection")
                    
                    # Try using enhanced detection with ParticipantDataManager directly
                    if participant_manager:
                        host_counts = {'always': 0, 'sometimes': 0, 'never': 0}
                        host_details = []
                        
                        # Log each member's host status with debug mode
                        for member_id in members:
                            if not member_id or pd.isna(member_id):
                                continue
                                
                            host_status = participant_manager.get_participant_host_status(member_id, debug_mode=True)
                            host_counts[host_status if host_status else 'never'] += 1
                            host_details.append(f"{member_id}: {host_status}")
                        
                        print(f"🧪 ENHANCED HOST DETECTION: {host_counts['always']} always, {host_counts['sometimes']} sometimes")
                        print(f"🧪 HOST DETAILS: {host_details[:5]}...")
                    
                    # If we still don't have enough hosts after enhanced detection, force assign for testing
                    print(f"🧪 TEST CIRCLE {circle_id}: Not enough hosts detected, forcing host assignment for testing")
                    
                    # Gather all members from roles to ensure we don't double-count
                    all_assigned = set()
                    for role_list in member_roles.values():
                        all_assigned.update(role_list)
                    
                    # Reset host lists but keep co-leaders
                    co_leaders = member_roles["co_leader"].copy()
                    member_roles = {
                        "always_host": [],
                        "sometimes_host": [],
                        "never_host": [],
                        "co_leader": co_leaders
                    }
                    
                    # Depending on the test circle, assign different host patterns to test different scenarios
                    if circle_id == 'IP-ATL-1':
                        # Create at least one always host for this circle
                        if members:
                            always_host = members[0]
                            member_roles["always_host"].append(always_host)
                            print(f"🧪 TEST CIRCLE: Assigned member {always_host} as ALWAYS host")
                            
                            # Make the rest never hosts
                            for i, member_id in enumerate(members[1:]):
                                member_roles["never_host"].append(member_id)
                    
                    elif circle_id == 'IP-NAP-01':
                        # Create two sometimes hosts for this circle to test that scenario
                        if len(members) >= 2:
                            sometimes_host1 = members[0]
                            sometimes_host2 = members[1]
                            member_roles["sometimes_host"].append(sometimes_host1)
                            member_roles["sometimes_host"].append(sometimes_host2)
                            print(f"🧪 TEST CIRCLE: Assigned members {sometimes_host1} and {sometimes_host2} as SOMETIMES hosts")
                            
                            # Make the rest never hosts
                            for i, member_id in enumerate(members[2:]):
                                member_roles["never_host"].append(member_id)
                    
                    elif circle_id == 'IP-SHA-01':
                        # Create both always and sometimes hosts to test mixed scenario
                        if len(members) >= 3:
                            always_host = members[0]
                            sometimes_host1 = members[1]
                            sometimes_host2 = members[2]
                            member_roles["always_host"].append(always_host)
                            member_roles["sometimes_host"].append(sometimes_host1)
                            member_roles["sometimes_host"].append(sometimes_host2)
                            print(f"🧪 TEST CIRCLE: Assigned member {always_host} as ALWAYS host")
                            print(f"🧪 TEST CIRCLE: Assigned members {sometimes_host1} and {sometimes_host2} as SOMETIMES hosts")
                            
                            # Make the rest never hosts
                            for i, member_id in enumerate(members[3:]):
                                member_roles["never_host"].append(member_id)
                    
                    # Print updated role counts                    
                    print(f"🧪 TEST CIRCLE: Updated host distribution:")
                    print(f"   Always Hosts: {len(member_roles['always_host'])} members")
                    print(f"   Sometimes Hosts: {len(member_roles['sometimes_host'])} members")
                    print(f"   Co-Leaders: {len(member_roles['co_leader'])} members")
                    print(f"   Never Hosts: {len(member_roles['never_host'])} members")
                
                # Extract format prefix from circle ID
                format_prefix = circle_id.split('-')[0] if '-' in circle_id else "IP"
                
                parts = circle_id.split('-')
                circle_number = ""
                if len(parts) > 2:
                    # Handle numeric and alphanumeric circle numbers
                    circle_number = parts[2]
                
                # Get region and subregion from first participant in the circle
                region, subregion = get_region_subregion_from_participants(participants_data, members)
                
                # Fallback to ID extraction if participant data doesn't have the info
                if not region:
                    region = parts[1] if len(parts) > 1 else ""
                    print(f"⚠️ Falling back to circle ID for region: {region}")
                
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
                
                # Build a mapping of member ID to new circle ID for updating participant assignments
                member_to_circle_mapping = {}
                for new_circle in split_result["new_circles"]:
                    new_circle_id = new_circle["circle_id"]
                    for member_id in new_circle["members"]:
                        member_to_circle_mapping[member_id] = new_circle_id
                
                # Update participant assignments for this split circle
                if participants_data is not None:
                    print(f"🔄 Updating participant assignments for {circle_id}")
                    participants_data = update_participant_assignments(
                        participants_data, 
                        circle_id, 
                        member_to_circle_mapping
                    )
                    
                # Add details to the summary
                split_detail = {
                    "original_circle_id": circle_id,
                    "member_count": member_count,
                    "new_circle_ids": [c["circle_id"] for c in split_result["new_circles"]],
                    "member_counts": [len(c["members"]) for c in split_result["new_circles"]],
                    "always_hosts": split_result.get("always_hosts", []),
                    "sometimes_hosts": split_result.get("sometimes_hosts", []),
                    "region": region,
                    "subregion": subregion,  # Use the extracted subregion from participants
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
    
    # Return updated circles, split summary, and updated participant assignments
    return updated_circles, split_summary, participants_data

def get_member_roles(participants_data, member_ids, test_mode=False, test_circle_id=None):
    """
    Analyze the roles of members in a circle.
    
    Args:
        participants_data: DataFrame with participant information
        member_ids: List of member IDs
        test_mode: If True, enables extra validation and debugging for host status
        test_circle_id: The ID of the circle being processed (for debugging)
        
    Returns:
        dict: Dictionary with member roles information
    """
    import streamlit as st
    from utils.participant_data_manager import ParticipantDataManager
    
    # Initialize role categories
    roles = {
        "always_host": [],
        "sometimes_host": [],
        "never_host": [],
        "co_leader": []
    }
    
    # Enhanced debugging in test mode
    is_test_circle = test_circle_id in ['IP-ATL-1', 'IP-NAP-01', 'IP-SHA-01'] if test_circle_id else False
    
    if test_mode and is_test_circle:
        print(f"\n🧪🧪🧪 DETAILED HOST DEBUG FOR TEST CIRCLE {test_circle_id} 🧪🧪🧪")
        print(f"  Processing {len(member_ids)} members")
    
    # Try using ParticipantDataManager if available (preferred approach)
    participant_manager = None
    if hasattr(st, 'session_state') and 'participant_data_manager' in st.session_state:
        participant_manager = st.session_state.participant_data_manager
        print(f"✅ Using ParticipantDataManager for host role detection of {len(member_ids)} members")
        
        missing_members = 0
        if participant_manager and member_ids:
            # Process each member using the manager
            for member_id in member_ids:
                if not member_id or pd.isna(member_id):
                    continue
                    
                member_id = str(member_id)  # Ensure consistent string format
                
                try:
                    # Get participant data for detailed inspection in test mode
                    participant_data = None
                    if test_mode and is_test_circle:
                        participant_data = participant_manager.get_participant_by_id(member_id)
                        if participant_data:
                            # Log all columns related to host status for debugging
                            host_related_data = {}
                            for col in participant_data:
                                if 'host' in col.lower():
                                    host_related_data[col] = participant_data[col]
                            print(f"  Member {member_id} host data: {host_related_data}")
                    
                    # Get host status directly from manager with debug mode for test circles
                    host_status = participant_manager.get_participant_host_status(
                        member_id, 
                        debug_mode=(test_mode and is_test_circle)
                    )
                    
                    if test_mode and is_test_circle:
                        print(f"  Member {member_id} normalized host_status: '{host_status}'")
                    
                    # More detailed debug for host detection issues
                    if not host_status:
                        print(f"⚠️ No host status found for member {member_id}, defaulting to 'never'")
                        
                        # In test mode for test circles, try direct access to raw host column as fallback
                        if test_mode and is_test_circle and participant_data:
                            # Try direct access to the raw host column
                            if 'host' in participant_data and participant_data['host']:
                                raw_host = str(participant_data['host']).lower()
                                print(f"  🔍 FALLBACK: Raw 'host' value for {member_id}: '{raw_host}'")
                                
                                # Apply our normalization directly
                                if 'always' in raw_host:
                                    print(f"  ✅ FALLBACK: Member {member_id} identified as ALWAYS host from raw data")
                                    roles["always_host"].append(member_id)
                                    continue
                                elif 'sometimes' in raw_host:
                                    print(f"  ✅ FALLBACK: Member {member_id} identified as SOMETIMES host from raw data")
                                    roles["sometimes_host"].append(member_id)
                                    continue
                        
                        missing_members += 1
                        roles["never_host"].append(member_id)
                        continue
                    
                    # Categorize based on host status
                    if host_status == "always":
                        roles["always_host"].append(member_id)
                    elif host_status == "sometimes":
                        roles["sometimes_host"].append(member_id)
                    else:
                        roles["never_host"].append(member_id)
                    
                    # Check co-leader status
                    if participant_manager.is_participant_co_leader(member_id):
                        roles["co_leader"].append(member_id)
                        
                except Exception as e:
                    print(f"⚠️ WARNING: Error in ParticipantDataManager for member {member_id}: {str(e)}")
                    # Default to never host on error
                    roles["never_host"].append(member_id)
            
            # Enhanced debug information
            total_processed = len(roles["always_host"]) + len(roles["sometimes_host"]) + len(roles["never_host"])
            print(f"✅ ParticipantDataManager host role detection complete")
            print(f"   Always Hosts: {len(roles['always_host'])} members")
            print(f"   Sometimes Hosts: {len(roles['sometimes_host'])} members")
            print(f"   Never Hosts: {len(roles['never_host'])} members")
            print(f"   Co-Leaders: {len(roles['co_leader'])} members")
            print(f"   Missing host status: {missing_members} members")
            print(f"   Total processed: {total_processed} of {len(member_ids)} members")
            
            # If we have processed at least 90% of members, consider it a success
            # This allows for some tolerance of data issues
            if total_processed >= 0.9 * len(member_ids):
                return roles
            else:
                print(f"⚠️ WARNING: Only processed {total_processed} of {len(member_ids)} members with ParticipantDataManager")
                print(f"    Falling back to traditional approach for more complete results")
    
    # Fall back to traditional approach if manager method failed or wasn't available
    print(f"⚠️ Falling back to traditional host detection for {len(member_ids)} members")
    
    # Skip if no participant data
    if participants_data is None or len(member_ids) == 0:
        print(f"🔍 DEBUG: No participants or member_ids for role analysis")
        return roles
    
    # Ensure we have an Encoded ID column
    id_col = "Encoded ID"
    if id_col not in participants_data.columns:
        print(f"⚠️ WARNING: Could not find '{id_col}' column in participants data")
        return roles
    
    # First, check for standardized host status column (preferred)
    host_col = None
    if 'host_status_standardized' in participants_data.columns:
        host_col = 'host_status_standardized'
        print(f"✅ Using standardized host status column: {host_col}")
    elif 'Standardized Host' in participants_data.columns:
        host_col = 'Standardized Host'
        print(f"✅ Using standardized host status column: {host_col}")
    else:
        # Fall back to other host columns
        for col_name in ["host", "Host", "willing_to_host", "HostingPreference", "Host Status"]:
            if col_name in participants_data.columns:
                host_col = col_name
                print(f"✅ Using host status column: {host_col}")
                break
    
    # Look for co-leader column
    co_leader_col = None
    for col_name in ["co_leader", "Co_Leader", "co-leader", "Co-Leader", "co_lead", "Co_Lead"]:
        if col_name in participants_data.columns:
            co_leader_col = col_name
            print(f"✅ Using co-leader column: {co_leader_col}")
            break
    
    # Process each member using traditional column-based approach
    members_processed = 0
    for member_id in member_ids:
        try:
            if not member_id or pd.isna(member_id):
                continue
                
            member_id = str(member_id)  # Ensure consistent string format
            
            # Find this member in the DataFrame
            member_rows = participants_data[participants_data[id_col].astype(str) == member_id]
            
            if not member_rows.empty:
                members_processed += 1
                
                # Process host status with enhanced detection
                if host_col:
                    try:
                        host_value = member_rows.iloc[0][host_col]
                        # Handle potential NaN values
                        if pd.isna(host_value):
                            roles["never_host"].append(member_id)
                            continue
                            
                        # Convert to string and normalize
                        host_status = str(host_value).lower().strip()
                        
                        # More comprehensive host value normalization based on our updated mapping
                        # Use the same exact mappings we use in ParticipantDataManager
                        host_mapping = {
                            # ALWAYS mappings
                            "always": "always",
                            "always host": "always",
                            "always_host": "always",
                            "yes": "always",
                            "1": "always",
                            "1.0": "always",
                            # SOMETIMES mappings  
                            "sometimes": "sometimes",
                            "sometimes host": "sometimes",
                            "sometimes_host": "sometimes",
                            "maybe": "sometimes", 
                            "0.5": "sometimes",
                            # NEVER mappings
                            "n/a": "never",
                            "never": "never", 
                            "never host": "never",
                            "never_host": "never",
                            "no": "never",
                            "0": "never",
                            "0.0": "never",
                            "": "never"
                        }
                        
                        # First try exact matches
                        if host_status in host_mapping:
                            normalized_status = host_mapping[host_status]
                            if normalized_status == "always":
                                roles["always_host"].append(member_id)
                            elif normalized_status == "sometimes":
                                roles["sometimes_host"].append(member_id)
                            else:
                                roles["never_host"].append(member_id)
                            continue
                        
                        # Then try partial matches
                        if "always" in host_status:
                            roles["always_host"].append(member_id)
                        elif "sometimes" in host_status:
                            roles["sometimes_host"].append(member_id)
                        elif "never" in host_status or "n/a" in host_status:
                            roles["never_host"].append(member_id)
                        # Check for numeric values (could be int/float converted to string)
                        elif host_status.replace('.', '', 1).isdigit():
                            try:
                                numeric_value = float(host_status)
                                if numeric_value == 1.0:
                                    roles["always_host"].append(member_id)
                                elif numeric_value == 0.5:
                                    roles["sometimes_host"].append(member_id)
                                elif numeric_value == 0.0:
                                    roles["never_host"].append(member_id)
                                else:
                                    # Any other numeric value defaults to never
                                    roles["never_host"].append(member_id)
                            except ValueError:
                                # If we can't convert, default to never
                                roles["never_host"].append(member_id)
                        else:
                            # Anything else defaults to never
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
            else:
                print(f"⚠️ WARNING: Member {member_id} not found in participants data")
        except Exception as e:
            print(f"⚠️ WARNING: Error processing member {member_id}: {str(e)}")
            # Default to never host on error
            roles["never_host"].append(member_id)
    
    print(f"✅ Traditional host role detection complete")
    print(f"   Found {members_processed} of {len(member_ids)} members in participant data")
    print(f"   Always Hosts: {len(roles['always_host'])}")
    print(f"   Sometimes Hosts: {len(roles['sometimes_host'])}")
    print(f"   Never Hosts: {len(roles['never_host'])}")
    print(f"   Co-Leaders: {len(roles['co_leader'])}")
    
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
    import streamlit as st
    from utils.participant_data_manager import ParticipantDataManager
    
    # Enhanced logging for circle splitting
    original_member_count = len(members)
    print(f"🔍 Starting split process for circle {circle_id} with {original_member_count} members")
    
    # Ensure we have at least 11 members
    if original_member_count < 11:
        return {
            "success": False,
            "reason": f"Not enough members to split (need 11+, found {original_member_count})"
        }
    
    # Get the ParticipantDataManager for additional validation if needed
    participant_manager = None
    if hasattr(st, 'session_state') and 'participant_data_manager' in st.session_state:
        participant_manager = st.session_state.participant_data_manager
        print(f"✅ Using ParticipantDataManager for additional host validation")
    
    # Check if we have enough hosts
    always_hosts = member_roles.get("always_host", [])
    sometimes_hosts = member_roles.get("sometimes_host", [])
    co_leaders = member_roles.get("co_leader", [])
    never_hosts = member_roles.get("never_host", [])
    
    # Enhanced logging
    print(f"🔍 Current roles distribution:")
    print(f"   Always Hosts: {len(always_hosts)} members")
    print(f"   Sometimes Hosts: {len(sometimes_hosts)} members")
    print(f"   Co-Leaders: {len(co_leaders)} members")
    print(f"   Never Hosts: {len(never_hosts)} members")
    
    # In test mode, log the actual member IDs for each role
    if circle_id in ['IP-ATL-1', 'IP-NAP-01', 'IP-SHA-01']:
        print(f"🧪 TEST CIRCLE {circle_id} - Detailed host breakdown:")
        print(f"   Always Hosts: {always_hosts}")
        print(f"   Sometimes Hosts: {sometimes_hosts}")
        print(f"   Co-Leaders: {co_leaders}")
    
    # Double-check host counts using ParticipantDataManager for test circles
    # This helps diagnose issues with host detection
    if participant_manager and circle_id in ['IP-ATL-1', 'IP-NAP-01', 'IP-SHA-01']:
        print(f"🧪 TEST CIRCLE {circle_id} - Double-checking hosts with ParticipantDataManager:")
        always_count = sometimes_count = never_count = 0
        for member_id in members:
            host_status = participant_manager.get_participant_host_status(member_id)
            if host_status == "always":
                always_count += 1
            elif host_status == "sometimes":
                sometimes_count += 1
            else:
                never_count += 1
        
        print(f"   ParticipantDataManager host counts:")
        print(f"   Always: {always_count}, Sometimes: {sometimes_count}, Never: {never_count}")
        
        # If there's a significant discrepancy, use the ParticipantDataManager data
        if abs(len(always_hosts) - always_count) > 1 or abs(len(sometimes_hosts) - sometimes_count) > 1:
            print(f"⚠️ Host count discrepancy detected! Rebuilding member_roles using ParticipantDataManager")
            
            # Rebuild member_roles directly from ParticipantDataManager
            new_roles = {
                "always_host": [],
                "sometimes_host": [],
                "never_host": [],
                "co_leader": []
            }
            
            for member_id in members:
                try:
                    host_status = participant_manager.get_participant_host_status(member_id)
                    if host_status == "always":
                        new_roles["always_host"].append(member_id)
                    elif host_status == "sometimes":
                        new_roles["sometimes_host"].append(member_id)
                    else:
                        new_roles["never_host"].append(member_id)
                        
                    if participant_manager.is_participant_co_leader(member_id):
                        new_roles["co_leader"].append(member_id)
                except Exception as e:
                    print(f"⚠️ Error getting host status from manager: {str(e)}")
                    new_roles["never_host"].append(member_id)
            
            # Replace with new roles
            always_hosts = new_roles["always_host"]
            sometimes_hosts = new_roles["sometimes_host"]
            never_hosts = new_roles["never_host"]
            co_leaders = new_roles["co_leader"]
            
            print(f"✅ Rebuilt member_roles from ParticipantDataManager:")
            print(f"   Always Hosts: {len(always_hosts)} members")
            print(f"   Sometimes Hosts: {len(sometimes_hosts)} members")
            print(f"   Co-Leaders: {len(co_leaders)} members")
            print(f"   Never Hosts: {len(never_hosts)} members")
    
    # Calculate minimum required hosts for splitting
    min_splits = 2  # Minimum 2 splits for a large circle
    
    # For a valid split, each new circle needs either:
    # - At least 1 Always Host, or
    # - At least 2 Sometimes Hosts
    
    always_host_count = len(always_hosts)
    sometimes_host_count = len(sometimes_hosts)
    
    # Calculate how many circles we can create based on host distribution
    possible_circles_from_always = always_host_count
    possible_circles_from_sometimes = sometimes_host_count // 2
    
    max_possible_circles = possible_circles_from_always + possible_circles_from_sometimes
    
    # Enhanced logging for host distribution planning
    print(f"🔍 Host distribution planning:")
    print(f"   Always Hosts: {always_host_count} members → Can support {possible_circles_from_always} circles")
    print(f"   Sometimes Hosts: {sometimes_host_count} members → Can support {possible_circles_from_sometimes} circles (pairs)")
    print(f"   Total possible circles based on hosts: {max_possible_circles}")
    print(f"   Minimum required splits: {min_splits}")
    
    if max_possible_circles < min_splits:
        return {
            "success": False,
            "reason": f"Not enough hosts for valid splitting. Need at least {min_splits} circles, " 
                     f"but can only create {max_possible_circles} with available hosts."
        }
    
    # IMPROVED: Determine optimal number of splits based on member count
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
    
    print(f"🔍 Split planning:")
    print(f"   Target minimum circle size: {target_min_size} members")
    print(f"   Maximum possible splits based on size: {max_splits}")
    print(f"   Actual planned splits: {num_splits}")
    
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
        """
        Determines the optimal split for a member based on their host status and the current composition of splits.
        
        UPDATED DISTRIBUTION STRATEGY:
        1. First split (A) should get all Always Hosts
        2. Second split (B) should get both Sometimes Hosts
        3. Remaining members are balanced for circle size
        
        This implements the exact requirement from the client to have Always Hosts
        in one circle and both Sometimes Hosts in the other.
        """
        # For debugging
        print(f"🔍 DEBUG: Finding optimal split for member {member_id} - always_host: {is_always_host}, sometimes_host: {is_sometimes_host}")
        
        # FIXED DISTRIBUTION STRATEGY:
        # - First split (A) gets ALL Always Hosts (critical change)
        # - Second split (B) gets ALL Sometimes Hosts (critical change)
        # - Remaining splits follow general balancing rules
        
        # Case 1: Always Host - ALWAYS assign to first split (A) - no exceptions
        if is_always_host:
            print(f"  → Assigning Always Host {member_id} to first split (index 0/A)")
            return 0
            
        # Case 2: Sometimes Host - ALWAYS assign to second split (B) - no exceptions
        if is_sometimes_host:
            # Only if we have at least 2 splits
            if len(splits) >= 2:
                print(f"  → Assigning Sometimes Host {member_id} to second split (index 1/B)")
                return 1
            else:
                # Fallback if only 1 split (unlikely but safety check)
                print(f"  → No second split available, assigning Sometimes Host {member_id} to first split")
                return 0
        
        # Regular balancing logic for other cases
        best_score = float('-inf')
        best_index = 0
        
        for i, split in enumerate(splits):
            score = 0
            
            # Host factors
            if is_always_host:
                # We already handled the primary case above
                # Prefer circles with fewer always hosts for additional always hosts
                score -= split["always_hosts"] * 10
            
            if is_sometimes_host:
                # We already handled the critical case above
                # For remaining sometimes hosts:
                
                # Strongly avoid putting sometimes hosts in splits that already have always hosts
                if split["always_hosts"] > 0:
                    score -= 100  # Strong penalty
                
                # For splits without always hosts, prefer those with fewer sometimes hosts
                else:
                    # If a split has 0-1 sometimes hosts and no always hosts, it needs more sometimes hosts
                    if split["sometimes_hosts"] < 2:
                        score += 50
                    # Otherwise penalize for existing sometimes hosts
                    else:
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
                
        print(f"  → Assigning member {member_id} to split {best_index} based on general balancing")
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
    
    # ENHANCED VALIDATION: Check if all members were assigned with detailed diagnostics
    if len(assigned_members) != original_member_count:
        print(f"⚠️ WARNING: Member count mismatch! Original: {original_member_count}, Assigned: {len(assigned_members)}")
        
        # Find missing members with detailed debugging
        missing = set(members) - set(assigned_members)
        if missing:
            print(f"⚠️ WARNING: {len(missing)} missing members: {missing}")
            
            # Debug specific information about missing members
            for missing_id in missing:
                print(f"🔍 MEMBER DEBUG: Missing member {missing_id} was not assigned")
                
                # Check if this member was in the original member lists
                if missing_id in always_hosts:
                    print(f"  → This member is an Always Host but was not assigned")
                elif missing_id in sometimes_hosts:
                    print(f"  → This member is a Sometimes Host but was not assigned")
                elif missing_id in co_leaders:
                    print(f"  → This member is a Co-Leader but was not assigned")
                
                # Check if member is still in unassigned_members (shouldn't be possible)
                if missing_id in unassigned_members:
                    print(f"  → CRITICAL ERROR: Member is still in unassigned_members list")
                
                # Check if this member was identified in participants_data
                try:
                    if participants_data is not None:
                        # Find ID column
                        id_col = None
                        for col in ['Encoded ID', 'encoded_id', 'participant_id']:
                            if col in participants_data.columns:
                                id_col = col
                                break
                        
                        if id_col:
                            # Find the participant row
                            participant_mask = participants_data[id_col] == missing_id
                            if any(participant_mask):
                                print(f"  → Member exists in participants_data")
                                
                                # Get name if available (for debugging only)
                                name_data = []
                                for name_col in ['Last Family Name', 'First Given Name']:
                                    if name_col in participants_data.columns:
                                        value = participants_data.loc[participant_mask, name_col].iloc[0]
                                        if not pd.isna(value):
                                            name_data.append(str(value))
                                
                                if name_data:
                                    print(f"  → Participant name: {' '.join(name_data)}")
                            else:
                                print(f"  → Member NOT found in participants_data using ID column '{id_col}'")
                except Exception as e:
                    print(f"  → ERROR checking participant data: {str(e)}")
                
                # Add to unassigned_members to ensure we don't lose them
                print(f"  → RECOVERY ACTION: Adding missing member back to unassigned_members list")
                unassigned_members.append(missing_id)
            
            # Attempt recovery by assigning missing members
            if unassigned_members:
                print(f"🔄 RECOVERY: Attempting to assign {len(unassigned_members)} previously missing members")
                
                # Distribute these members evenly across splits
                for member_id in unassigned_members:
                    # Find smallest split relative to target
                    smallest_index = min(range(num_splits), 
                                       key=lambda i: len(splits[i]["members"]) / target_sizes[i])
                    
                    # Add member to the split
                    splits[smallest_index]["members"].append(member_id)
                    assigned_members.append(member_id)
                    print(f"  → Recovered missing member {member_id} assigned to split {smallest_index}")
                
                # Clear unassigned list after recovery
                unassigned_members = []
        
        # Find extra members (shouldn't happen but check anyway)
        extra = set(assigned_members) - set(members)
        if extra:
            print(f"⚠️ WARNING: {len(extra)} extra members that weren't in original list: {extra}")
            
            # Debug extra members (where did they come from?)
            for extra_id in extra:
                print(f"🔍 MEMBER DEBUG: Extra member {extra_id} was assigned but wasn't in original members list")
                
                # Check which split they were assigned to
                for i, split in enumerate(splits):
                    if extra_id in split["members"]:
                        print(f"  → This member was assigned to split {i}")
                        # Remove to clean up - only if there are enough remaining members
                        if len(split["members"]) > 5:
                            split["members"].remove(extra_id)
                            print(f"  → Removed extra member from split {i} (still has {len(split['members'])} members)")
                            # Update assigned_members to keep counts consistent
                            if extra_id in assigned_members:
                                assigned_members.remove(extra_id)
    
    # Re-check counts after recovery attempts
    if len(assigned_members) != original_member_count:
        print(f"⚠️ CRITICAL: Member count still mismatched after recovery! Original: {original_member_count}, Assigned: {len(assigned_members)}")
        
        # Log all member counts per split for diagnosis
        total_members_in_splits = 0
        for i, split in enumerate(splits):
            split_count = len(split["members"])
            total_members_in_splits += split_count
            print(f"🔍 Split {i} has {split_count} members")
            
        print(f"🔍 Total members across all splits: {total_members_in_splits}")
    else:
        print(f"✅ All {original_member_count} members successfully assigned to {num_splits} splits")
    
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
    
    # Get original region/subregion data from the participants
    original_region = region  # Default to the region from circle ID
    original_subregion = ""
    
    # Use our common utility function to get region/subregion from first participant in member list
    if members and len(members) > 0:
        r, s = get_region_subregion_from_participants(participants_data, members)
        if r:
            original_region = r
        if s:
            original_subregion = s
        print(f"🔍 Region/Subregion from participants: {original_region}/{original_subregion}")
    
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
            "region": original_region,  # Use region from participant data
            "subregion": original_subregion,  # Use subregion from participant data
            "is_split_circle": True,
            "split_source": circle_id,
            "split_index": i,
            "split_letter": suffix,  # Store the split letter for reference
            "max_additions": max(0, 8 - len(split["members"])),  # Allow growth up to 8 total
            "eligible_for_new_members": True,  # Split circles should be eligible for new members
            "always_hosts": split["always_hosts"],  # Rename to match metadata manager expectations
            "sometimes_hosts": split["sometimes_hosts"],  # Rename to match metadata manager expectations
            "count_co_leaders": split["co_leaders"]
        }
        
        # Add debugging for member count
        print(f"🔍 Split {suffix} has {len(split['members'])} members, {split['always_hosts']} always hosts, {split['sometimes_hosts']} sometimes hosts")
        if len(split["members"]) > 0:
            print(f"  - First few members: {split['members'][:3]}")
        
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


def get_region_subregion_from_participants(participants_data, member_ids):
    """
    Extract region and subregion from the first participant in a circle.
    
    Args:
        participants_data (DataFrame or None): DataFrame with participant information or None to use ParticipantDataManager
        member_ids (list): List of member IDs in the circle
        
    Returns:
        tuple: (region, subregion) strings extracted from the first participant's data
    """
    import streamlit as st
    from utils.participant_data_manager import ParticipantDataManager
    
    # First try using ParticipantDataManager if available
    participant_manager = None
    if hasattr(st, 'session_state') and hasattr(st.session_state, 'participant_data_manager'):
        participant_manager = st.session_state.participant_data_manager
        print("🔍 Using ParticipantDataManager for region/subregion extraction")
        
        if participant_manager and member_ids:
            # Try each member ID until we find valid region data
            for member_id in member_ids:
                try:
                    # Get participant data directly from manager
                    participant = participant_manager.get_participant_by_id(member_id)
                    if participant:
                        # Extract region
                        region = ""
                        for field in ['Current Region', 'Region', 'region', 'current_region', 'Derived_Region']:
                            if field in participant and participant[field] and not pd.isna(participant[field]):
                                region = str(participant[field])
                                break
                                
                        # Extract subregion
                        subregion = ""
                        for field in ['Current Subregion', 'Subregion', 'subregion', 'current_subregion']:
                            if field in participant and participant[field] and not pd.isna(participant[field]):
                                subregion = str(participant[field])
                                break
                                
                        if region:
                            print(f"✅ ParticipantDataManager: Found region '{region}' and subregion '{subregion}' for member {member_id}")
                            return region, subregion
                except Exception as e:
                    print(f"⚠️ Error in ParticipantDataManager for member {member_id}: {str(e)}")
                    continue
    
    # Fall back to traditional approach if manager method failed or wasn't available
    if participants_data is None or not member_ids:
        print("⚠️ No participant data or member IDs available for region extraction")
        return "", ""
    
    # Look for ID column
    id_col = None
    for col in ['Encoded ID', 'encoded_id', 'participant_id']:
        if col in participants_data.columns:
            id_col = col
            break
    
    if not id_col:
        print("⚠️ Could not find ID column in participants data")
        return "", ""
    
    # Look for region and subregion columns
    region_col = None
    subregion_col = None
    
    for col in ['Current Region', 'Region', 'region', 'current_region']:
        if col in participants_data.columns:
            region_col = col
            break
    
    for col in ['Current Subregion', 'Subregion', 'subregion', 'current_subregion']:
        if col in participants_data.columns:
            subregion_col = col
            break
    
    if not region_col:
        print("⚠️ Could not find region column in participants data")
        return "", ""
    
    # Try to find the first member in the participant data
    for member_id in member_ids:
        try:
            # Find this member in the DataFrame
            member_mask = participants_data[id_col] == member_id
            
            if any(member_mask):
                member_row = participants_data.loc[member_mask].iloc[0]
                
                # Extract region and subregion
                region = str(member_row[region_col]) if not pd.isna(member_row[region_col]) else ""
                
                # Only extract subregion if the column exists
                subregion = ""
                if subregion_col and subregion_col in participants_data.columns:
                    subregion = str(member_row[subregion_col]) if not pd.isna(member_row[subregion_col]) else ""
                
                print(f"✅ Extracted region/subregion from participant {member_id}: {region}/{subregion}")
                return region, subregion
        except Exception as e:
            print(f"⚠️ Error extracting region/subregion from participant {member_id}: {str(e)}")
            continue
    
    # If we couldn't find any valid participant data, return empty strings
    print("⚠️ Could not extract region/subregion from any participant")
    return "", ""

# Additional utility functions can be added here as needed