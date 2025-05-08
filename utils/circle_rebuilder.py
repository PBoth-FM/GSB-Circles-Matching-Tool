import pandas as pd
import numpy as np

def rebuild_circle_member_lists(circles_df, participants_df):
    """
    Reconstruct complete member lists for all circles directly from participant data.
    This function addresses the disconnect between member_count and actual members lists.
    
    Args:
        circles_df: DataFrame containing circle data
        participants_df: DataFrame containing participant data with circle assignments
        
    Returns:
        Updated DataFrame with rebuilt member lists for each circle
    """
    print("\n🔄 REBUILDING CIRCLE MEMBER LISTS: Starting comprehensive rebuild...")
    
    # Create a deep copy to avoid modifying the original
    updated_circles = circles_df.copy() if isinstance(circles_df, pd.DataFrame) else pd.DataFrame()
    if updated_circles.empty:
        print("⚠️ WARNING: Input circles DataFrame is empty")
    
    # Ensure the participants DataFrame is valid
    if participants_df is None or len(participants_df) == 0:
        print("⚠️ WARNING: No participant data available to rebuild circle member lists")
        return updated_circles
        
    # Find the column that contains circle assignments
    # Use Current_Circle_ID as the primary column (as specified), then fall back to alternatives
    circle_col = None
    preferred_cols = ['Current_Circle_ID', 'assigned_circle', 'circle_id', 'Circle ID', 'proposed_NEW_circles_id']
    
    for col in preferred_cols:
        if col in participants_df.columns:
            circle_col = col
            print(f"✅ Found circle assignment column: '{circle_col}'")
            # Show how many participants have assignments in this column
            assigned_count = participants_df[~participants_df[circle_col].isna()].shape[0]
            total_count = len(participants_df)
            print(f"✅ Found {assigned_count} of {total_count} participants with circle assignments in column '{circle_col}'")
            break
    
    if not circle_col:
        print("⚠️ WARNING: Could not find circle assignment column in participants data")
        print(f"⚠️ Available columns: {participants_df.columns.tolist()}")
        return updated_circles  # No column found to rebuild memberships
    
    # Track which circles were updated and collect statistics
    circles_updated = 0
    all_circle_members = 0
    member_count_fixes = 0
    
    # Step 1: Identify ALL circles, including both original and split circles
    all_circle_ids = set()
    if 'circle_id' in updated_circles.columns:
        all_circle_ids = set(updated_circles['circle_id'].dropna().unique())
        print(f"🔍 Found {len(all_circle_ids)} circles in the input DataFrame")
        
        # Debug: Check if split circles exist in the input DataFrame
        split_circles = [cid for cid in all_circle_ids if 'SPLIT' in cid]
        if split_circles:
            print(f"🔍 Input DataFrame contains {len(split_circles)} split circles: {split_circles[:5]}")
    
    # Add circles from participant data
    participant_circles = set()
    if circle_col in participants_df.columns:
        participant_circles = set(participants_df[circle_col].dropna().unique())
        print(f"🔍 Found {len(participant_circles)} unique circle assignments in participant data")
        
        # Debug: Check if split circles exist in participant data
        split_circles_in_data = [cid for cid in participant_circles if isinstance(cid, str) and 'SPLIT' in cid]
        if split_circles_in_data:
            print(f"🔍 Participant data contains {len(split_circles_in_data)} split circles: {split_circles_in_data[:5]}")
    
    # Find circles that exist in participant data but not in circle DataFrame
    new_circles = participant_circles - all_circle_ids
    if new_circles:
        print(f"⚠️ Found {len(new_circles)} circles in participant data that aren't in circles DataFrame")
        print(f"Sample new circles: {list(new_circles)[:5]}")
    
    # Step 2: Create a mapping of all circle IDs to their members
    circle_member_map = {}
    
    # First, process all participants to map them to circles
    print(f"🔄 Building circle membership map from {len(participants_df)} participants...")
    for _, participant in participants_df.iterrows():
        if circle_col in participant and not pd.isna(participant[circle_col]) and 'Encoded ID' in participant and not pd.isna(participant['Encoded ID']):
            circle_id = str(participant[circle_col])
            member_id = str(participant['Encoded ID'])
            
            if circle_id not in circle_member_map:
                circle_member_map[circle_id] = []
            
            # Add the member to this circle
            if member_id not in circle_member_map[circle_id]:
                circle_member_map[circle_id].append(member_id)
    
    print(f"✅ Built membership map for {len(circle_member_map)} circles")
    
    # Step 3: Update the member lists and counts for all circles in the DataFrame
    # For each circle in the DataFrame
    for idx, circle in updated_circles.iterrows():
        circle_id = circle.get('circle_id', None)
        if not circle_id or pd.isna(circle_id):
            continue
        
        # Convert to string if needed
        circle_id = str(circle_id)
        
        # Get members for this circle from our mapping
        if circle_id in circle_member_map:
            # Members found in participant data
            member_ids = circle_member_map[circle_id]
            
            # Debug for specific test circles
            if circle_id in ['IP-ATL-1', 'IP-NAP-01', 'IP-SHA-01']:
                print(f"🔍 Circle {circle_id}: Found {len(member_ids)} members using column '{circle_col}'")
                if len(member_ids) > 0:
                    print(f"🔍 First few members: {member_ids[:3]}")
                
                # Check for member count mismatch
                if 'member_count' in updated_circles.columns:
                    current_count = updated_circles.at[idx, 'member_count']
                    if current_count != len(member_ids):
                        print(f"⚠️ Member count mismatch for {circle_id}: stored={current_count}, found={len(member_ids)}")
            
            # Update the circle
            updated_circles.at[idx, 'members'] = member_ids
            
            # CRITICAL FIX: Always update member_count to match rebuilt members list
            current_count = updated_circles.at[idx, 'member_count'] if 'member_count' in updated_circles.columns else 0
            if current_count != len(member_ids):
                print(f"🔄 Updating {circle_id} member count: {current_count} → {len(member_ids)}")
                updated_circles.at[idx, 'member_count'] = len(member_ids)
                member_count_fixes += 1
            
            circles_updated += 1
            all_circle_members += len(member_ids)
        else:
            # No members found for this circle - set empty list and zero count
            # This can happen if a circle exists in the circles DataFrame but no participants are assigned to it
            updated_circles.at[idx, 'members'] = []
            
            # Only update member_count if it's not already 0
            if 'member_count' in updated_circles.columns and updated_circles.at[idx, 'member_count'] != 0:
                print(f"⚠️ Circle {circle_id} has no members assigned - setting member_count to 0")
                updated_circles.at[idx, 'member_count'] = 0
                member_count_fixes += 1
    
    # Step 4: Add any circles that exist in participant data but aren't in the DataFrame
    # This ensures we don't miss any circles that exist only in the participant data
    new_circle_rows = []
    for circle_id in new_circles:
        if circle_id in circle_member_map:
            member_ids = circle_member_map[circle_id]
            
            # Debug info
            print(f"🔄 Adding new circle {circle_id} with {len(member_ids)} members")
            
            # Create a new row for this circle with the essential fields
            new_row = {
                'circle_id': circle_id,
                'members': member_ids,
                'member_count': len(member_ids)
            }
            
            # Add to our collection of new rows
            new_circle_rows.append(new_row)
            circles_updated += 1
            all_circle_members += len(member_ids)
    
    # Add all new circles at once if any were found
    if new_circle_rows:
        print(f"🔄 Adding {len(new_circle_rows)} new circles to the DataFrame")
        new_df = pd.DataFrame(new_circle_rows)
        updated_circles = pd.concat([updated_circles, new_df], ignore_index=True)
    
    # Log member count distribution
    if 'member_count' in updated_circles.columns:
        value_counts = updated_circles['member_count'].value_counts().sort_index()
        print(f"\n🔍 MEMBER COUNT DISTRIBUTION:")
        for count, occurrences in value_counts.items():
            print(f"  {count} members: {occurrences} circles")
        
        # Special focus on large circles (those with 11+ members)
        large_circles = updated_circles[updated_circles['member_count'] >= 11]
        if not large_circles.empty:
            print(f"\n🔍 LARGE CIRCLES (11+ members): {len(large_circles)} circles")
            for _, circle in large_circles.iterrows():
                circle_id = circle.get('circle_id', 'Unknown')
                member_count = circle.get('member_count', 0)
                print(f"  {circle_id}: {member_count} members")
    
    # Special case debugging: Check specific circles of interest
    if 'Current_Circle_ID' in participants_df.columns:
        print(f"\n🔍 DETAILED MEMBER ANALYSIS FOR TEST CIRCLES:")
        for test_id in ['IP-ATL-1', 'IP-NAP-01', 'IP-SHA-01']:
            # Find circle in updated DataFrame
            circle_row = updated_circles[updated_circles['circle_id'] == test_id]
            if not circle_row.empty:
                stored_count = circle_row.iloc[0].get('member_count', 0)
                members_list = circle_row.iloc[0].get('members', [])
                members_length = len(members_list) if isinstance(members_list, list) else 0
                
                print(f"\n🔍 DETAILED ANALYSIS OF {test_id}:")
                print(f"  Member count in DataFrame: {stored_count}")
                print(f"  Actual members list length: {members_length}")
                
                # Check if this matches what's in participant data
                if circle_col in participants_df.columns:
                    assigned_members = participants_df[participants_df[circle_col] == test_id]
                    print(f"  Total participants assigned in data: {len(assigned_members)}")
                    
                    # Status breakdown
                    if 'Status' in participants_df.columns:
                        continuing = len(assigned_members[assigned_members['Status'] == 'CONTINUING'])
                        new_members = len(assigned_members[assigned_members['Status'] == 'NEW'])
                        other = len(assigned_members) - continuing - new_members
                        print(f"  Status breakdown: CONTINUING={continuing}, NEW={new_members}, OTHER={other}")
                    
                    # Host status breakdown
                    if 'host_status_standardized' in participants_df.columns:
                        host_breakdown = assigned_members['host_status_standardized'].value_counts()
                        print(f"  Host status breakdown: {dict(host_breakdown)}")
            else:
                print(f"  ⚠️ Test circle {test_id} not found in updated circles DataFrame")
    
    print(f"✅ Successfully rebuilt member lists for {circles_updated} circles with a total of {all_circle_members} members")
    print(f"✅ Fixed {member_count_fixes} member count inconsistencies")
    
    return updated_circles