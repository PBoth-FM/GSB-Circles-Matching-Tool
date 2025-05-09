"""
Optimizer integration module for coordinating the circle splitting process with optimization.

This module serves as a bridge between the circle splitting process and the optimization
algorithm. It ensures that circle splitting happens before optimization, allowing split
circles to participate in the optimization process for receiving new members.
"""

import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Any, Optional

# Import circle splitting functionality
from modules.circle_splitter import split_large_circles
from utils.circle_metadata_manager import get_manager_from_session_state, initialize_or_update_manager

def preprocess_circles_for_optimization(circles_data, participants_data):
    """
    Preprocess circles before optimization, including splitting large circles.
    
    This function serves as the central coordinator for preprocessing steps that
    should happen before the optimization algorithm runs. It ensures that large
    circles are split before optimization so that new participants can be assigned
    to the split circles during the optimization process.
    
    Args:
        circles_data: DataFrame or list of dictionaries containing circle information
        participants_data: DataFrame containing participant information
        
    Returns:
        tuple: (
            updated_circles: DataFrame or list with preprocessed circle data,
            preprocessing_summary: Dictionary with details about processing steps
        )
    """
    preprocessing_summary = {
        "steps_performed": [],
        "split_circle_summary": None,
        "eligible_split_circles": 0
    }
    
    # Step 1: Split large circles - using third return value for updated participants
    print("\n🔄 PREPROCESSING: Starting circle splitting process")
    print("  Splitting circles with 11+ members before optimization to make split circles available for new members")
    updated_circles, split_summary, updated_participants = split_large_circles(
        circles_data, 
        participants_data,
        test_mode=False  # Ensure we're not in test mode for production use
    )
    
    # Store updated participants if provided
    if updated_participants is not None and not updated_participants.empty:
        print(f"✅ Using updated participants data with split circle assignments ({len(updated_participants)} participants)")
        st.session_state.processed_data = updated_participants
    
    preprocessing_summary["steps_performed"].append("split_large_circles")
    preprocessing_summary["split_circle_summary"] = split_summary
    
    # Store the split summary in session state for the UI to use
    if "split_circle_summary" not in st.session_state or st.session_state.split_circle_summary != split_summary:
        st.session_state.split_circle_summary = split_summary
        print("✅ Stored split circle summary in session state")
        
    # Log detailed information about the split results
    if split_summary:
        large_circles_found = split_summary.get('total_large_circles_found', 0)
        circles_split = split_summary.get('total_circles_successfully_split', 0)
        new_circles_created = split_summary.get('total_new_circles_created', 0)
        
        print(f"🔄 Circle splitting results:")
        print(f"  - Large circles found: {large_circles_found}")
        print(f"  - Circles successfully split: {circles_split}")
        print(f"  - New split circles created: {new_circles_created}")
        
        if new_circles_created > 0:
            print(f"🎯 Successfully split {circles_split} circles into {new_circles_created} new circles")
        else:
            print("ℹ️ No circles needed splitting")
    
    # Step 2: Update the CircleMetadataManager with split circle information
    if split_summary and split_summary.get('total_circles_successfully_split', 0) > 0:
        print(f"🔄 PREPROCESSING: Updating metadata manager with {split_summary.get('total_circles_successfully_split', 0)} split circles")
        update_metadata_manager_with_splits(split_summary)
        preprocessing_summary["steps_performed"].append("update_metadata_manager_with_splits")
    
    # Step 3: Ensure the split circles have max_additions = 8 (minus current member count)
    # Get the metadata manager to work with
    manager = get_manager_from_session_state(st.session_state)
    if manager and hasattr(manager, 'split_circles') and manager.split_circles:
        print(f"\n🔄 PREPROCESSING: Setting max_additions for {len(manager.split_circles)} split circles")
        
        # Get all split circle IDs
        split_circle_ids = list(manager.split_circles.keys())
        eligible_count = 0
        
        # Ensure each split circle has max_additions set to allow up to 8 members
        for split_id in split_circle_ids:
            if split_id in manager.circles:
                circle_data = manager.circles[split_id]
                member_count = circle_data.get('member_count', 0)
                
                # Calculate max_additions to allow growth to 8 members
                max_additions = max(0, 8 - member_count)
                circle_data['max_additions'] = max_additions
                
                # Set is_eligible to True if there's room for more members
                circle_data['is_eligible'] = max_additions > 0
                
                # Update the circle data in the manager
                manager.circles[split_id] = circle_data
                
                # Log eligibility
                if max_additions > 0:
                    eligible_count += 1
                    print(f"  ✅ Split circle {split_id} can take up to {max_additions} new members")
                else:
                    print(f"  ℹ️ Split circle {split_id} is already at capacity with {member_count} members")
        
        preprocessing_summary["eligible_split_circles"] = eligible_count
        print(f"✅ {eligible_count} of {len(split_circle_ids)} split circles are eligible for new members")
        
        # Step 4: Ensure the updated circle data is reflected in the updated_circles
        # This section applies updates to the circles in the format expected by the optimizer
        if updated_circles is not None:
            print("\n🔄 PREPROCESSING: Applying split circle updates to circles data")
            
            # Handle DataFrame or list format
            if isinstance(updated_circles, pd.DataFrame):
                # DataFrame format - update using the index
                for split_id in split_circle_ids:
                    if split_id in updated_circles['circle_id'].values:
                        circle_idx = updated_circles.index[updated_circles['circle_id'] == split_id].tolist()[0]
                        
                        # Update max_additions
                        circle_data = manager.circles.get(split_id, {})
                        updated_circles.at[circle_idx, 'max_additions'] = circle_data.get('max_additions', 0)
                        
                        # Update active status to ensure it's considered
                        updated_circles.at[circle_idx, 'active'] = True
                        
                        # Update split status flag
                        updated_circles.at[circle_idx, 'is_split_circle'] = True
                        
                        # Log the update
                        print(f"  ✅ Updated DataFame entry for {split_id} with max_additions={circle_data.get('max_additions', 0)}")
            else:
                # List format - update by iterating
                for i, circle in enumerate(updated_circles):
                    if circle.get('circle_id') in split_circle_ids:
                        # Get updated data from manager
                        split_id = circle.get('circle_id')
                        circle_data = manager.circles.get(split_id, {})
                        
                        # Update max_additions
                        updated_circles[i]['max_additions'] = circle_data.get('max_additions', 0)
                        
                        # Update active status to ensure it's considered
                        updated_circles[i]['active'] = True
                        
                        # Update split status flag
                        updated_circles[i]['is_split_circle'] = True
                        
                        # Log the update
                        print(f"  ✅ Updated list entry for {split_id} with max_additions={circle_data.get('max_additions', 0)}")
    
    # Step 5: Pre-optimization validation to verify split circles are prepared correctly
    print("\n🔍 PREPROCESSING: Validating split circle preparation")
    
    # Record validation results
    validation_results = {
        "original_circles_inactive": True,
        "split_circles_active": True,
        "split_circles_have_correct_max_additions": True,
        "problems_found": []
    }
    
    # Check if we've split any circles
    if manager and hasattr(manager, 'original_circles') and manager.original_circles:
        # Validate original circles are inactive
        for orig_id in manager.original_circles:
            circle_data = manager.circles.get(orig_id, {})
            if circle_data.get('active', True) or not circle_data.get('replaced_by_splits', False):
                validation_results["original_circles_inactive"] = False
                validation_results["problems_found"].append(f"Original circle {orig_id} not properly marked inactive")
                print(f"  ⚠️ Original circle {orig_id} is not properly marked inactive")
        
        # Validate split circles are active and have correct max_additions
        for split_id in manager.split_circles:
            circle_data = manager.circles.get(split_id, {})
            
            # Check active status
            if not circle_data.get('active', False):
                validation_results["split_circles_active"] = False
                validation_results["problems_found"].append(f"Split circle {split_id} not marked active")
                print(f"  ⚠️ Split circle {split_id} is not marked active")
            
            # Check max_additions is correctly set
            member_count = circle_data.get('member_count', 0)
            expected_max_additions = max(0, 8 - member_count)
            actual_max_additions = circle_data.get('max_additions', -1)
            
            if actual_max_additions != expected_max_additions:
                validation_results["split_circles_have_correct_max_additions"] = False
                validation_results["problems_found"].append(
                    f"Split circle {split_id} has incorrect max_additions: {actual_max_additions}, expected: {expected_max_additions}"
                )
                print(f"  ⚠️ Split circle {split_id} has incorrect max_additions: {actual_max_additions}, expected: {expected_max_additions}")
    
    # Add validation results to the summary
    preprocessing_summary["validation_results"] = validation_results
    
    # Debug dump of split circles specifically for UI integration diagnosis
    if manager and hasattr(manager, 'split_circles') and manager.split_circles:
        print("\n🔍 SPLIT CIRCLES DEBUG DUMP (PRE-OPTIMIZATION):")
        print(f"  Split circles tracked in manager: {len(manager.split_circles)}")
        for split_id, original_id in manager.split_circles.items():
            # Get the circle data
            circle_data = manager.get_circle_data(split_id)
            if circle_data:
                # Check key attributes for displaying in UI and CSV
                has_region = 'region' in circle_data and circle_data['region']
                has_subregion = 'subregion' in circle_data and circle_data['subregion']
                has_meeting_time = 'meeting_time' in circle_data and circle_data['meeting_time']
                has_members = 'members' in circle_data and len(circle_data['members']) > 0
                member_count = len(circle_data.get('members', []))
                max_additions = circle_data.get('max_additions', 0)
                
                print(f"  📊 {split_id} (from {original_id}):")
                print(f"     Region: {circle_data.get('region', 'Unknown')} (Has value: {has_region})")
                print(f"     Subregion: {circle_data.get('subregion', 'Unknown')} (Has value: {has_subregion})")
                print(f"     Meeting time: {circle_data.get('meeting_time', 'Unknown')} (Has value: {has_meeting_time})")
                print(f"     Member count: {member_count} (Has members: {has_members})")
                print(f"     Max additions: {max_additions}, Active: {circle_data.get('active', False)}")
                
                # Check for member IDs
                if has_members and len(circle_data['members']) > 0:
                    print(f"     Sample members: {circle_data['members'][:3]}...")
                    
                    # Also check that these members have the correct circle assignment in participants_data
                    if participants_data is not None:
                        # Find the column used for circle assignments
                        circle_col = None
                        for col in ['proposed_NEW_circles_id', 'assigned_circle', 'circle_id']:
                            if col in participants_data.columns:
                                circle_col = col
                                break
                        
                        if circle_col:
                            sample_id = circle_data['members'][0] if circle_data['members'] else None
                            if sample_id and sample_id in participants_data['Encoded ID'].values:
                                participant_row = participants_data[participants_data['Encoded ID'] == sample_id]
                                assigned_circle = participant_row[circle_col].iloc[0] if not participant_row.empty else None
                                
                                # Check if assignment matches
                                correct_assignment = assigned_circle == split_id
                                print(f"     Member {sample_id} circle assignment: {assigned_circle} (Correct: {correct_assignment})")
                                
                                if not correct_assignment:
                                    print(f"     ⚠️ WARNING: Member {sample_id} is not correctly assigned to {split_id}!")
                                    validation_results["problems_found"].append(
                                        f"Member {sample_id} shows {assigned_circle} instead of {split_id}"
                                    )
                else:
                    print(f"     ⚠️ WARNING: Split circle {split_id} has no members!")
                    validation_results["problems_found"].append(f"Split circle {split_id} has no members")
            else:
                print(f"  ⚠️ WARNING: Could not get data for split circle {split_id}!")
                validation_results["problems_found"].append(f"Missing data for split circle {split_id}")
    
    # Log validation summary
    if validation_results["problems_found"]:
        print(f"⚠️ Found {len(validation_results['problems_found'])} issues with split circle preparation")
    else:
        print("✅ Split circle preparation validation successful - all checks passed")
    
    # Return the processed circles
    return updated_circles, preprocessing_summary

def postprocess_optimization_results(optimization_results, circles_data, participants_data, preprocessing_summary):
    """
    Postprocess optimization results to ensure split circles are properly represented.
    
    This function ensures that split circles maintain appropriate properties after the
    optimization process, such as inheriting metadata from their original circles and
    being properly tracked in the CircleMetadataManager.
    
    Args:
        optimization_results: Results from the optimization algorithm
        circles_data: Original circle data before preprocessing
        participants_data: Participant data
        preprocessing_summary: Summary from the preprocessing step
        
    Returns:
        tuple: (
            final_results: Updated optimization results,
            postprocessing_summary: Summary of postprocessing steps
        )
    """
    postprocessing_summary = {
        "steps_performed": [],
        "circles_receiving_members": 0,
        "split_circles_receiving_members": 0
    }
    
    print("\n🔄 POSTPROCESSING: Starting optimization results post-processing")
    
    # Get the metadata manager for reference
    manager = get_manager_from_session_state(st.session_state)
    
    # Check if we have valid results
    if optimization_results is None:
        print("⚠️ POSTPROCESSING: No optimization results to process")
        return optimization_results, postprocessing_summary
    
    # Check if we performed circle splitting
    if "split_circle_summary" in preprocessing_summary and preprocessing_summary["split_circle_summary"]:
        split_summary = preprocessing_summary["split_circle_summary"]
        
        # Only postprocess if we actually split any circles
        if split_summary.get("total_circles_successfully_split", 0) > 0:
            print(f"🔄 POSTPROCESSING: Handling {split_summary.get('total_circles_successfully_split', 0)} split circles")
            
            # Get all split circle IDs from the summary
            split_circle_ids = []
            for detail in split_summary.get("split_details", []):
                split_circle_ids.extend(detail.get("new_circle_ids", []))
            
            if split_circle_ids:
                print(f"🔍 POSTPROCESSING: Looking for {len(split_circle_ids)} split circles in optimization results")
                postprocessing_summary["steps_performed"].append("process_split_circles")
                postprocessing_summary["split_circle_ids"] = split_circle_ids
                
                # Count how many split circles received new members
                split_circles_receiving_members = 0
                
                # Process all results
                circles_receiving_members = 0  # Initialize the variable
                
                if isinstance(optimization_results, list):
                    # Count circles receiving members
                    circles_receiving_members = sum(1 for result in optimization_results if result.get("new_members", []))
                    
                    # Count split circles receiving members
                    for result in optimization_results:
                        if result.get("circle_id") in split_circle_ids and result.get("new_members", []):
                            split_circles_receiving_members += 1
                            print(f"  ✅ Split circle {result.get('circle_id')} received {len(result.get('new_members', []))} new members")
                
                elif isinstance(optimization_results, pd.DataFrame):
                    # Count circles receiving members (assuming there's a new_members column)
                    if "new_members" in optimization_results.columns:
                        circles_receiving_members = sum(1 for _, row in optimization_results.iterrows() 
                                                    if row.get("new_members") and len(row.get("new_members", [])) > 0)
                    
                        # Count split circles receiving members
                        for _, row in optimization_results.iterrows():
                            if row.get("circle_id") in split_circle_ids and row.get("new_members") and len(row.get("new_members", [])) > 0:
                                split_circles_receiving_members += 1
                                print(f"  ✅ Split circle {row.get('circle_id')} received {len(row.get('new_members', []))} new members")
                else:
                    print("⚠️ POSTPROCESSING: Unrecognized optimization results format")
                
                # Store stats in summary
                postprocessing_summary["circles_receiving_members"] = circles_receiving_members
                postprocessing_summary["split_circles_receiving_members"] = split_circles_receiving_members
                
                # Log findings
                print(f"✅ POSTPROCESSING: {split_circles_receiving_members} of {len(split_circle_ids)} split circles received new members")
                print(f"✅ POSTPROCESSING: {circles_receiving_members} total circles received new members")
                
                # Update the metadata manager if needed
                if split_circles_receiving_members > 0 and manager is not None:
                    print("\n🔄 POSTPROCESSING: Updating metadata manager with split circle member allocations")
                    
                    # Iterate through the results looking for split circles that got new members
                    if isinstance(optimization_results, list):
                        for result in optimization_results:
                            circle_id = result.get("circle_id")
                            new_members = result.get("new_members", [])
                            
                            if circle_id in split_circle_ids and new_members:
                                # Update metadata manager with new member information
                                if circle_id in manager.circles:
                                    circle_data = manager.circles[circle_id]
                                    
                                    # Update member count
                                    current_members = circle_data.get("members", [])
                                    total_members = len(current_members) + len(new_members)
                                    circle_data["member_count"] = total_members
                                    
                                    # Update eligible status based on new count
                                    circle_data["is_eligible"] = total_members < 8
                                    
                                    # Update manager
                                    manager.circles[circle_id] = circle_data
                                    print(f"  ✅ Updated split circle {circle_id} metadata: now {total_members} members")
                    
                    elif isinstance(optimization_results, pd.DataFrame):
                        for _, row in optimization_results.iterrows():
                            circle_id = row.get("circle_id")
                            new_members = row.get("new_members", [])
                            
                            if circle_id in split_circle_ids and new_members and len(new_members) > 0:
                                # Update metadata manager with new member information
                                if circle_id in manager.circles:
                                    circle_data = manager.circles[circle_id]
                                    
                                    # Update member count
                                    current_members = circle_data.get("members", [])
                                    total_members = len(current_members) + len(new_members)
                                    circle_data["member_count"] = total_members
                                    
                                    # Update eligible status based on new count
                                    circle_data["is_eligible"] = total_members < 8
                                    
                                    # Update manager
                                    manager.circles[circle_id] = circle_data
                                    print(f"  ✅ Updated split circle {circle_id} metadata: now {total_members} members")
    
    # Check for any unmatched compatible new members that could have gone to split circles
    # This is a validation step to ensure the optimizer properly considered split circles
    print("\n🔍 POSTPROCESSING: Validating optimizer allocation to split circles")
    
    # Simple validation to check if any eligible split circles could take more members
    if manager and hasattr(manager, 'split_circles') and manager.split_circles:
        eligible_split_circles = {
            circle_id: manager.circles[circle_id].get('max_additions', 0)
            for circle_id in manager.split_circles.keys()
            if circle_id in manager.circles and manager.circles[circle_id].get('max_additions', 0) > 0
        }
        
        if eligible_split_circles:
            # Log the eligible split circles
            print(f"  Found {len(eligible_split_circles)} eligible split circles with capacity:")
            for circle_id, capacity in eligible_split_circles.items():
                print(f"  - {circle_id}: {capacity} open slots")
            
            # Get unmatched participants from the results
            unmatched_participant_ids = []
            
            # Check optimization results format
            if isinstance(optimization_results, list) and any(r.get("unmatched_participant_ids") for r in optimization_results):
                # Get info from the first result that has unmatched_participant_ids
                for result in optimization_results:
                    if "unmatched_participant_ids" in result:
                        unmatched_participant_ids = result.get("unmatched_participant_ids", [])
                        break
            elif isinstance(optimization_results, dict) and "unmatched_participant_ids" in optimization_results:
                unmatched_participant_ids = optimization_results.get("unmatched_participant_ids", [])
            
            if unmatched_participant_ids:
                print(f"  Found {len(unmatched_participant_ids)} unmatched participants")
                postprocessing_summary["unmatched_participant_count"] = len(unmatched_participant_ids)
            else:
                print("  No unmatched participants found")
                postprocessing_summary["unmatched_participant_count"] = 0
    
    # CRITICAL FIX: Update participants_data with split circle assignments
    # This ensures the CSV export includes the correct split circle assignments
    if manager and hasattr(manager, 'split_circles') and participants_data is not None:
        print("\n🔄 UPDATING PARTICIPANTS WITH SPLIT CIRCLE ASSIGNMENTS FOR CSV EXPORT:")
        print(f"  Checking {len(manager.split_circles)} split circles for CSV export integration")
        
        # Track stats for logging
        total_updated = 0
        circle_assignment_col = None
        
        # Find the column used for circle assignments
        for col in ['proposed_NEW_circles_id', 'assigned_circle', 'circle_id']:
            if col in participants_data.columns:
                circle_assignment_col = col
                print(f"  ✅ Found circle assignment column: {circle_assignment_col}")
                break
        
        if circle_assignment_col:
            # For each split circle, update the participants' assignments
            for split_id, original_id in manager.split_circles.items():
                # Get the circle data
                circle_data = manager.get_circle_data(split_id)
                if circle_data and 'members' in circle_data and circle_data['members']:
                    members = circle_data['members']
                    print(f"  Processing split circle {split_id} with {len(members)} members")
                    
                    # Update each member's circle assignment
                    for member_id in members:
                        # Skip invalid IDs
                        if member_id is None or pd.isna(member_id):
                            continue
                            
                        # Find this member in the participants data
                        if member_id in participants_data['Encoded ID'].values:
                            # Update their circle assignment
                            idx = participants_data[participants_data['Encoded ID'] == member_id].index
                            
                            # Get current assignment to check if updating is necessary
                            current_assignment = participants_data.loc[idx, circle_assignment_col].iloc[0]
                            if current_assignment != split_id:
                                # Update the assignment
                                participants_data.loc[idx, circle_assignment_col] = split_id
                                total_updated += 1
                
            print(f"  ✅ Updated {total_updated} participants with split circle assignments for CSV export")
            
            # Store the updated participants data in session state for CSV export
            import streamlit as st
            if 'results' in st.session_state and hasattr(st.session_state.results, 'shape'):
                # Only update the circle assignment column
                if circle_assignment_col in st.session_state.results.columns:
                    # Create a copy to avoid warnings
                    results_copy = st.session_state.results.copy()
                    
                    # Update the circle assignments
                    for idx, row in participants_data.iterrows():
                        if row['Encoded ID'] in results_copy['Encoded ID'].values:
                            results_idx = results_copy[results_copy['Encoded ID'] == row['Encoded ID']].index
                            results_copy.loc[results_idx, circle_assignment_col] = row[circle_assignment_col]
                    
                    # Update session state
                    st.session_state.results = results_copy
                    print(f"  ✅ Updated session state results with split circle assignments for CSV export")
        else:
            print(f"  ⚠️ WARNING: Could not find circle assignment column in participants data")
    
    # Return the processed results
    return optimization_results, postprocessing_summary

def update_metadata_manager_with_splits(split_summary):
    """
    Update the CircleMetadataManager to properly track split circles.
    
    This ensures the CircleMetadataManager is aware of all split circles and
    maintains the relationships between original circles and their splits.
    
    Args:
        split_summary: Summary dictionary from the circle splitting process
    """
    # Get the metadata manager from session state
    manager = get_manager_from_session_state(st.session_state)
    if not manager:
        print("⚠️ POSTPROCESSING: No CircleMetadataManager found in session state")
        return
    
    print(f"\n🔄 CIRCLE SPLITTING: Updating CircleMetadataManager with {len(split_summary.get('split_details', []))} split circles")
    
    # Keep track of all split circles for debugging
    all_original_circles = []
    all_split_circles = []
    
    # Process each split circle
    for split_detail in split_summary.get("split_details", []):
        original_circle_id = split_detail.get("original_circle_id")
        new_circle_ids = split_detail.get("new_circle_ids", [])
        
        # Additional diagnostics
        all_original_circles.append(original_circle_id)
        all_split_circles.extend(new_circle_ids)
        
        print(f"🔄 Processing split of {original_circle_id} into {len(new_circle_ids)} new circles: {new_circle_ids}")
        
        # Store original circle data before potential modifications
        original_circle_data = {}
        if manager.has_circle(original_circle_id):
            original_circle_data = manager.circles.get(original_circle_id, {}).copy()
            
            # Get member list length for validation
            original_members = original_circle_data.get('members', [])
            original_member_count = len(original_members) if isinstance(original_members, list) else 0
            print(f"  Original circle {original_circle_id} has {original_member_count} members and member_count={original_circle_data.get('member_count', 'N/A')}")
            
            # CRITICAL FIX: Mark the original circle as replaced by splits
            original_circle_data["replaced_by_splits"] = True
            original_circle_data["split_into"] = new_circle_ids
            original_circle_data["active"] = False  # Mark as inactive
            
            # CRITICAL: Update the metadata tracking for this circle in the manager
            manager.original_circles[original_circle_id] = new_circle_ids
            
            # Update the original circle with this information
            manager.add_or_update_circle(original_circle_id, original_circle_data)
            print(f"✅ Updated original circle {original_circle_id} to mark as replaced by splits")
        else:
            print(f"⚠️ WARNING: Original circle {original_circle_id} not found in CircleMetadataManager")
        
        # Add each new split circle to the manager
        for i, new_circle_id in enumerate(new_circle_ids):
            # Get member counts and member lists
            if i < len(split_detail.get("member_counts", [])):
                member_count = split_detail["member_counts"][i]
            else:
                member_count = 0
                
            if i < len(split_detail.get("members", [])):
                members = split_detail["members"][i]
            else:
                members = []
            
            # Validate the members list and member count
            actual_member_count = len(members) if isinstance(members, list) else 0
            if member_count != actual_member_count:
                print(f"⚠️ Member count mismatch for {new_circle_id}: stored={member_count}, actual={actual_member_count}")
                # Fix the member count to match reality
                member_count = actual_member_count
            
            # Get host counts
            always_hosts = 0
            if "always_hosts" in split_detail and i < len(split_detail["always_hosts"]):
                always_hosts = split_detail["always_hosts"][i]
                
            sometimes_hosts = 0
            if "sometimes_hosts" in split_detail and i < len(split_detail["sometimes_hosts"]):
                sometimes_hosts = split_detail["sometimes_hosts"][i]
            
            # Inherit metadata from original circle when available
            region = split_detail.get("region", original_circle_data.get("region", ""))
            subregion = split_detail.get("subregion", original_circle_data.get("subregion", ""))
            meeting_time = split_detail.get("meeting_time", original_circle_data.get("meeting_time", ""))
            
            # Prepare complete data for the split circle
            circle_data = {
                "circle_id": new_circle_id,
                "member_count": member_count,
                "members": members,
                "is_split_circle": True,
                "active": True,  # Mark as active
                "original_circle_id": original_circle_id,
                "split_letter": new_circle_id[-1] if new_circle_id[-1].isalpha() else "",  # Extract A, B, C, etc.
                # Inherit metadata from the original circle
                "region": region,
                "subregion": subregion,
                "meeting_time": meeting_time,
                # Add host information
                "always_hosts": always_hosts,
                "sometimes_hosts": sometimes_hosts,
                # Set max_additions (split circles can grow to 8 members max)
                "max_additions": max(0, 8 - member_count),
                # CRITICAL FIX: Explicitly mark as eligible for optimization if there's room
                "is_eligible": member_count < 8
            }
            
            # Copy any other useful metadata from the original circle
            for key in ["meeting_day", "meeting_frequency", "primary_language", "leader_name", "co_leader_name"]:
                if key in original_circle_data:
                    circle_data[key] = original_circle_data[key]
            
            # CRITICAL: Update the split circle tracking in the manager
            manager.split_circles[new_circle_id] = original_circle_id
            
            # Additional eligibility diagnostics
            if member_count < 8:
                print(f"  ✅ Split circle {new_circle_id} has {member_count} members and max_additions={circle_data['max_additions']}")
                print(f"  ✅ Circle is eligible for new members (is_eligible={circle_data.get('is_eligible', 'not set')})")
            else:
                print(f"  ⚠️ Split circle {new_circle_id} has {member_count} members, no room for new members")
                print(f"  ⚠️ Circle is not eligible (is_eligible={circle_data.get('is_eligible', 'not set')})")
            
            # Add to metadata manager
            manager.add_or_update_circle(new_circle_id, circle_data)
            print(f"✅ Added split circle {new_circle_id} to CircleMetadataManager with {member_count} members")
    
    print(f"✅ CircleMetadataManager now tracking {len(manager.split_circles)} split circles")
    
    # Verify the split circle tracking data
    print(f"\n🔍 VERIFYING SPLIT CIRCLE TRACKING:")
    print(f"  Original circles: {all_original_circles}")
    print(f"  Split circles: {all_split_circles}")
    print(f"  manager.original_circles contains {len(manager.original_circles)} entries")
    print(f"  manager.split_circles contains {len(manager.split_circles)} entries")
    
    # CRITICAL FIX: Use the enhanced metadata synchronization system to propagate changes
    print(f"\n🔄 UPDATING SESSION STATE: Using metadata synchronization to propagate split circle changes")
    
    # Collect inputs for synchronization
    circles_df = st.session_state.matched_circles if hasattr(st.session_state, 'matched_circles') else None
    results_df = st.session_state.results if hasattr(st.session_state, 'results') else None
    
    # POWERFUL SYNCHRONIZATION: Use the enhanced metadata synchronization function to update everything
    updated_circles_df, has_changes = manager.synchronize_metadata(
        circles_df=circles_df,
        results_df=results_df,
        split_summary=split_summary
    )
    
    if has_changes:
        # Update session state with synchronized data
        print(f"  ✅ Metadata synchronization successful - updating session state")
        if updated_circles_df is not None:
            print(f"  Replacing matched_circles in session state with synchronized DataFrame ({len(updated_circles_df)} rows)")
            st.session_state.matched_circles = updated_circles_df
            
            # Verify split circles are in the updated DataFrame
            if hasattr(updated_circles_df, 'circle_id'):
                split_circle_count = sum(1 for c_id in updated_circles_df['circle_id'] if 'SPLIT' in c_id)
            else:
                split_circle_count = sum(1 for c in updated_circles_df if 'SPLIT' in c.get('circle_id', ''))
            print(f"  ✅ Synchronized DataFrame contains {split_circle_count} split circles")
            
            # DEBUG: Check if specific test circles are in the updated data
            test_circles = ['IP-NAP-01', 'IP-SHA-01', 'IP-ATL-1']
            
            # Get active and inactive counts for reporting
            active_count = sum(1 for _, c in manager.circles.items() if not c.get("replaced_by_splits", False))
            inactive_count = sum(1 for _, c in manager.circles.items() if c.get("replaced_by_splits", False))
            print(f"  CircleMetadataManager contains {active_count} active and {inactive_count} inactive circles")
            
            # Check priority original circles - should be inactive and not in DataFrame
            for test_id in test_circles:
                in_original_circles = test_id in all_original_circles
                in_df = False
                
                # Check if test_id is in the updated DataFrame
                if hasattr(updated_circles_df, 'circle_id'):
                    in_df = test_id in updated_circles_df['circle_id'].values
                else:
                    in_df = any(c.get('circle_id') == test_id for c in updated_circles_df)
                
                if in_df and in_original_circles:
                    print(f"  ⚠️ ERROR: Original circle {test_id} is still in matched_circles even though it was split")
                elif not in_df and in_original_circles:
                    print(f"  ✅ Original circle {test_id} correctly removed from matched_circles (replaced by splits)")
                elif not in_df and not in_original_circles:
                    print(f"  ⚠️ Expected circle {test_id} is missing from matched_circles data")
            
            # Check a few split circles - should be active and in DataFrame
            split_examples = []
            if hasattr(updated_circles_df, 'to_dict'):
                # If it's a DataFrame, convert to records
                df_records = updated_circles_df.to_dict('records')
                split_examples = [c for c in df_records if 'SPLIT' in c.get('circle_id', '')][:3]
            else:
                # It's already a list
                split_examples = [c for c in updated_circles_df if 'SPLIT' in c.get('circle_id', '')][:3]
                
            for i, circle in enumerate(split_examples):
                circle_id = circle.get('circle_id', 'Unknown')
                member_count = circle.get('member_count', 0)
                print(f"  ✅ Split circle {circle_id} is in matched_circles with {member_count} members")
        else:
            print("⚠️ No active circles found in CircleMetadataManager")
    else:
        print("⚠️ matched_circles not found in session state")
    
    # 2. Update both the metadata manager and split circle summary in session state
    st.session_state.circle_metadata_manager = manager
    print("  ✅ Updated circle_metadata_manager in session state")
    
    # 3. Store the split summary in session state for UI reporting
    st.session_state.split_circle_summary = split_summary
    print("  ✅ Updated split_circle_summary in session state")
    
    # 4. Force a rerun of several key data-dependent functions to refresh UI data
    if hasattr(st.session_state, 'processed_data') and st.session_state.processed_data is not None:
        # Rebuild circle member lists to ensure all members are correctly assigned
        from utils.circle_rebuilder import rebuild_circle_member_lists
        
        try:
            # Get current circles and participants
            current_circles = st.session_state.matched_circles
            participant_data = st.session_state.processed_data
            
            # Rebuild all circle member lists
            print("\n🔄 REBUILDING ALL CIRCLE MEMBER LISTS:")
            updated_circles = rebuild_circle_member_lists(current_circles, participant_data)
            
            # Update session state with rebuilt circles
            st.session_state.matched_circles = updated_circles
            print("  ✅ Successfully rebuilt all circle member lists and updated session state")
        except Exception as e:
            print(f"  ⚠️ Error rebuilding circle member lists: {str(e)}")
    
    print("\n✅ SPLIT CIRCLE UPDATES COMPLETE: All session state variables updated")
    
    # Return success
    return True