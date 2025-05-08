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
        "split_circle_summary": None
    }
    
    # Step 1: Split large circles
    print("🔄 PREPROCESSING: Starting circle splitting process")
    updated_circles, split_summary = split_large_circles(circles_data, participants_data)
    preprocessing_summary["steps_performed"].append("split_large_circles")
    preprocessing_summary["split_circle_summary"] = split_summary
    
    # Store the split summary in session state for the UI to use
    if "split_circle_summary" not in st.session_state or st.session_state.split_circle_summary != split_summary:
        st.session_state.split_circle_summary = split_summary
        print("✅ Stored split circle summary in session state")
    
    # Step 2: Update the CircleMetadataManager with split circle information
    if split_summary and split_summary.get('total_circles_successfully_split', 0) > 0:
        print(f"🔄 PREPROCESSING: Updating metadata manager with {split_summary.get('total_circles_successfully_split', 0)} split circles")
        update_metadata_manager_with_splits(split_summary)
        preprocessing_summary["steps_performed"].append("update_metadata_manager_with_splits")
    
    # Additional preprocessing steps can be added here
    
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
        "steps_performed": []
    }
    
    # Check if we performed circle splitting
    if "split_circle_summary" in preprocessing_summary and preprocessing_summary["split_circle_summary"]:
        split_summary = preprocessing_summary["split_circle_summary"]
        
        # Only postprocess if we actually split any circles
        if split_summary["total_circles_successfully_split"] > 0:
            print(f"🔄 POSTPROCESSING: Handling {split_summary['total_circles_successfully_split']} split circles")
            
            # The metadata manager has already been updated in the preprocessing step,
            # so we just need to verify the changes are reflected in the results
            
            # Check for any split circles in optimization results
            split_circle_ids = []
            for detail in split_summary.get("split_details", []):
                split_circle_ids.extend(detail.get("new_circle_ids", []))
            
            if split_circle_ids:
                print(f"✅ POSTPROCESSING: Verified {len(split_circle_ids)} split circles exist")
                postprocessing_summary["steps_performed"].append("verify_split_circles")
                postprocessing_summary["split_circle_ids"] = split_circle_ids
    
    # Additional postprocessing steps can be added here
    
    # For now, just return the original results
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