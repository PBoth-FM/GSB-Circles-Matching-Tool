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
            
            # Update the CircleMetadataManager to reflect split circles
            update_metadata_manager_with_splits(split_summary)
            postprocessing_summary["steps_performed"].append("update_metadata_manager")
    
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
    
    print(f"🔄 Updating CircleMetadataManager with {len(split_summary.get('split_details', []))} split circles")
    
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
            
            # CRITICAL FIX: Mark the original circle as replaced by splits
            original_circle_data["replaced_by_splits"] = True
            original_circle_data["split_into"] = new_circle_ids
            original_circle_data["active"] = False  # Mark as inactive
            
            # CRITICAL: Update the metadata tracking for this circle in the manager
            if not original_circle_id in manager.original_circles:
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
                "max_additions": max(0, 8 - member_count)
            }
            
            # Copy any other useful metadata from the original circle
            for key in ["meeting_day", "meeting_frequency", "primary_language", "leader_name", "co_leader_name"]:
                if key in original_circle_data:
                    circle_data[key] = original_circle_data[key]
            
            # CRITICAL: Update the split circle tracking in the manager
            manager.split_circles[new_circle_id] = original_circle_id
            
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
    
    # CRITICAL FIX: Force a refresh of all data from the CircleMetadataManager to session state
    # This ensures all parts of the app see the changes
    if hasattr(st.session_state, 'matched_circles') and st.session_state.matched_circles is not None:
        # Get all ACTIVE circle data from manager as a DataFrame
        circles_data = []
        for circle_id, circle_data in manager.circles.items():
            # Only include active circles (not replaced by splits)
            if not circle_data.get("replaced_by_splits", False):
                # Make a copy of the data with circle_id included
                circle_copy = circle_data.copy()
                circle_copy["circle_id"] = circle_id
                circles_data.append(circle_copy)
        
        # Create a DataFrame from the circle data
        if circles_data:
            import pandas as pd
            updated_df = pd.DataFrame(circles_data)
            st.session_state.matched_circles = updated_df
            print(f"✅ Updated matched_circles in session state with {len(circles_data)} active circles")
            
            # DEBUG: Look for our test circles in the updated data
            test_circles = ['IP-NAP-01', 'IP-SHA-01', 'IP-ATL-1', 'IP-NAP-SPLIT-01-A', 'IP-NAP-SPLIT-01-B']
            for test_id in test_circles:
                found = any(c.get('circle_id') == test_id for c in circles_data)
                if found:
                    print(f"  ✅ Circle {test_id} is present in the updated matched_circles data")
                else:
                    # Check if it's an original circle that should be inactive
                    if test_id in all_original_circles:
                        print(f"  ✅ Original circle {test_id} is correctly NOT included in matched_circles (replaced by splits)")
                    else:
                        print(f"  ⚠️ Expected circle {test_id} is missing from the updated matched_circles data")
        else:
            print("⚠️ No active circles found in CircleMetadataManager")
    
    # Also update the split circle summary in session state for UI reporting
    st.session_state.split_circle_summary = split_summary
    print("✅ Updated split_circle_summary in session state")