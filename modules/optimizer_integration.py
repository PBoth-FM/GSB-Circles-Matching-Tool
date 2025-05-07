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
    
    # Process each split circle
    for split_detail in split_summary.get("split_details", []):
        original_circle_id = split_detail.get("original_circle_id")
        new_circle_ids = split_detail.get("new_circle_ids", [])
        
        # Check if the original circle exists and should be removed
        # Typically, we might want to keep it for historical reference
        if manager.has_circle(original_circle_id):
            manager.remove_circle(original_circle_id)
            print(f"✅ Removed original circle {original_circle_id} from CircleMetadataManager")
        
        # Add each new split circle to the manager
        for i, new_circle_id in enumerate(new_circle_ids):
            if i < len(split_detail.get("member_counts", [])):
                member_count = split_detail["member_counts"][i]
            else:
                member_count = 0
                
            if i < len(split_detail.get("members", [])):
                members = split_detail["members"][i]
            else:
                members = []
                
            # Prepare data for the split circle
            circle_data = {
                "circle_id": new_circle_id,
                "member_count": member_count,
                "members": members,
                "is_split_circle": True,
                "original_circle_id": original_circle_id,
                # Inherit metadata from the original circle
                "region": split_detail.get("region", ""),
                "subregion": split_detail.get("subregion", ""),
                "meeting_time": split_detail.get("meeting_time", ""),
                # Add host information
                "always_hosts": split_detail.get("always_hosts", [0])[i] if i < len(split_detail.get("always_hosts", [])) else 0,
                "sometimes_hosts": split_detail.get("sometimes_hosts", [0])[i] if i < len(split_detail.get("sometimes_hosts", [])) else 0,
                # Set max_additions (split circles can grow to 8 members max)
                "max_additions": max(0, 8 - member_count)
            }
            
            # Add to metadata manager
            manager.add_or_update_circle(new_circle_id, circle_data)
            print(f"✅ Added split circle {new_circle_id} to CircleMetadataManager")
    
    print(f"✅ CircleMetadataManager now tracking {len(manager.split_circles)} split circles")