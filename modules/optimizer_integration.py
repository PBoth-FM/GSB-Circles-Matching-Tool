"""
Integration module to connect the circle splitter with the optimization process.
This module ensures circle splitting happens before optimization, allowing new
participants to be assigned to split circles.
"""

import pandas as pd
import streamlit as st
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_circle_splitting_before_optimization(processed_data, circles_data=None):
    """
    Apply circle splitting logic before optimization occurs.
    
    Args:
        processed_data: DataFrame with processed participant data
        circles_data: Optional DataFrame with existing circles data
        
    Returns:
        updated_circles: DataFrame with circles after splitting
        split_summary: Dictionary with summary of the splitting process
    """
    logger.info("Starting circle splitting integration before optimization")
    print("🔄 INTEGRATION: Applying circle splitting before optimization")
    
    try:
        # Import circle splitter here to avoid circular imports
        from modules.circle_splitter import split_large_circles
        
        # If no circles_data is provided, try to get it from session state
        if circles_data is None:
            if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
                circles_data = st.session_state.matched_circles
                print(f"🔄 INTEGRATION: Using {len(circles_data)} circles from session state")
            else:
                # No circles data available yet - we might be in the first run
                print("ℹ️ INTEGRATION: No circles data found in session state")
                return None, {"status": "no_circles_data"}
        
        # Apply the circle splitting function
        print(f"🔄 INTEGRATION: Splitting circles with {len(processed_data)} participants")
        updated_circles, split_summary = split_large_circles(circles_data, processed_data)
        
        # Log the results
        print(f"🔄 INTEGRATION: Splitting complete. {split_summary['total_circles_successfully_split']} circles split into {split_summary['total_new_circles_created']} new circles")
        
        # Store results in session state for debugging
        st.session_state.split_circle_summary = split_summary
        
        # Update circle manager if it exists
        update_circle_manager_with_splits(updated_circles, split_summary)
        
        return updated_circles, split_summary
        
    except Exception as e:
        logger.error(f"Error during circle splitting integration: {str(e)}")
        print(f"❌ INTEGRATION ERROR: {str(e)}")
        print(traceback.format_exc())
        
        # Return the original data without splitting
        return circles_data, {"status": "error", "error_message": str(e)}

def update_circle_manager_with_splits(updated_circles, split_summary):
    """
    Update the CircleMetadataManager with split circle information.
    
    Args:
        updated_circles: DataFrame with updated circles after splitting
        split_summary: Dictionary with summary of the splitting process
    """
    try:
        # Import manager here to avoid circular imports
        from utils.circle_metadata_manager import get_manager_from_session_state
        
        # Get the manager from session state
        manager = get_manager_from_session_state(st.session_state)
        
        if manager:
            print("🔄 INTEGRATION: Updating CircleMetadataManager with split circles")
            
            # Filter for just the split circles
            if isinstance(updated_circles, pd.DataFrame):
                split_circles = updated_circles[updated_circles['circle_id'].str.contains('SPLIT')]
                
                # For each split circle, add or update it in the manager
                for _, circle in split_circles.iterrows():
                    circle_id = circle['circle_id']
                    
                    # Convert Series to dict for storage in the manager
                    circle_dict = circle.to_dict()
                    
                    # Set additional metadata
                    circle_dict['is_split_circle'] = True
                    circle_dict['metadata_source'] = 'circle_splitter'
                    
                    # Add the circle data to the manager
                    manager.add_or_update_circle(circle_id, circle_dict)
                    
                    print(f"✅ INTEGRATION: Added/updated split circle {circle_id} in CircleMetadataManager")
                
                # Remove the original large circles that were split
                for detail in split_summary.get('split_details', []):
                    original_id = detail.get('original_circle_id')
                    if original_id and manager.has_circle(original_id):
                        manager.remove_circle(original_id)
                        print(f"✅ INTEGRATION: Removed original large circle {original_id} from CircleMetadataManager")
            
            print(f"✅ INTEGRATION: CircleMetadataManager updated with {len(split_summary.get('split_details', []))} split circle details")
        else:
            print("⚠️ INTEGRATION: CircleMetadataManager not found in session state")
    
    except Exception as e:
        logger.error(f"Error updating CircleMetadataManager with splits: {str(e)}")
        print(f"❌ INTEGRATION ERROR: Failed to update CircleMetadataManager: {str(e)}")