import streamlit as st
import logging

# Configure logging
logger = logging.getLogger('feature_flags')

# Default feature flag values
DEFAULT_FLAGS = {
    # Data standardization flags
    'use_standardized_host_status': False,  # Use normalized host values (ALWAYS, SOMETIMES, NEVER)
    'use_standardized_member_lists': False,  # Use standardized member list format
    'use_optimizer_metadata': False,  # Use optimizer as source of truth for metadata
    
    # Validation flags
    'enable_metadata_validation': False,  # Enable detailed metadata validation in Debug tab
    'debug_data_standardization': False,  # Enable detailed data standardization logs
    
    # Debug flags
    'enable_feature_flags_ui': False,  # Enable UI for toggling feature flags
}

def initialize_feature_flags():
    """Initialize feature flags in session state if they don't exist"""
    if 'feature_flags' not in st.session_state:
        st.session_state.feature_flags = DEFAULT_FLAGS.copy()
        logger.info("Initialized feature flags with default values")

def get_flag(flag_name):
    """Get the value of a feature flag, defaulting to False if not found"""
    # Initialize flags if needed
    if not hasattr(st, 'session_state') or 'feature_flags' not in st.session_state:
        # When running outside of Streamlit context
        return DEFAULT_FLAGS.get(flag_name, False)
    
    # Initialize if not already done
    initialize_feature_flags()
    
    # Return flag value or default to False if flag doesn't exist
    return st.session_state.feature_flags.get(flag_name, False)

def set_flag(flag_name, value):
    """Set the value of a feature flag"""
    # Initialize if not already done
    initialize_feature_flags()
    
    # Set the flag value
    st.session_state.feature_flags[flag_name] = value
    logger.info(f"Set feature flag '{flag_name}' to {value}")

def reset_flags():
    """Reset all feature flags to default values"""
    st.session_state.feature_flags = DEFAULT_FLAGS.copy()
    logger.info("Reset all feature flags to default values")

def render_debug_flags():
    """Render a UI for toggling feature flags"""
    # Only show if the UI flag is enabled
    if not get_flag('enable_feature_flags_ui'):
        return
    
    st.subheader("Feature Flags")
    
    with st.expander("Configure Feature Flags", expanded=True):
        # Group flags by category
        st.markdown("### Data Standardization")
        
        # Host status standardization
        host_status = st.checkbox(
            "Use standardized host status", 
            value=get_flag('use_standardized_host_status'),
            help="Standardize host status values to ALWAYS, SOMETIMES, NEVER"
        )
        set_flag('use_standardized_host_status', host_status)
        
        # Member list standardization
        member_lists = st.checkbox(
            "Use standardized member lists", 
            value=get_flag('use_standardized_member_lists'),
            help="Standardize member lists to consistent format"
        )
        set_flag('use_standardized_member_lists', member_lists)
        
        # Optimizer metadata
        optimizer_metadata = st.checkbox(
            "Use optimizer metadata", 
            value=get_flag('use_optimizer_metadata'),
            help="Use optimizer as source of truth for metadata"
        )
        set_flag('use_optimizer_metadata', optimizer_metadata)
        
        st.markdown("### Validation & Debug")
        
        # Metadata validation
        metadata_validation = st.checkbox(
            "Enable metadata validation", 
            value=get_flag('enable_metadata_validation'),
            help="Show detailed metadata validation in Debug tab"
        )
        set_flag('enable_metadata_validation', metadata_validation)
        
        # Data standardization debug
        data_std_debug = st.checkbox(
            "Debug data standardization", 
            value=get_flag('debug_data_standardization'),
            help="Show detailed data standardization logs"
        )
        set_flag('debug_data_standardization', data_std_debug)
        
        st.markdown("---")
        
        # Reset button
        if st.button("Reset All Flags"):
            reset_flags()
            st.experimental_rerun()
