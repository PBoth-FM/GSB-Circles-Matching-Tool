import streamlit as st
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("feature_flags")

def initialize_feature_flags():
    """
    Initialize feature flags with default values if they don't exist.
    All new features start disabled by default.
    """
    if 'feature_flags' not in st.session_state:
        st.session_state.feature_flags = {
            # Meta-flag to control visibility of feature flag UI in debug tab
            'enable_feature_flags_ui': False,
            
            # Data standardization flags
            'use_standardized_host_status': False,
            'use_standardized_member_lists': False,
            'use_optimizer_metadata': False,
            'enable_metadata_validation': False,
            'debug_data_standardization': False
        }
        logger.info("Initialized feature flags with default values")
        
def get_flag(flag_name, default=False):
    """
    Get a feature flag value. Returns default if flag doesn't exist.
    
    Args:
        flag_name (str): The name of the flag to check.
        default (bool): The default value to return if flag is not found.
        
    Returns:
        bool: The value of the feature flag.
    """
    if 'feature_flags' not in st.session_state:
        initialize_feature_flags()
        
    return st.session_state.feature_flags.get(flag_name, default)

def set_flag(flag_name, value):
    """
    Set a feature flag value.
    
    Args:
        flag_name (str): The name of the flag to set.
        value (bool): The value to set the flag to.
    """
    if 'feature_flags' not in st.session_state:
        initialize_feature_flags()
        
    st.session_state.feature_flags[flag_name] = value
    logger.info(f"Set feature flag '{flag_name}' to {value}")

def render_debug_flags():
    """
    Render the feature flags UI in the debug tab.
    This should only be visible if enable_feature_flags_ui is set to True.
    """
    if not get_flag('enable_feature_flags_ui'):
        return
        
    st.write("## Feature Flags")
    st.write("Toggle feature flags to enable or disable experimental features.")
    
    # Create multiple columns for a more compact layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Data standardization section
        st.write("### Data Standardization")
        
        # Standardized host status flag
        use_std_host = st.checkbox(
            "Use standardized host status", 
            value=get_flag('use_standardized_host_status'),
            help="Normalize host status to ALWAYS, SOMETIMES, or NEVER across the application"
        )
        set_flag('use_standardized_host_status', use_std_host)
        
        # Standardized member lists flag
        use_std_members = st.checkbox(
            "Use standardized member lists", 
            value=get_flag('use_standardized_member_lists'),
            help="Normalize member lists to List[str] format regardless of input format"
        )
        set_flag('use_standardized_member_lists', use_std_members)
    
    with col2:
        # Metadata integration section
        st.write("### Metadata Integration")
        
        # Optimizer metadata flag
        use_opt_meta = st.checkbox(
            "Use optimizer metadata", 
            value=get_flag('use_optimizer_metadata'),
            help="Use metadata directly from optimizer instead of recalculating"
        )
        set_flag('use_optimizer_metadata', use_opt_meta)
        
        # Metadata validation flag
        enable_validation = st.checkbox(
            "Enable metadata validation", 
            value=get_flag('enable_metadata_validation'),
            help="Show validation reports for metadata consistency in the Debug tab"
        )
        set_flag('enable_metadata_validation', enable_validation)
    
    # Debug section (single column)
    st.write("### Debug Options")
    debug_std = st.checkbox(
        "Debug data standardization", 
        value=get_flag('debug_data_standardization'),
        help="Show detailed logs for data standardization operations"
    )
    set_flag('debug_data_standardization', debug_std)
    
    # Show current state of all flags in an expander
    with st.expander("Current feature flag state"):
        for flag, value in st.session_state.feature_flags.items():
            if flag != 'enable_feature_flags_ui':  # Skip showing the meta-flag
                st.write(f"{flag}: **{value}**")
