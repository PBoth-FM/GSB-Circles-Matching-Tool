import streamlit as st
import logging

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_FLAGS = {
    "use_standardized_host_status": False,
    "use_standardized_member_lists": False,
    "use_optimizer_metadata": False,
    "enable_metadata_validation": False,
    "debug_data_standardization": True  # Enable debug logs by default
}

def initialize_feature_flags():
    """Initialize feature flags if not already in session state"""
    if "feature_flags" not in st.session_state:
        st.session_state.feature_flags = DEFAULT_FLAGS.copy()
        logger.info("Initialized feature flags with default values")

def get_flag(flag_name, default=False):
    """Get the value of a feature flag"""
    initialize_feature_flags()
    return st.session_state.feature_flags.get(flag_name, default)

def set_flag(flag_name, value):
    """Set the value of a feature flag"""
    initialize_feature_flags()
    previous_value = st.session_state.feature_flags.get(flag_name)
    st.session_state.feature_flags[flag_name] = value
    
    if previous_value != value:
        logger.info(f"Feature flag '{flag_name}' changed: {previous_value} → {value}")
    
    return value
    
def toggle_flag(flag_name):
    """Toggle the value of a feature flag"""
    current_value = get_flag(flag_name)
    return set_flag(flag_name, not current_value)

def render_debug_flags():
    """Render debug UI for feature flags"""
    initialize_feature_flags()
    
    with st.expander("⚙️ Implementation Feature Flags", expanded=False):
        st.markdown("""### Feature Flags
        
        Toggle these flags to enable or disable new circle metadata features.
        Use this to test changes safely and roll back if needed.
        """)
        
        cols = st.columns(2)
        for i, (flag, value) in enumerate(st.session_state.feature_flags.items()):
            with cols[i % 2]:
                new_value = st.checkbox(
                    f"{flag}", 
                    value=value,
                    key=f"flag_{flag}"
                )
                if new_value != value:
                    set_flag(flag, new_value)
                    st.experimental_rerun()
        
        # Add button to reset all flags
        if st.button("Reset All Flags to Default"):
            st.session_state.feature_flags = DEFAULT_FLAGS.copy()
            st.experimental_rerun()
