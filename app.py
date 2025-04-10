import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from modules.data_loader import load_data, validate_data
from modules.data_processor import process_data, normalize_data
from modules.optimizer import run_matching_algorithm
from modules.ui_components import (
    render_match_tab, 
    render_details_tab, 
    render_debug_tab,
    render_results_overview,
    render_circle_table,
    render_unmatched_table
)
from utils.helpers import generate_download_link

# Configure Streamlit page
st.set_page_config(
    page_title="CirclesTool2",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'matched_circles' not in st.session_state:
    st.session_state.matched_circles = None
if 'unmatched_participants' not in st.session_state:
    st.session_state.unmatched_participants = None
if 'validation_errors' not in st.session_state:
    st.session_state.validation_errors = None
if 'deduplication_messages' not in st.session_state:
    st.session_state.deduplication_messages = []
if 'exec_time' not in st.session_state:
    st.session_state.exec_time = None
if 'config' not in st.session_state:
    st.session_state.config = {
        'debug_mode': False,
        'min_circle_size': 5,
        'existing_circle_handling': 'preserve',
        'optimization_weight_location': 3,
        'optimization_weight_time': 2,
        'enable_host_requirement': True
    }

def main():
    st.title("CirclesTool2")
    st.write("GSB Alumni Circle Matching Tool")
    
    # Create tabs for navigation
    tab1, tab2, tab3 = st.tabs(["Match", "Details", "Debug"])
    
    with tab1:
        # Use our custom match tab function instead of the imported one
        match_tab_callback()
        
    with tab2:
        render_details_tab()
            
    with tab3:
        render_debug_tab()

def run_optimization():
    """Run the optimization algorithm and store results in session state"""
    try:
        with st.spinner("Running matching algorithm..."):
            start_time = time.time()
            
            # Run the matching algorithm
            results, matched_circles, unmatched_participants = run_matching_algorithm(
                st.session_state.processed_data,
                st.session_state.config
            )
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.matched_circles = matched_circles
            st.session_state.unmatched_participants = unmatched_participants
            st.session_state.exec_time = time.time() - start_time
            
            st.success(f"Matching completed in {st.session_state.exec_time:.2f} seconds!")
            st.session_state.active_tab = "Results"
            
    except Exception as e:
        st.error(f"Error during optimization: {str(e)}")
        if st.session_state.config['debug_mode']:
            st.exception(e)

def process_uploaded_file(uploaded_file):
    """Process the uploaded CSV file"""
    try:
        with st.spinner("Processing data..."):
            # Load and validate data
            df, validation_errors, deduplication_messages = load_data(uploaded_file)
            st.session_state.df = df
            st.session_state.validation_errors = validation_errors
            st.session_state.deduplication_messages = deduplication_messages
            
            # Display validation errors if any
            if len(validation_errors) > 0:
                st.warning(f"Found {len(validation_errors)} validation issues:")
                for error in validation_errors[:5]:  # Show first 5 errors
                    st.write(f"- {error}")
                if len(validation_errors) > 5:
                    st.write(f"...and {len(validation_errors) - 5} more issues.")
            
            # Display deduplication messages if any
            if len(deduplication_messages) > 0:
                st.warning(f"Found and fixed {len(deduplication_messages)} duplicate Encoded IDs:")
                for message in deduplication_messages[:5]:  # Show first 5 messages
                    st.write(f"- {message}")
                if len(deduplication_messages) > 5:
                    st.write(f"...and {len(deduplication_messages) - 5} more duplicates fixed.")
            
            # Process and normalize data
            processed_data = process_data(df)
            normalized_data = normalize_data(processed_data)
            st.session_state.processed_data = normalized_data
            
            st.success(f"Data processed successfully! {len(normalized_data)} participants loaded.")
            
            # Display data summary
            st.subheader("Data Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Participants", len(normalized_data))
            
            with col2:
                current_continuing = len(normalized_data[normalized_data['Status'] == 'CURRENT-CONTINUING'])
                st.metric("Current Continuing", current_continuing)
            
            with col3:
                new_participants = len(normalized_data[normalized_data['Status'] == 'NEW'])
                st.metric("New Participants", new_participants)
            
            st.subheader("Configuration")
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                st.session_state.config['min_circle_size'] = st.slider(
                    "Minimum Circle Size", 
                    min_value=3, 
                    max_value=10, 
                    value=5
                )
                
                st.session_state.config['debug_mode'] = st.checkbox(
                    "Debug Mode", 
                    value=st.session_state.config['debug_mode']
                )
            
            with config_col2:
                st.session_state.config['existing_circle_handling'] = st.radio(
                    "Existing Circle Handling",
                    options=['preserve', 'dissolve', 'optimize'],
                    index=0
                )
                
                st.session_state.config['enable_host_requirement'] = st.checkbox(
                    "Enforce Host Requirements", 
                    value=True
                )
            
            # Add run button once data is loaded
            if st.button("Run Matching Algorithm"):
                run_optimization()
                
            # Display results if available
            if st.session_state.results is not None:
                render_results_overview()
                
                st.subheader("Circle Composition")
                render_circle_table()
                
                st.subheader("Unmatched Participants")
                render_unmatched_table()
                
                # Download button for results
                if st.session_state.results is not None:
                    st.download_button(
                        "Download Results CSV",
                        generate_download_link(st.session_state.results),
                        "circles_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        if 'debug_mode' in st.session_state.config and st.session_state.config['debug_mode']:
            st.exception(e)

# Define callback for the Match tab
def match_tab_callback():
    st.subheader("Upload Participant Data")
    
    uploaded_file = st.file_uploader(
        "Upload CSV with participant data", 
        type=["csv", "xlsx"],
        help="Upload a CSV file with participant data following the required format"
    )
    
    if uploaded_file is not None:
        process_uploaded_file(uploaded_file)

# Use the match tab callback directly
match_tab_callback_func = match_tab_callback

if __name__ == "__main__":
    main()
