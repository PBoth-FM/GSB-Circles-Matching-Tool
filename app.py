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
    render_demographics_tab,
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
if 'circle_eligibility_logs' not in st.session_state:
    st.session_state.circle_eligibility_logs = {}
if 'config' not in st.session_state:
    st.session_state.config = {
        'debug_mode': True,  # CRITICAL FIX: Force debug mode on to help diagnose compatibility issues
        'min_circle_size': 5,
        'existing_circle_handling': 'optimize',  # Changed default from 'preserve' to 'optimize'
        'optimization_weight_location': 3,
        'optimization_weight_time': 2,
        'enable_host_requirement': True
    }

def main():
    st.title("CirclesTool2")
    st.write("GSB Alumni Circle Matching Tool")
    
    # Create tabs for navigation, moved Demographics after Match per user request
    # Removed East Bay Debug tab to focus on Seattle testing
    tab1, tab2, tab3, tab4 = st.tabs(["Match", "Demographics", "Details", "Debug"])
    
    with tab1:
        # Use our custom match tab function instead of the imported one
        match_tab_callback()
        
    with tab2:
        render_demographics_tab()
    
    with tab3:
        render_details_tab()
            
    with tab4:
        render_debug_tab()

def run_optimization():
    """Run the optimization algorithm and store results in session state"""
    # Reset debug logs at the start of each optimization run
    # We're no longer importing the global circle_eligibility_logs variable
    from modules.optimizer_new import debug_eligibility_logs
    from modules.optimizer_new import update_session_state_eligibility_logs
    
    # We now manage logs through session state directly instead of a global variable
    # Initialize the logs dictionary in session state if needed
    import streamlit as st
    if 'circle_eligibility_logs' not in st.session_state:
        st.session_state.circle_eligibility_logs = {}
    
    # Clear the existing session state logs
    st.session_state.circle_eligibility_logs.clear()
    print(f"🧹 Cleared previous logs from session state")
    
    # Log the reset for debugging
    debug_eligibility_logs("Cleared circle eligibility logs before optimization run")
        
    # Log the reset for debugging
    print("🔄 CRITICAL DEBUG: Reset circle eligibility logs before optimization run")
    
    # Update configuration with UI selections
    # Get circle handling mode from UI (or use default if not set)
    selected_mode = st.session_state.get('existing_circle_handling', 'optimize')
    print(f"🔄 Using selected circle handling mode: {selected_mode}")
    
    # Update the config with the selected mode
    st.session_state.config['existing_circle_handling'] = selected_mode
    
    # Update other config parameters from UI
    st.session_state.config['optimization_weight_location'] = st.session_state.get('location_weight', 5.0)
    st.session_state.config['optimization_weight_time'] = st.session_state.get('time_weight', 5.0)
    
    try:
        with st.spinner("Running matching algorithm..."):
            start_time = time.time()
            
            # 🚨 CRITICAL TEST: Add a special Seattle test participant to force a match with IP-SEA-01
            import pandas as pd
            test_data = st.session_state.processed_data.copy()
            
            # Create a test participant for Seattle
            test_participant = {
                'Encoded ID': '99999000001',  # Made-up ID for test purposes
                'Status': 'NEW',
                'Current_Region': 'Seattle',
                'Current_Subregion': 'Seattle',
                'Derived_Region': 'Seattle',
                'first_choice_location': 'Downtown Seattle (Capital Hill/Madrona/Queen Ann/etc.)',  # Exact match with IP-SEA-01
                'second_choice_location': 'Bellevue/Mercer Island/Eastside',
                'third_choice_location': 'South Seattle',
                'first_choice_time': 'Wednesday (Evenings)',  # Exact match with IP-SEA-01
                'second_choice_time': 'Monday-Thursday (Evenings)', 
                'third_choice_time': 'M-Th (Evenings)'
            }
            
            # Add additional required columns to match the dataframe structure
            for col in test_data.columns:
                if col not in test_participant:
                    test_participant[col] = None
            
            # Add the test participant to the data
            test_data = pd.concat([test_data, pd.DataFrame([test_participant])], ignore_index=True)
            
            print(f"\n🔍 ADDED SEATTLE TEST PARTICIPANT:")
            print(f"  ID: {test_participant['Encoded ID']}")
            print(f"  Status: {test_participant['Status']}")
            print(f"  Region: {test_participant['Current_Region']}")
            print(f"  Location: {test_participant['first_choice_location']}")
            print(f"  Time: {test_participant['first_choice_time']}")
            print(f"  This participant should match with IP-SEA-01 due to exact location and time match.")
            print(f"  Circle handling mode: {st.session_state.config['existing_circle_handling']}")
            
            # Run the matching algorithm with enhanced return values for debugging
            results, matched_circles, unmatched_participants = run_matching_algorithm(
                test_data,  # Use our modified data with the test participant
                st.session_state.config
            )
            
            # Add extensive diagnostic logging to understand data structure
            print("\n🔬🔬🔬 DETAILED RESULT ANALYSIS 🔬🔬🔬")
            print(f"Raw results length: {len(results)}")
            
            # Check for test participants that might be inflating counts
            # First, check if results is a DataFrame or a list of dictionaries
            test_participants = []
            if isinstance(results, pd.DataFrame):
                # If it's a DataFrame, use proper DataFrame filtering
                if 'Encoded ID' in results.columns:
                    mask = results['Encoded ID'].astype(str).str.startswith('99999')
                    test_participants = results[mask]
                    print(f"⚠️ FOUND {len(test_participants)} TEST PARTICIPANTS in results (DataFrame)")
                    if not test_participants.empty:
                        print(f"  First test participant ID: {test_participants.iloc[0]['Encoded ID']}")
            else:
                # If it's a list of dictionaries, use the original approach
                try:
                    test_participants = [r for r in results if isinstance(r, dict) and str(r.get('Encoded ID', '')).startswith('99999')]
                    if test_participants:
                        print(f"⚠️ FOUND {len(test_participants)} TEST PARTICIPANTS in results (List)")
                        print(f"  First test participant: {test_participants[0].get('Encoded ID', 'Unknown')}")
                except Exception as e:
                    print(f"⚠️ Error when checking for test participants: {str(e)}")
                    print(f"  Type of results: {type(results)}")
                    print(f"  Sample item type: {type(results[0]) if results and len(results) > 0 else 'No items'}")
            
            # Check for duplicate Encoded IDs
            if 'Encoded ID' in results.columns:
                total_ids = len(results['Encoded ID'])
                unique_ids = len(results['Encoded ID'].unique())
                print(f"Total IDs: {total_ids}, Unique IDs: {unique_ids}")
                
                if total_ids > unique_ids:
                    print(f"⚠️ FOUND {total_ids - unique_ids} DUPLICATE IDs in results")
                    print("🛠️ Fixing duplicates in results DataFrame")
                    
                    # De-duplicate the results DataFrame
                    results = results.drop_duplicates(subset=['Encoded ID'], keep='first')
                    print(f"✅ After de-duplication: {results.shape[0]} participants (was {total_ids})")
            
            # Count matched vs unmatched
            if 'proposed_NEW_circles_id' in results.columns:
                matched_in_results = len(results[results['proposed_NEW_circles_id'] != 'UNMATCHED'])
                unmatched_in_results = len(results[results['proposed_NEW_circles_id'] == 'UNMATCHED'])
                print(f"From results DataFrame - Matched: {matched_in_results}, Unmatched: {unmatched_in_results}")
            
            # Check the unmatched_participants parameter
            print(f"Unmatched participants parameter length: {len(unmatched_participants)}")
            
            # Check matched circles
            print(f"Matched circles length: {len(matched_circles)}")
            
            # Calculate total expected participants
            total_in_circles = sum(circle.get('member_count', 0) if isinstance(circle, dict) else 0 
                                  for circle in matched_circles)
            print(f"Total participants in all circles (member_count): {total_in_circles}")
            
            # Compare with original data
            original_participant_count = len(test_data)
            print(f"Original data participant count: {original_participant_count}")
            
            # Properly handle test_participants which could be a DataFrame or a list
            test_count = 0
            if isinstance(test_participants, pd.DataFrame):
                test_count = len(test_participants) if not test_participants.empty else 0
            elif isinstance(test_participants, list):
                test_count = len(test_participants)
            elif hasattr(test_participants, '__len__'):  # Handle any other iterable
                test_count = len(test_participants)
            print(f"Test participant adjustment: -{test_count}")
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.matched_circles = matched_circles
            st.session_state.unmatched_participants = unmatched_participants
            st.session_state.exec_time = time.time() - start_time
            
            # Calculate and store diversity score immediately after optimization
            from modules.ui_components import calculate_total_diversity_score
            total_diversity_score = calculate_total_diversity_score(matched_circles, results)
            st.session_state.total_diversity_score = total_diversity_score
            print(f"DEBUG - Calculated diversity score immediately after optimization: {total_diversity_score}")
            
            # DEBUGGING: Check if we actually have eligibility logs in session state
            print(f"After optimization, circle_eligibility_logs contains {len(st.session_state.circle_eligibility_logs)} entries")
            if len(st.session_state.circle_eligibility_logs) == 0:
                print("WARNING: No circle eligibility logs found in session state!")
                
                # Try loading from file
                try:
                    from modules.optimizer_new import load_circle_eligibility_logs_from_file
                    file_logs = load_circle_eligibility_logs_from_file()
                    if file_logs and len(file_logs) > 0:
                        print(f"✅ Loaded {len(file_logs)} logs from file")
                        st.session_state.circle_eligibility_logs = file_logs
                    else:
                        print("ℹ️ No logs found in file backup")
                        # Initialize with empty logs - no test data generation
                        st.session_state.circle_eligibility_logs = {}
                        
                        # Show a warning message to the user
                        st.warning("No circle eligibility data available. This usually means the optimization process didn't generate circle data properly. Please try running the optimization again.")
                except Exception as e:
                    print(f"❌ Error during file operations: {str(e)}")
                    st.session_state.circle_eligibility_logs = {}
                    st.warning("Unable to load circle data. Please try running the optimization again.")
            
            st.success(f"Matching completed in {st.session_state.exec_time:.2f} seconds!")
            st.session_state.active_tab = "Results"
            
    except Exception as e:
        st.error(f"Error during optimization: {str(e)}")
        if st.session_state.config['debug_mode']:
            st.exception(e)

def process_uploaded_file(uploaded_file):
    """Process the uploaded CSV file with participant data"""
    try:
        # Force clear any cached values
        st.cache_data.clear()
        with st.spinner("Processing data..."):
            # Load and validate data
            df, validation_errors, deduplication_messages = load_data(uploaded_file)
            st.session_state.df = df
            st.session_state.validation_errors = validation_errors
            st.session_state.deduplication_messages = deduplication_messages
            
            # ENHANCED DIAGNOSTICS: Print detailed info about the input data
            print("\n🔬🔬🔬 SUPER DETAILED DATA DIAGNOSTICS IN PROCESS_UPLOADED_FILE 🔬🔬🔬")
            print(f"🔬 DataFrame shape: {df.shape}")
            print(f"🔬 DataFrame columns: {df.columns.tolist()}")
            
            # Count participants by status for debugging
            if 'Status' in df.columns:
                status_counts = df['Status'].value_counts().to_dict()
                st.session_state.status_counts = status_counts
                print(f"🔬 Status counts: {status_counts}")
            else:
                print("🔬 'Status' column not found")
                
            # Check circle ID column
            circle_columns = ['Current_Circle_ID', 'current_circles_id', 'Current Circle ID']
            found_col = None
            for col in circle_columns:
                if col in df.columns:
                    found_col = col
                    break
                    
            if found_col:
                print(f"🔬 Found circle ID column: {found_col}")
                
                # Count non-null values
                non_null_count = df[~df[found_col].isna()].shape[0]
                print(f"🔬 Participants with non-null {found_col}: {non_null_count}")
                
                # Check CURRENT-CONTINUING participants with circle IDs
                if 'Status' in df.columns:
                    continuing = df[df['Status'] == 'CURRENT-CONTINUING']
                    print(f"🔬 CURRENT-CONTINUING participants: {len(continuing)}")
                    
                    with_circles = continuing[~continuing[found_col].isna()]
                    print(f"🔬 CURRENT-CONTINUING with circle IDs: {len(with_circles)}")
                    
                    if len(with_circles) > 0:
                        unique_circles = with_circles[found_col].unique()
                        print(f"🔬 Unique circle IDs: {len(unique_circles)}")
                        print(f"🔬 First 10 circle IDs: {list(unique_circles)[:10]}")
                        
                        # Get circle member counts
                        circle_counts = with_circles[found_col].value_counts()
                        print(f"🔬 Circle member counts (top 10):")
                        for circle, count in circle_counts.head(10).items():
                            print(f"   {circle}: {count} members")
                            
                        # Check for any problematic circle patterns
                        problematic_patterns = ['IP-TEST', 'IP-NEW-TES']
                        for pattern in problematic_patterns:
                            problem_circles = [c for c in unique_circles if pattern in c]
                            if problem_circles:
                                print(f"🚨 WARNING: Found {len(problem_circles)} circles with test pattern '{pattern}'")
                                print(f"   Circle IDs: {problem_circles}")
                                print(f"   These may be test circles and should be removed from production data!")
            else:
                print("🔬 No valid circle ID column found")
                
            print("🔬🔬🔬 END OF SUPER DETAILED DATA DIAGNOSTICS 🔬🔬🔬\n")
            
            # Display validation errors if any
            if len(validation_errors) > 0:
                st.warning(f"Found {len(validation_errors)} validation issues:")
                for error in validation_errors[:5]:  # Show first 5 errors
                    st.write(f"- {error}")
                if len(validation_errors) > 5:
                    st.write(f"...and {len(validation_errors) - 5} more issues.")
            
            # Display just the count of fixed duplicate Encoded IDs
            if len(deduplication_messages) > 0:
                st.warning(f"Found and fixed {len(deduplication_messages)} duplicate Encoded IDs")
            
            # Process and normalize data - pass debug_mode from session state if available
            debug_mode = st.session_state.get('debug_mode', False)
            
            processed_data = process_data(df, debug_mode=debug_mode)
            
            # Check for "Moving out" status records that will be excluded
            moving_out_count = 0
            if 'Status' in df.columns:
                for status in df['Status']:
                    if isinstance(status, str) and "MOVING OUT" in status.upper():
                        moving_out_count += 1
                
                if moving_out_count > 0:
                    st.info(f"{moving_out_count} records with 'Moving Out' status will be excluded from matching")
            
            normalized_data = normalize_data(processed_data, debug_mode=debug_mode)
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
            # Keep only Debug Mode and set other options to fixed values
            st.session_state.config['min_circle_size'] = 5  # Fixed value
            # CRITICAL FIX: Allow existing_circle_handling to be set by the user
            # This gives flexibility to allow NEW participants to be matched with existing circles
            existing_circle_handling = st.radio(
                "Existing Circle Handling", 
                options=['optimize', 'preserve', 'dissolve'],
                index=0,  # Default to 'optimize'
                help="'optimize' (RECOMMENDED) allows new participants to join existing circles while keeping CURRENT-CONTINUING members in place. 'preserve' keeps existing circles intact but prevents NEW members from joining them. 'dissolve' breaks up all circles and creates new ones."
            )
            st.session_state.config['existing_circle_handling'] = existing_circle_handling
            st.session_state.config['enable_host_requirement'] = True  # Fixed value
            
            # Only show Debug Mode as a configurable option
            st.session_state.config['debug_mode'] = st.checkbox(
                "Debug Mode", 
                value=st.session_state.config['debug_mode'],
                help="Enable to see detailed logs and diagnostic information"
            )
            
            # Add run button once data is loaded
            if st.button("Run Matching Algorithm", key="match_run_algorithm_button"):
                run_optimization()
                
            # Display results if available
            if st.session_state.results is not None:
                # We don't need to render the overview in the match tab
                # render_results_overview() - removed to avoid duplicate charts
                
                # Display stats about the matching
                if 'results' in st.session_state and st.session_state.results is not None:
                    results_df = st.session_state.results
                    total_participants = len(results_df)
                    
                    # EXTENSIVE DIAGNOSTIC LOGGING FOR PARTICIPANT COUNT ISSUE
                    print("\n🔍🔍🔍 MATCH COUNT DIAGNOSTICS 🔍🔍🔍")
                    print(f"Raw results_df shape: {results_df.shape}")
                    
                    # Check for duplicate IDs which could be inflating the count
                    if 'Encoded ID' in results_df.columns:
                        total_ids = len(results_df['Encoded ID'])
                        unique_ids = len(results_df['Encoded ID'].unique())
                        print(f"Total IDs: {total_ids}, Unique IDs: {unique_ids}")
                        if total_ids > unique_ids:
                            print(f"⚠️ FOUND {total_ids - unique_ids} DUPLICATE IDs!")
                            # Show examples of duplicates
                            duplicate_mask = results_df.duplicated(subset=['Encoded ID'], keep=False)
                            duplicates = results_df[duplicate_mask]
                            if len(duplicates) > 0:
                                print(f"Examples of duplicated IDs:")
                                dup_examples = duplicates['Encoded ID'].unique()[:5]
                                for dup_id in dup_examples:
                                    dup_rows = results_df[results_df['Encoded ID'] == dup_id]
                                    print(f"  ID {dup_id} appears {len(dup_rows)} times")
                                    if len(dup_rows) > 1:
                                        for i, (_, row) in enumerate(dup_rows.iterrows()):
                                            circle_id = row.get('proposed_NEW_circles_id', 'N/A')
                                            status = row.get('Status', 'N/A')
                                            print(f"    Instance {i+1}: Circle={circle_id}, Status={status}")
                            
                            # CRITICAL FIX: De-duplicate the results dataframe to fix inflated participant count
                            print("🛠️ APPLYING FIX: Removing duplicate Encoded IDs from results dataframe")
                            # Keep the first occurrence of each Encoded ID
                            results_df = results_df.drop_duplicates(subset=['Encoded ID'], keep='first')
                            print(f"✅ After de-duplication: {results_df.shape[0]} participants (was {total_ids})")
                            
                            # Update the session state with the de-duplicated dataframe
                            st.session_state.results = results_df
                            # Recalculate total participants
                            total_participants = len(results_df)
                    
                    # FIXED: More accurate calculation of matched participants
                    # Only count non-empty circle IDs to avoid counting filtered records
                    matched_count = 0
                    unmatched_count = 0
                    
                    if 'proposed_NEW_circles_id' in results_df.columns:
                        # Check all values for diagnostics
                        circle_values = results_df['proposed_NEW_circles_id'].value_counts().to_dict()
                        print(f"Circle ID values count:")
                        
                        # Group by UNMATCHED, nan/null, and actual circles
                        unmatched_count = circle_values.get('UNMATCHED', 0)
                        null_count = results_df['proposed_NEW_circles_id'].isna().sum()
                        circle_count = sum(v for k, v in circle_values.items() 
                                           if k != 'UNMATCHED' and not pd.isna(k))
                        
                        print(f"  Assigned to circles: {circle_count}")
                        print(f"  UNMATCHED: {unmatched_count}")
                        print(f"  Null/NaN: {null_count}")
                        
                        # Filter out any null or empty values
                        notna_mask = results_df['proposed_NEW_circles_id'].notna()
                        valid_circle_mask = notna_mask & (results_df['proposed_NEW_circles_id'] != 'UNMATCHED')
                        matched_count = len(results_df[valid_circle_mask])
                        unmatched_count = len(results_df[results_df['proposed_NEW_circles_id'] == 'UNMATCHED'])
                        
                        # Compare with actual circle data
                        circles_participant_count = 0
                        if 'matched_circles' in st.session_state:
                            circles = st.session_state.matched_circles
                            
                            # Handle different types properly
                            for circle in circles:
                                if isinstance(circle, dict):
                                    # Dictionary circle
                                    circles_participant_count += circle.get('member_count', 0)
                                elif isinstance(circle, pd.DataFrame):
                                    # DataFrame circle
                                    if 'member_count' in circle.columns:
                                        circles_participant_count += circle['member_count'].sum()
                                elif isinstance(circle, str):
                                    # String entries (circle IDs) - can't get member count directly
                                    print(f"⚠️ Found string circle entry: {circle}")
                                else:
                                    # Other types - log for debugging
                                    print(f"⚠️ Unknown circle type: {type(circle)}")
                                    
                            print(f"Total participants in matched_circles: {circles_participant_count}")
                    
                    # Check if we have test participants and remove them
                    if 'Encoded ID' in results_df.columns:
                        try:
                            # Identify test participants (IDs starting with 99999)
                            mask = results_df['Encoded ID'].astype(str).str.startswith('99999')
                            test_participants_df = results_df[mask]
                            
                            if len(test_participants_df) > 0:
                                print(f"⚠️ FOUND {len(test_participants_df)} TEST PARTICIPANTS that might be inflating counts")
                                print(f"   Test participant IDs: {test_participants_df['Encoded ID'].tolist()}")
                                
                                # Remove test participants from results
                                results_df = results_df[~mask]
                                print(f"✅ FILTERED OUT {len(test_participants_df)} test participants from results")
                                print(f"   New result count: {len(results_df)} (was {len(results_df) + len(test_participants_df)})")
                                
                                # Update the session state with filtered results
                                st.session_state.results = results_df
                                # Recalculate total participants
                                total_participants = len(results_df)
                        except Exception as e:
                            print(f"⚠️ Error filtering test participants: {str(e)}")
                            print(f"Type of Encoded ID column: {results_df['Encoded ID'].dtype}")
                    
                    # Log final counts 
                    print(f"FINAL COUNTS: Total={total_participants}, Matched={matched_count}, Unmatched={unmatched_count}")
                    print(f"Match rate: {(matched_count / total_participants) * 100:.1f}%")
                    print("🔍🔍🔍 END MATCH COUNT DIAGNOSTICS 🔍🔍🔍\n")
                    print(f"  Matched count: {matched_count}")
                    print(f"  Unmatched count: {unmatched_count}")
                    
                    # Create columns for the metrics (now 3 columns instead of 4)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
                            st.metric("Circles Created", len(st.session_state.matched_circles))
                    
                    with col2:
                        st.metric("Participants Matched", matched_count)
                        
                    with col3:
                        if total_participants > 0:
                            match_rate = (matched_count / total_participants) * 100
                            st.metric("Match Success Rate", f"{match_rate:.1f}%")
                
                st.subheader("Circle Composition")
                
                # Display circle table with specified columns
                if ('matched_circles' in st.session_state and 
                    st.session_state.matched_circles is not None and 
                    (isinstance(st.session_state.matched_circles, list) or 
                     not (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty))):
                    
                    # Get the data - handle the case where matched_circles might be a list of dictionaries
                    if isinstance(st.session_state.matched_circles, pd.DataFrame):
                        circles_df = st.session_state.matched_circles.copy()
                    elif isinstance(st.session_state.matched_circles, list):
                        # Convert list of dictionaries to DataFrame
                        try:
                            circles_df = pd.DataFrame(st.session_state.matched_circles)
                        except Exception as e:
                            st.warning(f"Could not convert circles to DataFrame: {str(e)}")
                            print(f"Error converting matched_circles to DataFrame: {str(e)}")
                            circles_df = pd.DataFrame()  # Empty DataFrame as fallback
                    else:
                        # Unknown type
                        st.warning(f"Unexpected matched_circles type: {type(st.session_state.matched_circles)}")
                        circles_df = pd.DataFrame()  # Empty DataFrame as fallback
                    
                    # Extract key columns as specified
                    display_cols = ['circle_id', 'region', 'subregion', 'meeting_time', 'member_count', 'new_members', 'max_additions', 'always_hosts', 'sometimes_hosts']
                    
                    # Filter to only include columns that exist in the dataframe
                    if hasattr(circles_df, 'columns'):
                        existing_cols = [col for col in display_cols if col in circles_df.columns]
                    else:
                        existing_cols = []
                    
                    if existing_cols:
                        display_df = circles_df[existing_cols].copy()
                        
                        # Rename columns for display
                        display_df.columns = [col.replace('_', ' ').title() for col in existing_cols]
                        
                        # Sort by circle ID if available
                        if 'Circle Id' in display_df.columns:
                            display_df = display_df.sort_values('Circle Id')
                        
                        # Show the table
                        st.dataframe(display_df, use_container_width=True)
                    else:
                        st.warning("Circle data doesn't contain the expected columns.")
                else:
                    st.warning("No matching results available. Please run the matching algorithm first.")
                
                # Display unmatched participants
                st.subheader("Unmatched Participants")
                
                if 'results' in st.session_state and st.session_state.results is not None:
                    results_df = st.session_state.results
                    
                    if 'proposed_NEW_circles_id' in results_df.columns:
                        unmatched_df = results_df[results_df['proposed_NEW_circles_id'] == 'UNMATCHED'].reset_index(drop=True)
                        
                        if len(unmatched_df) == 0:
                            st.success("All participants were successfully matched!")
                        else:
                            st.write(f"Total unmatched: {len(unmatched_df)} participants")
                            
                            # Create a display DataFrame with just the columns we need
                            # We'll get the specific columns from the Results CSV by position
                            
                            # Get all column names from the DataFrame for debugging
                            all_columns = list(unmatched_df.columns)
                            
                            # Define the display columns based on their position in the CSV
                            display_cols = []
                            
                            # Encoded ID (column 3 in CSV) - Identify by name
                            encoded_id_col = None
                            if 'Encoded ID' in all_columns:
                                encoded_id_col = 'Encoded ID'
                                display_cols.append(encoded_id_col)
                            
                            # Try to find Derived_Region (column 32 in CSV file)
                            # First approach: Try to find by column name
                            region_col = None
                            region_candidates = ['Derived_Region', 'Derived Region', 'Region Requested', 'Region']
                            for col in region_candidates:
                                if col in all_columns:
                                    region_col = col
                                    display_cols.append(region_col)
                                    break
                                    
                            # Alternative approach: Enable debug to see all column names
                            # Uncomment to see columns when debugging
                            # st.write("All columns in dataframe:", all_columns)
                            
                            # Reason Unmatched (column 5 in CSV)
                            reason_col = None
                            if 'unmatched_reason' in all_columns:
                                reason_col = 'unmatched_reason'
                                display_cols.append(reason_col)
                            
                            # Find choice columns - these need to be identified by position or pattern
                            
                            # 1st, 2nd, 3rd Choice Location columns
                            location_cols = []
                            for col in all_columns:
                                if any(pattern in col.lower() for pattern in [
                                    "location_choice_1", "meeting_location_1", "location_pref_1", 
                                    "1st_choice_location", "first_choice_location"
                                ]):
                                    location_cols.append(col)
                                    break
                            
                            for col in all_columns:
                                if any(pattern in col.lower() for pattern in [
                                    "location_choice_2", "meeting_location_2", "location_pref_2", 
                                    "2nd_choice_location", "second_choice_location"
                                ]):
                                    location_cols.append(col)
                                    break
                                    
                            for col in all_columns:
                                if any(pattern in col.lower() for pattern in [
                                    "location_choice_3", "meeting_location_3", "location_pref_3", 
                                    "3rd_choice_location", "third_choice_location"
                                ]):
                                    location_cols.append(col)
                                    break
                            
                            # 1st, 2nd, 3rd Choice Time columns
                            time_cols = []
                            for col in all_columns:
                                if any(pattern in col.lower() for pattern in [
                                    "time_choice_1", "meeting_time_1", "time_pref_1", 
                                    "1st_choice_time", "first_choice_time"
                                ]):
                                    time_cols.append(col)
                                    break
                            
                            for col in all_columns:
                                if any(pattern in col.lower() for pattern in [
                                    "time_choice_2", "meeting_time_2", "time_pref_2", 
                                    "2nd_choice_time", "second_choice_time"
                                ]):
                                    time_cols.append(col)
                                    break
                                    
                            for col in all_columns:
                                if any(pattern in col.lower() for pattern in [
                                    "time_choice_3", "meeting_time_3", "time_pref_3", 
                                    "3rd_choice_time", "third_choice_time"
                                ]):
                                    time_cols.append(col)
                                    break
                            
                            # As a fallback, try to find columns by pattern if positional search failed
                            if not location_cols or not time_cols:
                                # Find all location and time columns
                                pattern_location_cols = []
                                pattern_time_cols = []
                                
                                for col in all_columns:
                                    if any(pattern in col.lower() for pattern in ['location_choice', 'meeting_location', 'location_pref']):
                                        pattern_location_cols.append(col)
                                    if any(pattern in col.lower() for pattern in ['time_choice', 'meeting_time', 'time_pref']):
                                        pattern_time_cols.append(col)
                                
                                # Sort them to ensure they're in order (1st, 2nd, 3rd)
                                pattern_location_cols.sort()
                                pattern_time_cols.sort()
                                
                                # Use up to 3 of each
                                location_cols = pattern_location_cols[:3]
                                time_cols = pattern_time_cols[:3]
                            
                            # Add the location and time columns to display_cols, interleaving them
                            for i in range(min(3, max(len(location_cols), len(time_cols)))):
                                if i < len(location_cols):
                                    display_cols.append(location_cols[i])
                                if i < len(time_cols):
                                    display_cols.append(time_cols[i])
                            
                            # Remove any columns that don't exist in the dataframe
                            display_cols = [col for col in display_cols if col in all_columns]
                            
                            # Debug info
                            # st.write(f"All columns: {all_columns}")
                            # st.write(f"Display columns: {display_cols}")
                            
                            if display_cols:
                                st.dataframe(unmatched_df[display_cols], use_container_width=True)
                    else:
                        st.warning("Results data doesn't contain matching information.")
                
                # Download button for results - placed at bottom of page
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
