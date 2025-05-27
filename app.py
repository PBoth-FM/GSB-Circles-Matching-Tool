import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Import modules (keeping only essential ones for the unified reporting system)
from modules.data_loader import load_data
from modules.data_processor import process_data, normalize_data
from modules.optimizer import run_matching_algorithm

def get_unified_dataframe():
    """Generate the same post-processed dataframe used for CSV export"""
    if 'results' not in st.session_state or st.session_state.results is None:
        return pd.DataFrame()
    
    results_df = st.session_state.results.copy()
    
    # Apply post-processing fixes for circle IDs
    try:
        from utils.circle_id_postprocessor import apply_postprocessing_fixes
        results_df = apply_postprocessing_fixes(results_df)
        
        # Update session state with post-processed data
        st.session_state.results = results_df
        
    except Exception as e:
        print(f"Post-processing failed: {e}")
    
    return results_df

def main():
    st.set_page_config(
        page_title="Circle Matching System",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 Circle Matching System")
    st.markdown("---")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload participant data",
            type=['csv'],
            help="Upload a CSV file with participant information"
        )
        
        if uploaded_file is not None:
            # Load and process data
            data, errors, dedup_messages = load_data(uploaded_file)
            
            if errors:
                st.error("Data validation errors:")
                for error in errors:
                    st.write(f"• {error}")
            else:
                st.success(f"✅ Loaded {len(data)} participants")
                
                # Store in session state
                st.session_state.raw_data = data
                
                # Auto-run matching algorithm with fixed parameters
                with st.spinner("Running matching algorithm..."):
                    try:
                        # Process data
                        processed_data = process_data(data)
                        normalized_data = normalize_data(processed_data)
                        
                        # Fixed configuration parameters
                        config = {
                            'min_circle_size': 5,  # Always 5 for new circles
                            'enable_host_requirement': True,  # Will be handled per circle type
                            'existing_circle_handling': 'optimize'  # Always optimize
                        }
                        
                        # Run matching
                        results, matched_circles, unmatched = run_matching_algorithm(
                            normalized_data, config
                        )
                        
                        # Store results
                        st.session_state.results = results
                        st.session_state.matched_circles = matched_circles
                        st.session_state.unmatched = unmatched
                        
                        st.success("✅ Matching completed successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error running matching algorithm: {str(e)}")
                        st.write("Please check your data and try again.")
    
    # Main content area
    if 'results' in st.session_state and st.session_state.results is not None:
        st.header("📊 Matching Results")
        
        # Get unified dataframe for consistent reporting
        unified_df = get_unified_dataframe()
        circles_in_unified = unified_df[unified_df['proposed_NEW_circles_id'].notna()]
        
        if len(circles_in_unified) > 0:
            # Calculate metrics using unified data
            total_participants = len(unified_df)
            matched_participants = len(circles_in_unified)
            num_circles_created = circles_in_unified['proposed_NEW_circles_id'].nunique()
            match_rate = (matched_participants / total_participants) * 100 if total_participants > 0 else 0
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Participants", total_participants)
            with col2:
                st.metric("Matched Participants", matched_participants)
            with col3:
                st.metric("Circles Created", num_circles_created)
            with col4:
                st.metric("Match Success Rate", f"{match_rate:.1f}%")
            
            # Circle Size Distribution
            st.subheader("Circle Size Distribution")
            if len(circles_in_unified) > 0:
                # Calculate circle sizes from unified dataframe
                circle_size_counts = circles_in_unified['proposed_NEW_circles_id'].value_counts()
                size_counts = circle_size_counts.value_counts().sort_index()
                
                # Create DataFrame for plotting
                size_df = pd.DataFrame({
                    'Circle Size': size_counts.index,
                    'Number of Circles': size_counts.values
                })
                
                # Create histogram
                fig = px.bar(
                    size_df,
                    x='Circle Size',
                    y='Number of Circles',
                    title='Distribution of Circle Sizes',
                    text='Number of Circles',
                    color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
                )
                
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    xaxis=dict(
                        title="Number of Members",
                        tickmode='linear',
                        dtick=1
                    ),
                    yaxis=dict(title="Number of Circles"),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show summary statistics
                circle_sizes = circle_size_counts.values
                avg_size = circle_sizes.mean()
                median_size = pd.Series(circle_sizes).median()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Circle Size", f"{avg_size:.1f}")
                with col2:
                    st.metric("Median Circle Size", f"{median_size:.0f}")
            else:
                st.info("No circle data available for size distribution")
            
            # Circle Composition Table
            st.subheader("Circle Composition")
            
            if len(circles_in_unified) > 0:
                # Create Circle Composition table from unified dataframe
                circle_composition_data = []
                
                # Group by circle to create composition summary
                for circle_id in unified_df['proposed_NEW_circles_id'].dropna().unique():
                    circle_members = unified_df[unified_df['proposed_NEW_circles_id'] == circle_id]
                    
                    # Get circle metadata
                    first_member = circle_members.iloc[0]
                    region = first_member.get('proposed_NEW_Region', first_member.get('Derived_Region', 'Unknown'))
                    subregion = first_member.get('proposed_NEW_Subregion', first_member.get('Subregion', 'Unknown'))
                    meeting_time = first_member.get('proposed_NEW_DayTime', first_member.get('Meeting_Time', 'Unknown'))
                    
                    # Count members and new members
                    member_count = len(circle_members)
                    new_members = len(circle_members[circle_members['Status'].str.contains('NEW', na=False)])
                    
                    # Calculate max additions (handle string "None" values)
                    max_additions_values = circle_members['co_leader_max_new_members'].dropna()
                    # Filter out string "None" values and convert to numeric
                    numeric_values = []
                    for val in max_additions_values:
                        if val != 'None' and str(val).lower() != 'none':
                            try:
                                numeric_values.append(int(float(val)))
                            except (ValueError, TypeError):
                                continue
                    max_additions = min(numeric_values) if numeric_values else 0
                    
                    circle_composition_data.append({
                        'Circle Id': circle_id,
                        'Region': region,
                        'Subregion': subregion,
                        'Meeting Time': meeting_time,
                        'Member Count': member_count,
                        'New Members': new_members,
                        'Max Additions': max_additions
                    })
                
                # Create DataFrame and display with column resizing
                composition_df = pd.DataFrame(circle_composition_data)
                
                # Display the unified Circle Composition table with resizable columns
                st.dataframe(
                    composition_df,
                    use_container_width=True,
                    column_config={
                        "Circle Id": st.column_config.TextColumn(width="medium"),
                        "Region": st.column_config.TextColumn(width="medium"),
                        "Subregion": st.column_config.TextColumn(width="large"),
                        "Meeting Time": st.column_config.TextColumn(width="large"),
                        "Member Count": st.column_config.NumberColumn(width="small"),
                        "New Members": st.column_config.NumberColumn(width="small"),
                        "Max Additions": st.column_config.NumberColumn(width="small")
                    }
                )
            else:
                st.info("No circle data available for composition table")
            
            # Unmatched Participants
            st.subheader("Unmatched Participants")
            
            if 'results' in st.session_state and st.session_state.results is not None:
                results_df = st.session_state.results
                
                # Filter unmatched participants
                unmatched_df = results_df[
                    (results_df['proposed_NEW_circles_id'].isna()) | 
                    (results_df['proposed_NEW_circles_id'] == 'UNMATCHED')
                ].copy()
                
                if len(unmatched_df) > 0:
                    # Select relevant columns for display
                    unmatched_display_cols = ['Encoded ID', 'First Name', 'Last Name', 'Derived_Region', 
                                            'Subregion', 'Status', 'Meeting_Time', 'Location_Preference']
                    
                    # Filter to existing columns
                    existing_unmatched_cols = [col for col in unmatched_display_cols if col in unmatched_df.columns]
                    
                    if existing_unmatched_cols:
                        unmatched_display = unmatched_df[existing_unmatched_cols].copy()
                        
                        # Rename columns for better display
                        column_renames = {
                            'Derived_Region': 'Region',
                            'Meeting_Time': 'Meeting Time',
                            'Location_Preference': 'Location Preference'
                        }
                        
                        for old_name, new_name in column_renames.items():
                            if old_name in unmatched_display.columns:
                                unmatched_display = unmatched_display.rename(columns={old_name: new_name})
                        
                        # Display with column resizing
                        st.dataframe(
                            unmatched_display,
                            use_container_width=True,
                            column_config={
                                "Encoded ID": st.column_config.TextColumn(width="small"),
                                "First Name": st.column_config.TextColumn(width="medium"),
                                "Last Name": st.column_config.TextColumn(width="medium"),
                                "Region": st.column_config.TextColumn(width="medium"),
                                "Subregion": st.column_config.TextColumn(width="large"),
                                "Status": st.column_config.TextColumn(width="medium"),
                                "Meeting Time": st.column_config.TextColumn(width="large"),
                                "Location Preference": st.column_config.TextColumn(width="large")
                            }
                        )
                        
                        st.info(f"Total unmatched participants: {len(unmatched_df)}")
                    else:
                        st.warning("Could not display unmatched participants - missing expected columns.")
                else:
                    st.success("🎉 All participants were successfully matched to circles!")
            else:
                st.warning("No results data available for unmatched participants display.")
            
            # CSV Download Section
            st.subheader("Download Results")
            
            if 'results' in st.session_state and st.session_state.results is not None:
                # Use the unified dataframe for CSV download (same data as displayed)
                csv_data = get_unified_dataframe()
                
                if len(csv_data) > 0:
                    csv_str = csv_data.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_str,
                        file_name=f"circle_matching_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download the complete results including all participant assignments and circle details"
                    )
                    
                    st.info(f"📊 CSV contains {len(csv_data)} participant records with complete assignment details")
                else:
                    st.warning("No data available for CSV download")
            else:
                st.warning("No results available for download")
        
        else:
            st.warning("No matching results to display. Please check the algorithm output.")
    
    else:
        st.info("👆 Please upload a CSV file and run the matching algorithm to see results.")

if __name__ == "__main__":
    main()