import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render_match_tab():
    """Render the main matching tab content"""
    st.subheader("Upload Participant Data")
    
    # File uploader functionality is now handled directly in app.py through match_tab_callback
    # This function remains as a placeholder for imported reference
    st.info("Please upload a CSV file with participant data to begin.")
    
    # Show file format instructions
    with st.expander("CSV File Format Requirements"):
        st.markdown("""
            ### Required Columns
            - **Encoded ID**: Unique identifier for each participant
            - **Status**: Current status (e.g., 'CURRENT-CONTINUING', 'NEW', 'MOVING OUT')
            - **Requested_Region**: Preferred region
            - **first_choice_location**: First choice location/subregion
            - **first_choice_time**: First choice meeting time
            - **second_choice_location**: Second choice location/subregion (optional)
            - **second_choice_time**: Second choice meeting time (optional)
            - **third_choice_location**: Third choice location/subregion (optional)
            - **third_choice_time**: Third choice meeting time (optional)
            - **host**: Hosting preference ('Always', 'Sometimes', 'Never Host', 'n/a', or blank)
            
            ### Sample Format
            ```
            Encoded ID,Status,Requested_Region,first_choice_location,first_choice_time,...
            123456789,NEW,South Florida,Miami,Monday (Evening),...
            987654321,CURRENT-CONTINUING,Boston,Cambridge/Somerville,Saturday (Morning),...
            ```
            """)

def render_details_tab():
    """Render the details tab content"""
    if 'results' not in st.session_state or st.session_state.results is None:
        st.info("Run the matching algorithm first to see detailed results.")
        return
    
    st.subheader("Match Details")
    
    # Performance metrics
    if 'exec_time' in st.session_state:
        st.metric("Execution Time", f"{st.session_state.exec_time:.2f} seconds")
    
    # Create tabs for different detail views
    detail_tab1, detail_tab2, detail_tab3 = st.tabs(["Circle Details", "Participant Details", "Visualizations"])
    
    with detail_tab1:
        render_circle_details()
        
    with detail_tab2:
        render_participant_details()
        
    with detail_tab3:
        render_visualizations()

def render_debug_tab():
    """Render the debug tab content"""
    st.subheader("Debug Information")
    
    if 'config' in st.session_state and st.session_state.config.get('debug_mode', False):
        # Display status counts if available
        if 'status_counts' in st.session_state:
            st.write("### Participant Status Counts")
            status_df = pd.DataFrame({
                'Status': list(st.session_state.status_counts.keys()),
                'Count': list(st.session_state.status_counts.values())
            })
            st.dataframe(status_df, use_container_width=True)
            
        # Display data samples if available
        if 'df' in st.session_state:
            st.write("### Raw Data Preview")
            st.dataframe(st.session_state.df.head(), use_container_width=True)
            
            # Show column names for debugging
            st.write("### Column Names")
            st.write(list(st.session_state.df.columns))
            
        if 'processed_data' in st.session_state:
            st.write("### Processed Data Preview")
            st.dataframe(st.session_state.processed_data.head(), use_container_width=True)
            
            # Count participants by status in processed data
            if 'Status' in st.session_state.processed_data.columns:
                st.write("### Processed Data Status Counts")
                processed_status_counts = st.session_state.processed_data['Status'].value_counts().reset_index()
                processed_status_counts.columns = ['Status', 'Count']
                st.dataframe(processed_status_counts, use_container_width=True)
    
        # Time compatibility tester
        st.write("### Time Compatibility Tester")
        col1, col2 = st.columns(2)
        
        with col1:
            time1 = st.text_input("Time Preference 1", "Monday (Evening)")
        
        with col2:
            time2 = st.text_input("Time Preference 2", "Monday-Thursday (Evening)")
        
        if st.button("Test Compatibility"):
            # This would call a function to test time compatibility
            st.write("Compatibility result would be shown here")
            
        # Region code mapping tester
        st.write("### Region Code Mapping Tester")
        
        region_input = st.text_input("Region Input", "South Florida")
        if st.button("Test Region Normalization"):
            from utils.normalization import normalize_regions
            normalized = normalize_regions(region_input)
            st.write(f"Normalized: {normalized}")
            
        subregion_input = st.text_input("Subregion Input", "Miami")
        if st.button("Test Subregion Normalization"):
            from utils.normalization import normalize_subregions
            normalized = normalize_subregions(subregion_input)
            st.write(f"Normalized: {normalized}")
        
        # Algorithm logs
        if 'optimization_logs' in st.session_state:
            st.write("### Optimization Logs")
            st.text_area("Logs", st.session_state.optimization_logs, height=300)
    else:
        st.info("Enable debug mode in the configuration panel to access debug tools.")

def render_results_overview():
    """Render the results overview section"""
    if 'results' not in st.session_state or st.session_state.results is None:
        return
    
    # Results metrics
    results = st.session_state.results
    total_participants = len(results)
    matched_participants = len(results[results['proposed_NEW_circles_id'] != "UNMATCHED"])
    unmatched_participants = total_participants - matched_participants
    match_rate = (matched_participants / total_participants * 100) if total_participants > 0 else 0
    
    st.subheader("Results Overview")
    
    # Display deduplication messages if any exist
    if 'deduplication_messages' in st.session_state and st.session_state.deduplication_messages:
        with st.expander(f"⚠️ {len(st.session_state.deduplication_messages)} Duplicate Encoded IDs were detected and fixed", expanded=True):
            for message in st.session_state.deduplication_messages[:5]:  # Show first 5 messages
                st.write(f"- {message}")
            if len(st.session_state.deduplication_messages) > 5:
                st.write(f"...and {len(st.session_state.deduplication_messages) - 5} more duplicates fixed.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Participants", total_participants)
    
    with col2:
        st.metric("Matched", matched_participants)
    
    with col3:
        st.metric("Unmatched", unmatched_participants)
    
    with col4:
        st.metric("Match Rate", f"{match_rate:.1f}%")
    
    # If circles data is available
    if 'matched_circles' in st.session_state and not st.session_state.matched_circles.empty:
        circles = st.session_state.matched_circles
        
        # Create a histogram of circle sizes
        if 'member_count' in circles.columns:
            st.subheader("Circle Size Distribution")
            
            # Create a histogram using plotly
            import plotly.express as px
            
            # Count circles by size
            size_counts = circles['member_count'].value_counts().reset_index()
            size_counts.columns = ['Circle Size', 'Count']
            size_counts = size_counts.sort_values('Circle Size')
            
            # Create bar chart
            fig = px.bar(
                size_counts,
                x='Circle Size',
                y='Count',
                title='Distribution of Circle Sizes',
                labels={'Count': 'Number of Circles', 'Circle Size': 'Number of Members'},
                text='Count'  # Show the count values on bars
            )
            
            fig.update_traces(textposition='outside')
            fig.update_layout(xaxis_title="Number of Members", yaxis_title="Number of Circles")
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)

def render_circle_table():
    """Render the circle composition table"""
    if 'matched_circles' not in st.session_state or st.session_state.matched_circles.empty:
        st.info("No circles have been formed yet.")
        return
    
    circles = st.session_state.matched_circles
    
    # Add filter options
    col1, col2 = st.columns(2)
    
    # Initialize default values
    selected_region = 'All'
    selected_subregion = 'All'
    
    with col1:
        if 'region' in circles.columns:
            regions = ['All'] + sorted(circles['region'].unique().tolist())
            selected_region = st.selectbox("Filter by Region", regions)
    
    with col2:
        if 'subregion' in circles.columns:
            subregions = ['All'] + sorted(circles['subregion'].unique().tolist())
            selected_subregion = st.selectbox("Filter by Subregion", subregions)
    
    # Apply filters
    filtered_circles = circles.copy()
    
    if 'region' in circles.columns and selected_region != 'All':
        filtered_circles = filtered_circles[filtered_circles['region'] == selected_region]
    
    if 'subregion' in circles.columns and selected_subregion != 'All':
        filtered_circles = filtered_circles[filtered_circles['subregion'] == selected_subregion]
    
    # Display the filtered circles
    if not filtered_circles.empty:
        # Add new_members column (for new circles, this equals member_count)
        filtered_circles['new_members'] = filtered_circles['member_count']
        
        # Display the count of filtered circles
        st.write(f"Showing {len(filtered_circles)} circles")
        
        # Display the dataframe with the updated column order
        st.dataframe(
            filtered_circles,
            use_container_width=True,
            hide_index=True,
            column_order=["circle_id", "region", "subregion", "meeting_time", "member_count", "new_members", "always_hosts", "sometimes_hosts"]
        )
    else:
        st.info("No circles match the selected filters.")

def render_unmatched_table():
    """Render the unmatched participants table"""
    if 'unmatched_participants' not in st.session_state or st.session_state.unmatched_participants.empty:
        st.info("All participants have been matched.")
        return
    
    unmatched = st.session_state.unmatched_participants
    
    # Add filter options
    col1, col2 = st.columns(2)
    
    # Initialize default values
    selected_reason = 'All'
    selected_region = 'All'
    
    with col1:
        if 'unmatched_reason' in unmatched.columns:
            reasons = ['All'] + sorted(unmatched['unmatched_reason'].unique().tolist())
            selected_reason = st.selectbox("Filter by Reason", reasons)
    
    with col2:
        if 'Requested_Region' in unmatched.columns:
            regions = ['All'] + sorted(unmatched['Requested_Region'].unique().tolist())
            selected_region = st.selectbox("Filter by Requested Region", regions, key="unmatched_region")
    
    # Apply filters
    filtered_unmatched = unmatched.copy()
    
    if 'unmatched_reason' in unmatched.columns and selected_reason != 'All':
        filtered_unmatched = filtered_unmatched[filtered_unmatched['unmatched_reason'] == selected_reason]
    
    if 'Requested_Region' in unmatched.columns and selected_region != 'All':
        filtered_unmatched = filtered_unmatched[filtered_unmatched['Requested_Region'] == selected_region]
    
    # Display columns of interest
    display_columns = ["Encoded ID", "Requested_Region", "unmatched_reason", 
                       "first_choice_location", "first_choice_time", 
                       "second_choice_location", "second_choice_time",
                       "host"]
    
    display_columns = [col for col in display_columns if col in filtered_unmatched.columns]
    
    # Display the filtered unmatched participants
    if not filtered_unmatched.empty:
        # Display the count of unmatched participants
        st.write(f"Showing {len(filtered_unmatched)} unmatched participants")
        
        st.dataframe(
            filtered_unmatched[display_columns],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No unmatched participants match the selected filters.")

def render_circle_details():
    """Render detailed information about each circle"""
    if 'matched_circles' not in st.session_state or st.session_state.matched_circles.empty:
        st.info("No circles have been formed yet.")
        return
    
    circles = st.session_state.matched_circles
    results = st.session_state.results
    
    # Create a dropdown to select a circle
    circle_ids = sorted(circles['circle_id'].unique().tolist())
    selected_circle = st.selectbox("Select Circle to View", circle_ids)
    
    # Get the selected circle data
    circle = circles[circles['circle_id'] == selected_circle].iloc[0]
    
    # Display circle information
    st.subheader(f"Circle: {selected_circle}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Region:**", circle['region'])
        st.write("**Subregion:**", circle['subregion'])
    
    with col2:
        st.write("**Meeting Time:**", circle['meeting_time'])
        st.write("**Member Count:**", circle['member_count'])
    
    with col3:
        st.write("**Always Hosts:**", circle['always_hosts'])
        st.write("**Sometimes Hosts:**", circle['sometimes_hosts'])
    
    # Get all members of this circle
    if 'members' in circle:
        members = circle['members']
        circle_members = results[results['Encoded ID'].isin(members)]
        
        st.write("### Circle Members")
        
        # Display member information
        if not circle_members.empty:
            display_columns = ["Encoded ID", "Status", "host", "proposed_NEW_host", "proposed_NEW_co_leader"]
            display_columns = [col for col in display_columns if col in circle_members.columns]
            
            st.dataframe(
                circle_members[display_columns],
                use_container_width=True,
                hide_index=True
            )

def render_participant_details():
    """Render detailed information about individual participants"""
    if 'results' not in st.session_state or st.session_state.results is None:
        st.info("Run the matching algorithm first to see participant details.")
        return
    
    results = st.session_state.results
    
    # Add search by Encoded ID
    encoded_id = st.text_input("Search by Encoded ID")
    
    if encoded_id:
        # Find the participant
        participant = results[results['Encoded ID'] == encoded_id]
        
        if not participant.empty:
            participant = participant.iloc[0]
            
            st.subheader("Participant Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Encoded ID:**", participant['Encoded ID'])
                st.write("**Status:**", participant['Status'])
                st.write("**Requested Region:**", participant.get('Requested_Region', 'N/A'))
            
            with col2:
                st.write("**Host Status:**", participant.get('host', 'N/A'))
                st.write("**Circle Assignment:**", participant.get('proposed_NEW_circles_id', 'N/A'))
                
                if participant.get('proposed_NEW_circles_id', 'UNMATCHED') == 'UNMATCHED':
                    st.write("**Unmatched Reason:**", participant.get('unmatched_reason', 'N/A'))
            
            st.write("### Preferences")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**1st Location:**", participant.get('first_choice_location', 'N/A'))
                st.write("**2nd Location:**", participant.get('second_choice_location', 'N/A'))
                st.write("**3rd Location:**", participant.get('third_choice_location', 'N/A'))
            
            with col2:
                st.write("**1st Time:**", participant.get('first_choice_time', 'N/A'))
                st.write("**2nd Time:**", participant.get('second_choice_time', 'N/A'))
                st.write("**3rd Time:**", participant.get('third_choice_time', 'N/A'))
            
            st.write("### Assignment Details")
            
            if participant.get('proposed_NEW_circles_id', 'UNMATCHED') != 'UNMATCHED':
                st.write("**Assigned Circle:**", participant.get('proposed_NEW_circles_id', 'N/A'))
                st.write("**Assigned Subregion:**", participant.get('proposed_NEW_Subregion', 'N/A'))
                st.write("**Assigned Time:**", participant.get('proposed_NEW_DayTime', 'N/A'))
                st.write("**Assigned as Host:**", participant.get('proposed_NEW_host', 'No'))
                st.write("**Assigned as Co-leader:**", participant.get('proposed_NEW_co_leader', 'No'))
                
                # Calculate preference match scores
                location_match = (participant.get('first_choice_location', '') == participant.get('proposed_NEW_Subregion', '')) or \
                                (participant.get('second_choice_location', '') == participant.get('proposed_NEW_Subregion', '')) or \
                                (participant.get('third_choice_location', '') == participant.get('proposed_NEW_Subregion', ''))
                
                time_match = (participant.get('first_choice_time', '') == participant.get('proposed_NEW_DayTime', '')) or \
                            (participant.get('second_choice_time', '') == participant.get('proposed_NEW_DayTime', '')) or \
                            (participant.get('third_choice_time', '') == participant.get('proposed_NEW_DayTime', ''))
                
                st.write("**Location Preference Match:**", "Yes" if location_match else "No")
                st.write("**Time Preference Match:**", "Yes" if time_match else "No")
            else:
                st.write("This participant could not be matched to a circle.")
                st.write("**Unmatched Reason:**", participant.get('unmatched_reason', 'N/A'))
        else:
            st.warning(f"No participant found with Encoded ID: {encoded_id}")
    else:
        st.info("Enter an Encoded ID to view participant details.")

def render_visualizations():
    """Render visualizations of the matching results"""
    if 'results' not in st.session_state or st.session_state.results is None:
        st.info("Run the matching algorithm first to see visualizations.")
        return
    
    results = st.session_state.results
    
    st.subheader("Matching Visualizations")
    
    # Match rate by region
    if 'Requested_Region' in results.columns:
        st.write("### Match Rate by Region")
        
        region_stats = results.groupby('Requested_Region').apply(
            lambda x: pd.Series({
                'Total': len(x),
                'Matched': sum(x['proposed_NEW_circles_id'] != "UNMATCHED"),
                'Unmatched': sum(x['proposed_NEW_circles_id'] == "UNMATCHED"),
            })
        ).reset_index()
        
        region_stats['Match_Rate'] = (region_stats['Matched'] / region_stats['Total'] * 100).round(1)
        
        # Create bar chart
        fig = px.bar(
            region_stats,
            x='Requested_Region',
            y=['Matched', 'Unmatched'],
            title='Match Rates by Region',
            labels={'value': 'Participants', 'Requested_Region': 'Region'},
            hover_data=['Match_Rate'],
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Unmatched reasons
    if 'unmatched_reason' in results.columns:
        unmatched = results[results['proposed_NEW_circles_id'] == "UNMATCHED"]
        
        if not unmatched.empty:
            st.write("### Unmatched Reasons")
            
            reason_counts = unmatched['unmatched_reason'].value_counts().reset_index()
            reason_counts.columns = ['Reason', 'Count']
            
            fig = px.pie(
                reason_counts,
                values='Count',
                names='Reason',
                title='Reasons for Unmatched Participants'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Circle size distribution
    if 'matched_circles' in st.session_state and not st.session_state.matched_circles.empty:
        circles = st.session_state.matched_circles
        
        if 'member_count' in circles.columns:
            st.write("### Circle Size Distribution")
            
            size_counts = circles['member_count'].value_counts().reset_index()
            size_counts.columns = ['Circle Size', 'Count']
            size_counts = size_counts.sort_values('Circle Size')
            
            fig = px.bar(
                size_counts,
                x='Circle Size',
                y='Count',
                title='Distribution of Circle Sizes'
            )
            
            st.plotly_chart(fig, use_container_width=True)