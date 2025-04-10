import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

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

def render_demographics_tab():
    """Render the demographics analysis tab content"""
    st.subheader("Demographics Analysis")
    
    # Check if we have results or processed data to work with
    if ('results' not in st.session_state or st.session_state.results is None) and \
       ('processed_data' not in st.session_state or st.session_state.processed_data is None):
        st.info("Run the matching algorithm first to see demographic analysis.")
        return
    
    # Get the data to work with - prefer results data if available, otherwise use processed data
    if 'results' in st.session_state and st.session_state.results is not None:
        data = st.session_state.results.copy()
    else:
        data = st.session_state.processed_data.copy()
    
    # Add region and match status filters
    col1, col2 = st.columns(2)
    
    # Filter by region
    regions = ["All"]
    # Add region options if available 
    if 'Derived_Region' in data.columns:
        regions.extend(sorted(data['Derived_Region'].dropna().unique().tolist()))
    elif 'Requested_Region' in data.columns:
        regions.extend(sorted(data['Requested_Region'].dropna().unique().tolist()))
    
    with col1:
        selected_region = st.selectbox("Filter by Region", regions)
    
    # Filter by match status if results are available
    with col2:
        match_options = ["All", "Matched", "Unmatched"]
        selected_match = st.selectbox("Filter by Match Status", match_options)
    
    # Apply filters
    filtered_data = data.copy()
    
    # Apply region filter
    if selected_region != "All":
        if 'Derived_Region' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['Derived_Region'] == selected_region]
        elif 'Requested_Region' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['Requested_Region'] == selected_region]
    
    # Apply match status filter if results are available
    if 'results' in st.session_state and selected_match != "All":
        if selected_match == "Matched" and 'proposed_NEW_circles_id' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['proposed_NEW_circles_id'] != "UNMATCHED"]
        elif selected_match == "Unmatched" and 'proposed_NEW_circles_id' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['proposed_NEW_circles_id'] == "UNMATCHED"]
    
    # Create tabs for different demographic views
    demo_tab1, demo_tab2 = st.tabs(["Class Vintage", "Other Demographics"])
    
    with demo_tab1:
        render_class_vintage_analysis(filtered_data)
    
    with demo_tab2:
        st.info("Additional demographic analyses will be added in future updates.")

def render_class_vintage_analysis(data):
    """Render the Class Vintage analysis visualizations"""
    st.subheader("Class Vintage Distribution")
    
    # Check if we have Class Vintage data
    if 'Class_Vintage' not in data.columns:
        st.warning("Class Vintage data is not available. Please ensure GSB Class data was included in the uploaded file.")
        return
    
    # Create a copy to work with
    df = data.copy()
    
    # Filter out rows with missing Class Vintage
    df = df[df['Class_Vintage'].notna()]
    
    if len(df) == 0:
        st.warning("No Class Vintage data is available after filtering.")
        return
    
    # Define the proper order for Class Vintage categories
    vintage_order = [
        "01-10 yrs", "11-20 yrs", "21-30 yrs", 
        "31-40 yrs", "41-50 yrs", "51-60 yrs", "61+ yrs"
    ]
    
    # Count by Class Vintage
    vintage_counts = df['Class_Vintage'].value_counts().reindex(vintage_order).fillna(0).astype(int)
    
    # Create a DataFrame for plotting
    vintage_df = pd.DataFrame({
        'Class Vintage': vintage_counts.index,
        'Count': vintage_counts.values
    })
    
    # Create histogram using plotly with Stanford cardinal red color
    fig = px.bar(
        vintage_df,
        x='Class Vintage',
        y='Count',
        title='Distribution of Class Vintage',
        text='Count',  # Display count values on bars
        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
    )
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': vintage_order},
        xaxis_title="Class Vintage (Years Since Graduation)",
        yaxis_title="Count of Participants"
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Create a breakdown by Status
    if 'Status' in df.columns:
        st.subheader("Class Vintage by Status")
        
        # Create a crosstab of Class Vintage vs Status
        status_vintage = pd.crosstab(
            df['Class_Vintage'], 
            df['Status'],
            rownames=['Class Vintage'],
            colnames=['Status']
        ).reindex(vintage_order)
        
        # Add a Total column
        status_vintage['Total'] = status_vintage.sum(axis=1)
        
        # Calculate percentages
        for col in status_vintage.columns:
            if col != 'Total':
                status_vintage[f'{col} %'] = (status_vintage[col] / status_vintage['Total'] * 100).round(1)
        
        # Reorder columns to group counts with percentages
        cols = []
        for status in status_vintage.columns:
            if status != 'Total' and not status.endswith(' %'):
                cols.append(status)
                cols.append(f'{status} %')
        cols.append('Total')
        
        # Show the table
        st.dataframe(status_vintage[cols], use_container_width=True)
        
        # Create a stacked bar chart
        fig = px.bar(
            status_vintage.reset_index(),
            x='Class Vintage',
            y=[col for col in status_vintage.columns if col != 'Total' and not col.endswith(' %')],
            title='Class Vintage Distribution by Status',
            barmode='stack',
            color_discrete_sequence=['#8C1515', '#2E2D29']  # Stanford colors
        )
        
        # Customize layout
        fig.update_layout(
            xaxis={'categoryorder': 'array', 'categoryarray': vintage_order},
            xaxis_title="Class Vintage (Years Since Graduation)",
            yaxis_title="Count of Participants",
            legend_title="Status"
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True)
    
    # Create a breakdown by Match Status if available
    if 'proposed_NEW_circles_id' in df.columns:
        st.subheader("Class Vintage by Match Status")
        
        # Create a match status column
        df['Match Status'] = df['proposed_NEW_circles_id'].apply(
            lambda x: "Unmatched" if x == "UNMATCHED" else "Matched"
        )
        
        # Create a crosstab of Class Vintage vs Match Status
        match_vintage = pd.crosstab(
            df['Class_Vintage'], 
            df['Match Status'],
            rownames=['Class Vintage'],
            colnames=['Match Status']
        ).reindex(vintage_order)
        
        # Add a Total column
        match_vintage['Total'] = match_vintage.sum(axis=1)
        
        # Calculate match rate percentage
        if 'Matched' in match_vintage.columns and 'Unmatched' in match_vintage.columns:
            match_vintage['Match Rate %'] = (match_vintage['Matched'] / match_vintage['Total'] * 100).round(1)
        
        # Show the table
        st.dataframe(match_vintage, use_container_width=True)
        
        # Create a stacked bar chart
        fig = px.bar(
            match_vintage.reset_index(),
            x='Class Vintage',
            y=['Matched', 'Unmatched'],
            title='Class Vintage Distribution by Match Status',
            barmode='stack',
            color_discrete_sequence=['#008566', '#8C1515']  # Green for matched, red for unmatched
        )
        
        # Customize layout
        fig.update_layout(
            xaxis={'categoryorder': 'array', 'categoryarray': vintage_order},
            xaxis_title="Class Vintage (Years Since Graduation)",
            yaxis_title="Count of Participants",
            legend_title="Match Status"
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a line chart for match rate
        if 'Match Rate %' in match_vintage.columns:
            # Create a line chart for match rate
            fig = px.line(
                match_vintage.reset_index(),
                x='Class Vintage',
                y='Match Rate %',
                title='Match Rate by Class Vintage',
                markers=True,
                color_discrete_sequence=['#00505C']  # Stanford blue
            )
            
            # Customize layout
            fig.update_layout(
                xaxis={'categoryorder': 'array', 'categoryarray': vintage_order},
                xaxis_title="Class Vintage (Years Since Graduation)",
                yaxis_title="Match Rate (%)",
                yaxis=dict(range=[0, 100])  # Set y-axis range from 0 to 100%
            )
            
            # Add a horizontal reference line at 80% (target match rate)
            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                xref="paper",
                y0=80,
                y1=80,
                line=dict(color="#8C1515", width=2, dash="dash")
            )
            
            # Add annotation for the target line
            fig.add_annotation(
                x=0.5,
                y=82,
                xref="paper",
                text="Target Match Rate (80%)",
                showarrow=False,
                font=dict(color="#8C1515")
            )
            
            # Show the plot
            st.plotly_chart(fig, use_container_width=True)

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

        # Display result counts by status if available
        if 'results' in st.session_state and 'Status' in st.session_state.results.columns:
            st.write("### Results Status Counts")
            results_status_counts = st.session_state.results['Status'].value_counts().reset_index()
            results_status_counts.columns = ['Status', 'Count']
            st.dataframe(results_status_counts, use_container_width=True)

            # Count matched versus unmatched for each status
            st.write("### Status Match Rates")
            match_by_status = pd.crosstab(
                st.session_state.results['Status'], 
                st.session_state.results['proposed_NEW_circles_id'] != 'UNMATCHED',
                rownames=['Status'], 
                colnames=['Matched']
            ).reset_index()

            # Add percentage columns
            total_by_status = match_by_status.sum(axis=1).values
            match_by_status['Total'] = total_by_status
            match_by_status['Match %'] = (match_by_status[True] / match_by_status['Total'] * 100).round(1)

            # Rename columns for clarity
            match_by_status.columns = ['Status', 'Unmatched', 'Matched', 'Total', 'Match %']

            st.dataframe(match_by_status, use_container_width=True)

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
                st.write("### Processed Data Status Counts (Binary)")
                processed_status_counts = st.session_state.processed_data['Status'].value_counts().reset_index()
                processed_status_counts.columns = ['Status', 'Count']
                st.dataframe(processed_status_counts, use_container_width=True)

            # Display Raw_Status counts if available
            if 'Raw_Status' in st.session_state.processed_data.columns:
                st.write("### Processed Data Detailed Status Counts")
                raw_status_counts = st.session_state.processed_data['Raw_Status'].value_counts().reset_index()
                raw_status_counts.columns = ['Raw Status', 'Count']
                st.dataframe(raw_status_counts, use_container_width=True)

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

    # Create a copy of the circles dataframe to avoid modifying the original
    circles = st.session_state.matched_circles.copy()

    # Special handling for the members column which could be causing display issues
    if 'members' in circles.columns:
        # Convert members to string if it's a list or other object
        try:
            circles['members'] = circles['members'].apply(lambda x: str(x) if not isinstance(x, str) else x)
        except Exception as e:
            st.error(f"Error converting members column: {str(e)}")

    # Preprocess all numeric columns to ensure they're numeric
    # This prevents the "not supported between instances of 'float' and 'str'" error
    for col in ['member_count', 'new_members', 'always_hosts', 'sometimes_hosts']:
        if col in circles.columns:
            try:
                circles[col] = pd.to_numeric(circles[col], errors='coerce').fillna(0).astype(int)
            except Exception as e:
                st.error(f"Error converting {col} to numeric: {str(e)}")

    # Make sure new_members column exists (for new circles, this equals member_count)
    if 'new_members' not in circles.columns and 'member_count' in circles.columns:
        circles['new_members'] = circles['member_count']

    # Add filter options
    col1, col2 = st.columns(2)

    # Initialize default values
    selected_region = 'All'
    selected_subregion = 'All'

    # Create safe string versions of region and subregion for filters
    # This prevents the '<' comparison error between float and string
    try:
        # Convert all columns to string first for safer operations
        safe_circles = circles.copy()

        # Convert object columns to strings for filtering
        for col in ['region', 'subregion']:
            if col in safe_circles.columns:
                # Replace any None/NaN values with empty string before conversion
                safe_circles[col] = safe_circles[col].fillna('')
                safe_circles[col] = safe_circles[col].astype(str)

        # Now create the filters using the safe string columns
        with col1:
            if 'region' in safe_circles.columns:
                regions = ['All'] + sorted(safe_circles['region'].unique().tolist())
                selected_region = st.selectbox("Filter by Region", regions)

        with col2:
            if 'subregion' in safe_circles.columns:
                subregions = ['All'] + sorted(safe_circles['subregion'].unique().tolist())
                selected_subregion = st.selectbox("Filter by Subregion", subregions)

        # Apply filters using the safe string columns
        filtered_circles = safe_circles.copy()

        if selected_region != 'All':
            filtered_circles = filtered_circles[filtered_circles['region'] == selected_region]

        if selected_subregion != 'All':
            filtered_circles = filtered_circles[filtered_circles['subregion'] == selected_subregion]

    except Exception as e:
        st.error(f"Error during filtering setup: {str(e)}")
        # Create a backup filtered dataset
        filtered_circles = circles.copy()
        st.warning("Advanced filtering disabled due to error - showing all circles")

    # Display the filtered circles
    if not filtered_circles.empty:
        # Display the count of filtered circles
        st.write(f"Showing {len(filtered_circles)} circles")

        # Prepare a safer version of the dataframe for display
        try:
            # Create a display-only dataframe with simplified columns
            display_df = pd.DataFrame()

            # Define display-ready columns - explicitly select and convert them
            column_mapping = {
                "circle_id": "Circle ID",
                "region": "Region",
                "subregion": "Subregion",
                "meeting_time": "Meeting Time",
                "member_count": "Member Count",
                "new_members": "New Members",
                "max_additions": "Max Additions",
                "always_hosts": "Always Hosts",
                "sometimes_hosts": "Sometimes Hosts"
            }

            # Build display dataframe with clean columns
            for orig_col, display_col in column_mapping.items():
                if orig_col in filtered_circles.columns:
                    # For numeric columns, ensure they're properly converted
                    if orig_col in ["member_count", "new_members", "always_hosts", "sometimes_hosts"]:
                        display_df[display_col] = pd.to_numeric(filtered_circles[orig_col], 
                                                               errors='coerce').fillna(0).astype(int)
                    else:
                        # For string columns, ensure they're strings
                        display_df[display_col] = filtered_circles[orig_col].astype(str)

            # Convert Max Additions to integer format, handling decimals
            if 'Max Additions' in display_df.columns:
                # First convert to float to handle any decimal values
                display_df['Max Additions'] = pd.to_numeric(display_df['Max Additions'], errors='coerce')
                # Floor decimal values and convert to int, keeping NaN values
                display_df['Max Additions'] = display_df['Max Additions'].apply(
                    lambda x: int(x) if pd.notnull(x) else x
                )

            # Display the clean dataframe with row numbers
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=False
            )
        except Exception as e:
            st.error(f"Error preparing display table: {str(e)}")

            # Ultra-fallback - just show the raw data with no column reordering
            st.write("Using emergency fallback display method:")
            try:
                # Just show the raw data as string representations
                raw_display = pd.DataFrame()
                for col in filtered_circles.columns:
                    raw_display[col] = filtered_circles[col].astype(str)

                st.dataframe(
                    raw_display,
                    use_container_width=True,
                    hide_index=True
                )
            except Exception as e2:
                st.error(f"Even fallback display failed: {str(e2)}")
                st.write("Data cannot be displayed in table format. Please check the downloadable CSV for complete results.")
    else:
        st.info("No circles match the selected filters.")

def render_unmatched_table():
    """Render the unmatched participants table"""
    if 'unmatched_participants' not in st.session_state or st.session_state.unmatched_participants.empty:
        st.info("All participants have been matched.")
        return

    # Create a deep copy to avoid modifying the original
    unmatched = st.session_state.unmatched_participants.copy()

    # Pre-process all columns to ensure consistent types
    for col in unmatched.columns:
        try:
            # For every column, normalize to string if it's an object type (might contain mixed types)
            if unmatched[col].dtype == 'object':
                unmatched[col] = unmatched[col].fillna('').astype(str)
        except Exception as e:
            st.error(f"Error pre-processing column {col}: {str(e)}")

    # Add filter options
    col1, col2 = st.columns(2)

    # Initialize default values
    selected_reason = 'All'
    selected_region = 'All'

    # Create filters only after converting all columns to consistent types
    try:
        with col1:
            if 'unmatched_reason' in unmatched.columns:
                # Use the string-converted column for consistent filters
                reasons = ['All'] + sorted(unmatched['unmatched_reason'].unique().tolist())
                selected_reason = st.selectbox("Filter by Reason", reasons)

        with col2:
            if 'Requested_Region' in unmatched.columns:
                # Use the string-converted column for consistent filters
                regions = ['All'] + sorted(unmatched['Requested_Region'].unique().tolist())
                selected_region = st.selectbox("Filter by Requested Region", regions, key="unmatched_region")

        # Apply filters to the pre-processed dataframe
        filtered_unmatched = unmatched.copy()

        # Filter by reason if selected
        if 'unmatched_reason' in unmatched.columns and selected_reason != 'All':
            filtered_unmatched = filtered_unmatched[filtered_unmatched['unmatched_reason'] == selected_reason]

        # Filter by region if selected
        if 'Requested_Region' in unmatched.columns and selected_region != 'All':
            filtered_unmatched = filtered_unmatched[filtered_unmatched['Requested_Region'] == selected_region]

    except Exception as e:
        st.error(f"Error setting up filters: {str(e)}")
        # Fall back to showing all participants without filtering
        filtered_unmatched = unmatched.copy()
        st.warning("Filtering disabled due to error - showing all unmatched participants")

    # Display the filtered unmatched participants
    if not filtered_unmatched.empty:
        # Display the count of unmatched participants
        st.write(f"Showing {len(filtered_unmatched)} unmatched participants")

        # Prepare a safer version of the dataframe for display
        try:
            # Create a completely new display dataframe with carefully controlled types
            display_df = pd.DataFrame()

            # Define display-ready columns with nice names
            column_mapping = {
                "Encoded ID": "Participant ID",
                "Requested_Region": "Region",
                "unmatched_reason": "Reason Unmatched",
                "first_choice_location": "1st Choice Location",
                "first_choice_time": "1st Choice Time",
                "second_choice_location": "2nd Choice Location",
                "second_choice_time": "2nd Choice Time",
                "host": "Host Status"
            }

            # Build display dataframe with only the columns we need
            for orig_col, display_col in column_mapping.items():
                if orig_col in filtered_unmatched.columns:
                    # For all columns, ensure they're strings for consistent display
                    display_df[display_col] = filtered_unmatched[orig_col].fillna('').astype(str)

            # Display the clean dataframe with row numbers
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=False
            )
        except Exception as e:
            st.error(f"Error preparing display table: {str(e)}")

            # Ultra-fallback - just list IDs and reasons in text format
            st.write("Using emergency fallback display method:")
            try:
                # Display 1 participant per line with basic info
                for i in range(min(100, len(filtered_unmatched))): # Limit to 100 to avoid cluttering
                    row = filtered_unmatched.iloc[i]
                    id_val = row.get('Encoded ID', 'Unknown ID')
                    reason = row.get('unmatched_reason', 'Unknown reason')
                    st.write(f"• Participant: {id_val} - Reason: {reason}")

                if len(filtered_unmatched) > 100:
                    st.write(f"... and {len(filtered_unmatched) - 100} more participants")
            except Exception as e2:
                st.error(f"Even fallback display failed: {str(e2)}")
                st.write("Data cannot be displayed. Please check the downloadable CSV for complete results.")
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

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write("**Region:**", circle['region'])
        st.write("**Subregion:**", circle['subregion'])

    with col2:
        st.write("**Meeting Time:**", circle['meeting_time'])
        st.write("**Member Count:**", circle['member_count'])

    with col3:
        st.write("**Always Hosts:**", circle['always_hosts'])
        st.write("**Sometimes Hosts:**", circle['sometimes_hosts'])

    # Add information about max additions if available
    with col4:
        st.write("**New Members:**", circle.get('new_members', 0))

        # Display max_additions if present in the dataframe
        if 'max_additions' in circle:
            max_adds = circle['max_additions']
            if max_adds == 0:
                st.write("**Max Additions:** None")
                st.write("*Co-leader preference: no new members allowed*")
            else:
                st.write(f"**Max Additions:** {max_adds}")
                if circle.get('new_members', 0) > 0:
                    remaining = max(0, max_adds - circle.get('new_members', 0))
                    st.write(f"*Used {circle.get('new_members', 0)} of {max_adds}, {remaining} remaining*")
        else:
            st.write("**Max Additions:** No limit specified")

    # Get all members of this circle
    if 'members' in circle:
        try:
            members = circle['members']

            # Handle different types of members (it might be a string representation of a list)
            if isinstance(members, str):
                try:
                    # Handle string representation of a list 
                    if members.startswith('[') and members.endswith(']'):
                        import ast
                        members = ast.literal_eval(members)  # Convert string representation to actual list
                    else:
                        # Handle comma-separated list
                        members = [m.strip() for m in members.split(',')]
                except Exception as e:
                    st.error(f"Error parsing members string: {str(e)}")
                    members = []

            # Make sure members is iterable
            if not hasattr(members, '__iter__') or isinstance(members, str):
                members = [members]  # Convert to a list if it's a scalar

            try:
                # Find members in results
                circle_members = results[results['Encoded ID'].isin(members)]
                st.write(f"Found {len(circle_members)} of {len(members)} members in results")
            except Exception as e:
                st.error(f"Error filtering members: {str(e)}")
                circle_members = pd.DataFrame()
        except Exception as e:
            st.error(f"Error processing circle members: {str(e)}")
            circle_members = pd.DataFrame()

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
                # Display both binary status and detailed status if available
                st.write("**Status:**", participant['Status'])
                if 'Raw_Status' in participant:
                    st.write("**Detailed Status:**", participant['Raw_Status'])
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

        try:
            # Create a safer version of the dataframe for visualization
            viz_results = results.copy()

            # Make sure the proposed_NEW_circles_id column is string for safe comparison
            if 'proposed_NEW_circles_id' in viz_results.columns:
                viz_results['proposed_NEW_circles_id'] = viz_results['proposed_NEW_circles_id'].fillna("UNMATCHED").astype(str)

            # Create stats using the safe dataframe
            region_stats = viz_results.groupby('Requested_Region').apply(
                lambda x: pd.Series({
                    'Total': len(x),
                    'Matched': sum(x['proposed_NEW_circles_id'] != "UNMATCHED"),
                    'Unmatched': sum(x['proposed_NEW_circles_id'] == "UNMATCHED"),
                })
            ).reset_index()
        except Exception as e:
            st.error(f"Error generating region statistics: {str(e)}")
            # Create a minimal fallback
            region_stats = pd.DataFrame(columns=['Requested_Region', 'Total', 'Matched', 'Unmatched'])
            st.warning("Could not generate region statistics due to data type issues.")

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
        try:
            # Create a safer version for visualization
            viz_results = results.copy()

            # Make sure proposed_NEW_circles_id is a string for safe comparison
            if 'proposed_NEW_circles_id' in viz_results.columns:
                viz_results['proposed_NEW_circles_id'] = viz_results['proposed_NEW_circles_id'].fillna("UNMATCHED").astype(str)

            # Filter to only unmatched entries
            unmatched = viz_results[viz_results['proposed_NEW_circles_id'] == "UNMATCHED"]

            if not unmatched.empty:
                st.write("### Unmatched Reasons")

                # Make sure unmatched_reason is string for safe aggregation
                unmatched['unmatched_reason'] = unmatched['unmatched_reason'].fillna("Unknown").astype(str)

                reason_counts = unmatched['unmatched_reason'].value_counts().reset_index()
                reason_counts.columns = ['Reason', 'Count']

                fig = px.pie(
                    reason_counts,
                    values='Count',
                    names='Reason',
                    title='Reasons for Unmatched Participants'
                )

                # Only show chart inside the try block if we successfully created it
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating unmatched reason visualization: {str(e)}")
            st.warning("Could not generate unmatched reasons visualization due to data type issues.")

    # Circle size distribution
    if 'matched_circles' in st.session_state and not st.session_state.matched_circles.empty:
        try:
            circles = st.session_state.matched_circles.copy()

            if 'member_count' in circles.columns:
                st.write("### Circle Size Distribution")

                # Ensure member_count is numeric
                circles['member_count'] = pd.to_numeric(circles['member_count'], errors='coerce').fillna(0).astype(int)

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
        except Exception as e:
            st.error(f"Error generating circle size distribution: {str(e)}")
            st.warning("Could not generate circle size distribution due to data type issues.")