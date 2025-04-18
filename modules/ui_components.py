import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

# Function to render different tabs in the UI
def render_match_tab():
    """Render the main matching tab content"""
    # Organizing into columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Participant Data")
        
        # File uploader for participant data
        uploaded_file = st.file_uploader("Upload a CSV file with participant data", type=["csv"])
        
        if uploaded_file is not None:
            # Add a 'Process Data' button for explicit processing
            process_button = st.button("Process Data")
            
            if process_button or 'processed_data' not in st.session_state:
                # If button is clicked or no processed data exists, process the data
                from app import process_uploaded_file
                process_uploaded_file(uploaded_file)
                st.success("Data processed successfully!")
                st.session_state.button_clicked = True
        else:
            # Clear session state if no file is uploaded
            if 'processed_data' in st.session_state:
                st.warning("File removed. Upload a new file to continue.")
                # Keep a backup of the current data for inspection
                if 'backup_data' not in st.session_state:
                    st.session_state.backup_data = st.session_state.processed_data.copy() if st.session_state.processed_data is not None else None
                
                # Clear current working data
                st.session_state.processed_data = None
                st.session_state.button_clicked = False
    
    with col2:
        st.subheader("Matching Algorithm")
        
        # Optimization button with options - only if data has been processed
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            from app import run_optimization
            
            # Create expandable section for advanced options
            with st.expander("Advanced Options"):
                st.selectbox("Circle Size Preference", 
                            options=["Balanced", "Larger Circles", "Smaller Circles"],
                            index=0,
                            key="circle_size_pref",
                            help="'Balanced' tries to get everyone in a circle. 'Larger Circles' favors having fewer, larger circles. 'Smaller Circles' favors more intimate groups.")
                
                # Weight for location preferences
                st.slider("Location Match Weight", 
                        min_value=1.0, 
                        max_value=10.0, 
                        value=5.0, 
                        step=0.5,
                        key="location_weight",
                        help="Higher values prioritize participants' location preferences over other factors.")
                
                # Weight for time preferences
                st.slider("Time Match Weight", 
                        min_value=1.0, 
                        max_value=10.0, 
                        value=5.0, 
                        step=0.5,
                        key="time_weight",
                        help="Higher values prioritize participants' time preferences over other factors.")
                
                # Weight for full circles (only relevant for Balanced mode)
                st.slider("Complete Circle Weight", 
                        min_value=1.0, 
                        max_value=10.0, 
                        value=3.0, 
                        step=0.5,
                        key="circle_weight",
                        help="Higher values prioritize having complete circles over partial matches.")
                
                # Tradeoff between honoring existing circles and optimizing for preferences
                st.slider("Existing Circle Preservation", 
                        min_value=0.0, 
                        max_value=10.0, 
                        value=7.0, 
                        step=0.5,
                        key="existing_circle_weight",
                        help="Higher values favor keeping members in their current circles rather than moving them.")
                
                # Maximum iterations for the solver
                st.number_input("Max Solver Iterations", 
                                min_value=1000, 
                                max_value=1000000, 
                                value=100000, 
                                step=10000,
                                key="max_iterations",
                                help="Maximum number of iterations for the optimization solver. Higher values may improve solutions but take longer.")
                
                # Tolerance for solver convergence
                st.number_input("Solver Tolerance", 
                                min_value=0.0001, 
                                max_value=0.1, 
                                value=0.001, 
                                format="%f",
                                step=0.001,
                                key="solver_tolerance",
                                help="Numerical tolerance for solver convergence. Lower values give more precise results but may take longer.")
            
            # Run Optimization button
            if st.button("Run Matching Algorithm"):
                with st.spinner("Running optimization - this may take a moment..."):
                    run_optimization()
                
                # Show results
                if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
                    st.success("Matching completed successfully!")
                    st.session_state.active_tab = 1  # Switch to Results tab
                    
                    # Use this to force a rerun to switch tabs
                    # Replaced st.experimental_rerun() with st.rerun()
                    st.rerun()
        else:
            st.info("Upload and process data to enable matching.")


def render_details_tab():
    """Render the details tab content"""
    if ('matched_circles' not in st.session_state or 
        st.session_state.matched_circles is None or 
        (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty)):
        st.warning("No matching results available. Please run the matching algorithm first.")
        return
    
    # Create tabs for different views
    detail_tab1, detail_tab2, detail_tab3 = st.tabs(["Overview", "Circles", "Participants"])
    
    with detail_tab1:
        render_results_overview()
    
    with detail_tab2:
        render_circle_details()
    
    with detail_tab3:
        render_participant_details()


def render_demographics_tab():
    """Render the demographics analysis tab content"""
    st.subheader("Demographics Analysis")
    
    # Add filter by region and match status at the top
    col1, col2 = st.columns(2)
    
    with col1:
        # Get available regions
        available_regions = []
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            if 'Current_Region' in df.columns:
                available_regions = sorted(df['Current_Region'].dropna().unique().tolist())
            elif 'Region' in df.columns:
                available_regions = sorted(df['Region'].dropna().unique().tolist())
            
        # Add "All Regions" option
        available_regions = ["All Regions"] + available_regions
        
        # Region filter
        selected_region = st.selectbox("Filter by Region", options=available_regions, index=0)
    
    with col2:
        # Match status filter
        match_options = ["All Participants", "Matched", "Unmatched"]
        selected_match = st.selectbox("Filter by Match Status", options=match_options, index=0)
    
    # Get the filtered data based on selections
    filtered_data = None
    if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
        filtered_data = st.session_state.processed_data.copy()
        
        # Apply region filter if not "All Regions"
        if selected_region != "All Regions":
            if 'Current_Region' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Current_Region'] == selected_region]
            elif 'Region' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Region'] == selected_region]
        
        # Apply match status filter
        if selected_match == "Matched" and 'proposed_NEW_circles_id' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['proposed_NEW_circles_id'] != "UNMATCHED"]
        elif selected_match == "Unmatched" and 'proposed_NEW_circles_id' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['proposed_NEW_circles_id'] == "UNMATCHED"]
    
    # Create tabs for different demographic views
    demo_tab1, demo_tab2, demo_tab3, demo_tab4, demo_tab5 = st.tabs(["Class Vintage", "Employment", "Industry", "Racial Identity", "Other Demographics"])
    
    with demo_tab1:
        render_class_vintage_analysis(filtered_data)
    
    with demo_tab2:
        render_employment_analysis(filtered_data)
    
    with demo_tab3:
        render_industry_analysis(filtered_data)
    
    with demo_tab4:
        render_racial_identity_analysis(filtered_data)
    
    with demo_tab5:
        st.info("Additional demographic analyses will be added in future updates.")

def render_vintage_diversity_histogram():
    """
    Create a histogram showing the distribution of circles based on 
    the number of different class vintages they contain
    """
    # First check if we have the necessary data
    if 'matched_circles' not in st.session_state or st.session_state.matched_circles is None:
        st.warning("No matched circles data available to analyze vintage diversity.")
        return
        
    # Check if the matched_circles dataframe is empty
    if hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty:
        st.warning("No matched circles data available (empty dataframe).")
        return
        
    if 'results' not in st.session_state or st.session_state.results is None:
        st.warning("No results data available to analyze vintage diversity.")
        return
    
    # Get the circle data
    circles_df = st.session_state.matched_circles.copy()
    results_df = st.session_state.results.copy()
    
    # Filter out circles with no members
    if 'member_count' not in circles_df.columns:
        st.warning("Circles data does not have member_count column")
        return
    
    circles_df = circles_df[circles_df['member_count'] > 0]
    
    if len(circles_df) == 0:
        st.warning("No circles with members available for analysis.")
        return
    
    # Dictionary to track unique vintages per circle
    circle_vintage_counts = {}
    circle_vintage_diversity_scores = {}
    
    # Get vintage data for each member of each circle
    for _, circle_row in circles_df.iterrows():
        circle_id = circle_row['circle_id']
        
        # Initialize empty set to track unique vintages
        unique_vintages = set()
        
        # Get the list of members for this circle
        if 'members' in circle_row and circle_row['members']:
            # For list representation
            if isinstance(circle_row['members'], list):
                member_ids = circle_row['members']
            # For string representation - convert to list
            elif isinstance(circle_row['members'], str):
                try:
                    if circle_row['members'].startswith('['):
                        member_ids = eval(circle_row['members'])
                    else:
                        member_ids = [circle_row['members']]
                except Exception as e:
                    # Skip if parsing fails
                    continue
            else:
                # Skip if members data is not in expected format
                continue
                
            # For each member, look up their vintage in results_df
            for member_id in member_ids:
                member_data = results_df[results_df['Encoded ID'] == member_id]
                
                if not member_data.empty and 'Class_Vintage' in member_data.columns:
                    vintage = member_data['Class_Vintage'].iloc[0]
                    if pd.notna(vintage):
                        unique_vintages.add(vintage)
        
        # Store the count of unique vintages for this circle
        if unique_vintages:  # Only include if there's at least one valid vintage
            count = len(unique_vintages)
            circle_vintage_counts[circle_id] = count
            # The diversity score is the number of unique vintages
            circle_vintage_diversity_scores[circle_id] = count
    
    # Create histogram data from the vintage counts
    if not circle_vintage_counts:
        st.warning("No vintage data available for circles.")
        return
        
    # Count circles by number of unique vintages
    diversity_counts = pd.Series(circle_vintage_counts).value_counts().sort_index()
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Number of Vintages': diversity_counts.index,
        'Number of Circles': diversity_counts.values
    })
    
    st.subheader("Vintage Diversity Within Circles")
    
    # Create histogram using plotly with Stanford cardinal red color
    fig = px.bar(
        plot_df,
        x='Number of Vintages',
        y='Number of Circles',
        title='Distribution of Circles by Number of Class Vintages',
        text='Number of Circles',  # Display count values on bars
        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
    )
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis=dict(
            title="Number of Different Class Vintages",
            tickmode='linear',
            dtick=1  # Force integer labels
        ),
        yaxis_title="Number of Circles"
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True, key="plot_331")
    
    # Show a table with the data
    st.caption("Data table:")
    st.dataframe(plot_df, hide_index=True)
    
    # Display the Vintage diversity score
    st.subheader("Vintage Diversity Score")
    st.write("""
    For each circle, the vintage diversity score is calculated as follows:
    - 1 point: All members in the same class vintage
    - 2 points: Members from two different class vintages
    - 3 points: Members from three different class vintages
    - And so on, with more points for more diverse circles
    """)
    
    # Calculate average and total diversity scores
    total_diversity_score = sum(circle_vintage_diversity_scores.values()) if circle_vintage_diversity_scores else 0
    avg_diversity_score = total_diversity_score / len(circle_vintage_diversity_scores) if circle_vintage_diversity_scores else 0
    
    # Create two columns for the metrics
    col1, col2 = st.columns(2)
    
    # Show average and total scores
    with col1:
        st.metric("Average Vintage Diversity Score", f"{avg_diversity_score:.2f}")
    with col2:
        st.metric("Total Vintage Diversity Score", f"{total_diversity_score}")
    
    # Add a brief explanation
    total_circles = sum(diversity_counts.values)
    diverse_circles = sum(diversity_counts[diversity_counts.index > 1].values)
    diverse_pct = (diverse_circles / total_circles * 100) if total_circles > 0 else 0
    
    st.write(f"Out of {total_circles} total circles, {diverse_circles} ({diverse_pct:.1f}%) contain members from multiple class vintages.")


def render_employment_diversity_histogram():
    """
    Create a histogram showing the distribution of circles based on 
    the number of different employment categories they contain
    """
    # First check if we have the necessary data
    if 'matched_circles' not in st.session_state or st.session_state.matched_circles is None:
        st.warning("No matched circles data available to analyze employment diversity.")
        return
        
    # Check if the matched_circles dataframe is empty
    if hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty:
        st.warning("No matched circles data available (empty dataframe).")
        return
        
    if 'results' not in st.session_state or st.session_state.results is None:
        st.warning("No results data available to analyze employment diversity.")
        return
    
    # Get the circle data
    circles_df = st.session_state.matched_circles.copy()
    results_df = st.session_state.results.copy()
    
    # Filter out circles with no members
    if 'member_count' not in circles_df.columns:
        st.warning("Circles data does not have member_count column")
        return
    
    circles_df = circles_df[circles_df['member_count'] > 0]
    
    if len(circles_df) == 0:
        st.warning("No circles with members available for analysis.")
        return
    
    # Dictionary to track unique employment categories per circle
    circle_employment_counts = {}
    circle_employment_diversity_scores = {}
    
    # Get employment data for each member of each circle
    for _, circle_row in circles_df.iterrows():
        circle_id = circle_row['circle_id']
        
        # Initialize empty set to track unique employment categories
        unique_employment_categories = set()
        
        # Get the list of members for this circle
        if 'members' in circle_row and circle_row['members']:
            # For list representation
            if isinstance(circle_row['members'], list):
                member_ids = circle_row['members']
            # For string representation - convert to list
            elif isinstance(circle_row['members'], str):
                try:
                    if circle_row['members'].startswith('['):
                        member_ids = eval(circle_row['members'])
                    else:
                        member_ids = [circle_row['members']]
                except Exception as e:
                    # Skip if parsing fails
                    continue
            else:
                # Skip if members data is not in expected format
                continue
                
            # For each member, look up their employment category in results_df
            for member_id in member_ids:
                member_data = results_df[results_df['Encoded ID'] == member_id]
                
                if not member_data.empty and 'Employment_Category' in member_data.columns:
                    employment_category = member_data['Employment_Category'].iloc[0]
                    if pd.notna(employment_category):
                        unique_employment_categories.add(employment_category)
        
        # Store the count of unique employment categories for this circle
        if unique_employment_categories:  # Only include if there's at least one valid category
            count = len(unique_employment_categories)
            circle_employment_counts[circle_id] = count
            # Calculate diversity score: 1 point if everyone is in the same category,
            # 2 points if two categories, 3 points if three categories
            circle_employment_diversity_scores[circle_id] = count
    
    # Create histogram data from the employment counts
    if not circle_employment_counts:
        st.warning("No employment data available for circles.")
        return
        
    # Count circles by number of unique employment categories
    diversity_counts = pd.Series(circle_employment_counts).value_counts().sort_index()
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Number of Employment Categories': diversity_counts.index,
        'Number of Circles': diversity_counts.values
    })
    
    st.subheader("Employment Diversity Within Circles")
    
    # Create histogram using plotly with Stanford cardinal red color
    fig = px.bar(
        plot_df,
        x='Number of Employment Categories',
        y='Number of Circles',
        title='Distribution of Circles by Number of Employment Categories',
        text='Number of Circles',  # Display count values on bars
        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
    )
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis=dict(
            title="Number of Different Employment Categories",
            tickmode='linear',
            dtick=1,  # Force integer labels
            range=[0.5, 3.5]  # Since we have 3 categories max
        ),
        yaxis_title="Number of Circles"
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True, key="plot_488")
    
    # Show a table with the data
    st.caption("Data table:")
    st.dataframe(plot_df, hide_index=True)
    
    # Display the Employment diversity score
    st.subheader("Employment Diversity Score")
    st.write("""
    For each circle, the employment diversity score is calculated as follows:
    - 1 point: All members in the same employment category
    - 2 points: Members from two different employment categories
    - 3 points: Members from all three employment categories
    """)
    
    # Calculate average and total diversity scores
    total_diversity_score = sum(circle_employment_diversity_scores.values()) if circle_employment_diversity_scores else 0
    avg_diversity_score = total_diversity_score / len(circle_employment_diversity_scores) if circle_employment_diversity_scores else 0
    
    # Create two columns for the metrics
    col1, col2 = st.columns(2)
    
    # Show average and total scores
    with col1:
        st.metric("Average Employment Diversity Score", f"{avg_diversity_score:.2f}")
    with col2:
        st.metric("Total Employment Diversity Score", f"{total_diversity_score}")
    
    # Add a brief explanation
    total_circles = sum(diversity_counts.values)
    diverse_circles = sum(diversity_counts[diversity_counts.index > 1].values)
    diverse_pct = (diverse_circles / total_circles * 100) if total_circles > 0 else 0
    
    st.write(f"Out of {total_circles} total circles, {diverse_circles} ({diverse_pct:.1f}%) contain members from multiple employment categories.")

def render_industry_diversity_histogram():
    """
    Create a histogram showing the distribution of circles based on 
    the number of different industry categories they contain
    """
    # First check if we have the necessary data
    if 'matched_circles' not in st.session_state or st.session_state.matched_circles is None:
        st.warning("No matched circles data available to analyze industry diversity.")
        return
        
    # Check if the matched_circles dataframe is empty
    if hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty:
        st.warning("No matched circles data available (empty dataframe).")
        return
        
    if 'results' not in st.session_state or st.session_state.results is None:
        st.warning("No results data available to analyze industry diversity.")
        return
    
    # Get the circle data
    circles_df = st.session_state.matched_circles.copy()
    results_df = st.session_state.results.copy()
    
    # Filter out circles with no members
    if 'member_count' not in circles_df.columns:
        st.warning("Circles data does not have member_count column")
        return
    
    circles_df = circles_df[circles_df['member_count'] > 0]
    
    if len(circles_df) == 0:
        st.warning("No circles with members available for analysis.")
        return
    
    # Dictionary to track unique industry categories per circle
    circle_industry_counts = {}
    circle_industry_diversity_scores = {}
    
    # Get industry data for each member of each circle
    for _, circle_row in circles_df.iterrows():
        circle_id = circle_row['circle_id']
        
        # Initialize empty set to track unique industry categories
        unique_industry_categories = set()
        
        # Get the list of members for this circle
        if 'members' in circle_row and circle_row['members']:
            # For list representation
            if isinstance(circle_row['members'], list):
                member_ids = circle_row['members']
            # For string representation - convert to list
            elif isinstance(circle_row['members'], str):
                try:
                    if circle_row['members'].startswith('['):
                        member_ids = eval(circle_row['members'])
                    else:
                        member_ids = [circle_row['members']]
                except Exception as e:
                    # Skip if parsing fails
                    continue
            else:
                # Skip if members data is not in expected format
                continue
                
            # For each member, look up their industry category in results_df
            for member_id in member_ids:
                member_data = results_df[results_df['Encoded ID'] == member_id]
                
                if not member_data.empty and 'Industry_Category' in member_data.columns:
                    industry_category = member_data['Industry_Category'].iloc[0]
                    if pd.notna(industry_category):
                        unique_industry_categories.add(industry_category)
        
        # Store the count of unique industry categories for this circle
        if unique_industry_categories:  # Only include if there's at least one valid category
            count = len(unique_industry_categories)
            circle_industry_counts[circle_id] = count
            # Calculate diversity score: 1 point if everyone is in the same category,
            # 2 points if two categories, etc.
            circle_industry_diversity_scores[circle_id] = count
    
    # Create histogram data from the industry counts
    if not circle_industry_counts:
        st.warning("No industry data available for circles.")
        return
        
    # Count circles by number of unique industry categories
    diversity_counts = pd.Series(circle_industry_counts).value_counts().sort_index()
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Number of Industry Categories': diversity_counts.index,
        'Number of Circles': diversity_counts.values
    })
    
    st.subheader("Industry Diversity Within Circles")
    
    # Create histogram using plotly with Stanford cardinal red color
    fig = px.bar(
        plot_df,
        x='Number of Industry Categories',
        y='Number of Circles',
        title='Distribution of Circles by Number of Industry Categories',
        text='Number of Circles',  # Display count values on bars
        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
    )
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis=dict(
            title="Number of Different Industry Categories",
            tickmode='linear',
            dtick=1,  # Force integer labels
            range=[0.5, 4.5]  # Since we have 4 categories max
        ),
        yaxis_title="Number of Circles"
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True, key="plot_643")
    
    # Show a table with the data
    st.caption("Data table:")
    st.dataframe(plot_df, hide_index=True)
    
    # Display the Industry diversity score
    st.subheader("Industry Diversity Score")
    st.write("""
    For each circle, the industry diversity score is calculated as follows:
    - 1 point: All members in the same industry category
    - 2 points: Members from two different industry categories
    - 3 points: Members from three different industry categories
    - 4 points: Members from all four industry categories
    """)
    
    # Calculate average and total diversity scores
    total_diversity_score = sum(circle_industry_diversity_scores.values()) if circle_industry_diversity_scores else 0
    avg_diversity_score = total_diversity_score / len(circle_industry_diversity_scores) if circle_industry_diversity_scores else 0
    
    # Create two columns for the metrics
    col1, col2 = st.columns(2)
    
    # Show average and total scores
    with col1:
        st.metric("Average Industry Diversity Score", f"{avg_diversity_score:.2f}")
    with col2:
        st.metric("Total Industry Diversity Score", f"{total_diversity_score}")
    
    # Add a brief explanation
    total_circles = sum(diversity_counts.values)
    diverse_circles = sum(diversity_counts[diversity_counts.index > 1].values)
    diverse_pct = (diverse_circles / total_circles * 100) if total_circles > 0 else 0
    
    st.write(f"Out of {total_circles} total circles, {diverse_circles} ({diverse_pct:.1f}%) contain members from multiple industry categories.")

def render_employment_analysis(data):
    """Render the Employment analysis visualizations"""
    st.subheader("Employment Analysis")
    
    # Check if data is None
    if data is None:
        st.warning("No data available for analysis. Please upload participant data.")
        return
    
    # Create a copy to work with
    df = data.copy()
    
    # Debug: show column names
    st.caption("Debugging information:")
    with st.expander("Show data columns and sample values"):
        st.write("Available columns:")
        for col in df.columns:
            st.text(f"- {col}")
        
        # Try to find Employment Status column
        employment_status_col = None
        for col in df.columns:
            if "employment status" in col.lower():
                employment_status_col = col
                break
        
        if employment_status_col:
            st.write(f"Found Employment Status column: {employment_status_col}")
            # Show some sample values
            sample_values = df[employment_status_col].dropna().head(5).tolist()
            st.write(f"Sample values: {sample_values}")
        else:
            st.write("No Employment Status column found in the data")
    
    # Check if we need to create Employment Category
    if 'Employment_Category' not in df.columns:
        if employment_status_col:
            st.info(f"Creating Employment Category from {employment_status_col}...")
            
            # Define function to categorize employment status
            def categorize_employment(status):
                if pd.isna(status):
                    return None
                
                # Convert to string in case it's not
                status_str = str(status)
                
                # Apply categorization rules
                if "Employed full-time for wages" in status_str:
                    return "Employed full-time for wages"
                elif ("Self-Employed" in status_str or "Self-employed" in status_str) and "Employed full-time for wages" not in status_str:
                    return "Self-employed"
                else:
                    return "All Else"
            
            # Apply the categorization function
            df['Employment_Category'] = df[employment_status_col].apply(categorize_employment)
            
            # Update session state with the new Employment_Category
            if 'results' in st.session_state and st.session_state.results is not None:
                # Copy the newly created Employment_Category to the results DataFrame
                # First, create a dictionary mapping Encoded ID to Employment_Category
                emp_cat_mapping = dict(zip(df['Encoded ID'], df['Employment_Category']))
                
                # Then apply this mapping to the results DataFrame
                if 'Encoded ID' in st.session_state.results.columns:
                    st.session_state.results['Employment_Category'] = st.session_state.results['Encoded ID'].map(emp_cat_mapping)
                    st.info("Updated results data with Employment Categories")
        else:
            st.warning("Employment Status data is not available. Please ensure Employment Status data was included in the uploaded file.")
            return
    
    # Filter out rows with missing Employment Category
    df = df[df['Employment_Category'].notna()]
    
    if len(df) == 0:
        st.warning("No Employment Category data is available after filtering.")
        return
    
    # Define the proper order for Employment Categories
    employment_order = [
        "Employed full-time for wages", "Self-employed", "All Else"
    ]
    
    # FIRST: Display diversity within circles IF we have matched circles
    if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
        if not (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty):
            render_employment_diversity_histogram()
        else:
            st.info("Run the matching algorithm to see the Employment diversity within circles.")
    else:
        st.info("Run the matching algorithm to see the Employment diversity within circles.")
    
    # SECOND: Display Distribution of Employment
    st.subheader("Distribution of Employment")
    
    # Count by Employment Category
    employment_counts = df['Employment_Category'].value_counts().reindex(employment_order).fillna(0).astype(int)
    
    # Create a DataFrame for plotting
    employment_df = pd.DataFrame({
        'Employment Category': employment_counts.index,
        'Count': employment_counts.values
    })
    
    # Create histogram using plotly with Stanford cardinal red color
    fig = px.bar(
        employment_df,
        x='Employment Category',
        y='Count',
        title='Distribution of Employment Categories',
        text='Count',  # Display count values on bars
        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
    )
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': employment_order},
        xaxis_title="Employment Category",
        yaxis_title="Count of Participants"
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True, key="plot_803")
    
    # Create a breakdown by Status if Status column exists
    if 'Status' in df.columns:
        st.subheader("Employment by Status")
        
        # Create a crosstab of Employment Category vs Status
        status_employment = pd.crosstab(
            df['Employment_Category'], 
            df['Status'],
            rownames=['Employment Category'],
            colnames=['Status']
        ).reindex(employment_order)
        
        # Add a Total column
        status_employment['Total'] = status_employment.sum(axis=1)
        
        # Calculate percentages
        for col in status_employment.columns:
            if col != 'Total':
                status_employment[f'{col} %'] = (status_employment[col] / status_employment['Total'] * 100).round(1)
        
        # Reorder columns to group counts with percentages
        cols = []
        for status in status_employment.columns:
            if status != 'Total' and not status.endswith(' %'):
                cols.append(status)
                cols.append(f'{status} %')
        cols.append('Total')
        
        # Show the table
        st.dataframe(status_employment[cols], use_container_width=True)
        
        # Create a stacked bar chart
        fig = px.bar(
            status_employment.reset_index(),
            x='Employment Category',
            y=[col for col in status_employment.columns if col != 'Total' and not col.endswith(' %')],
            title='Employment Distribution by Status',
            barmode='stack',
            color_discrete_sequence=['#8C1515', '#2E2D29']  # Stanford colors
        )
        
        # Customize layout
        fig.update_layout(
            xaxis={'categoryorder': 'array', 'categoryarray': employment_order},
            xaxis_title="Employment Category",
            yaxis_title="Count of Participants",
            legend_title="Status"
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True, key="plot_855")


def render_industry_analysis(data):
    """Render the Industry analysis visualizations"""
    st.subheader("Industry Analysis")
    
    # Check if data is None
    if data is None:
        st.warning("No data available for analysis. Please upload participant data.")
        return
    
    # Create a copy to work with
    df = data.copy()
    
    # Debug: show column names
    st.caption("Debugging information:")
    with st.expander("Show data columns and sample values"):
        st.write("Available columns:")
        for col in df.columns:
            st.text(f"- {col}")
        
        # Try to find Industry Sector column
        industry_sector_col = None
        for col in df.columns:
            if "industry sector" in col.lower():
                industry_sector_col = col
                break
        
        if industry_sector_col:
            st.write(f"Found Industry Sector column: {industry_sector_col}")
            # Show some sample values
            sample_values = df[industry_sector_col].dropna().head(5).tolist()
            st.write(f"Sample values: {sample_values}")
        else:
            st.write("No Industry Sector column found in the data")
    
    # Check if we need to create Industry Category
    if 'Industry_Category' not in df.columns:
        if industry_sector_col:
            st.info(f"Creating Industry Category from {industry_sector_col}...")
            
            # Define function to categorize industry sector
            def categorize_industry(sector):
                if pd.isna(sector):
                    return None
                
                # Convert to string in case it's not
                sector_str = str(sector)
                
                # Apply categorization rules
                if "Technology" in sector_str:
                    return "Technology"
                elif "Consulting" in sector_str and "Technology" not in sector_str:
                    return "Consulting"
                elif any(term in sector_str for term in ["Finance", "Investment", "Private Equity"]) and \
                     "Technology" not in sector_str and "Consulting" not in sector_str:
                    return "Finance / Investment / Private Equity"
                else:
                    return "All Else"
            
            # Apply the categorization function
            df['Industry_Category'] = df[industry_sector_col].apply(categorize_industry)
            
            # Update session state with the new Industry_Category
            if 'results' in st.session_state and st.session_state.results is not None:
                # Copy the newly created Industry_Category to the results DataFrame
                # First, create a dictionary mapping Encoded ID to Industry_Category
                ind_cat_mapping = dict(zip(df['Encoded ID'], df['Industry_Category']))
                
                # Then apply this mapping to the results DataFrame
                if 'Encoded ID' in st.session_state.results.columns:
                    st.session_state.results['Industry_Category'] = st.session_state.results['Encoded ID'].map(ind_cat_mapping)
                    st.info("Updated results data with Industry Categories")
        else:
            st.warning("Industry Sector data is not available. Please ensure Industry Sector data was included in the uploaded file.")
            return
    
    # Filter out rows with missing Industry Category
    df = df[df['Industry_Category'].notna()]
    
    if len(df) == 0:
        st.warning("No Industry Category data is available after filtering.")
        return
    
    # Define the proper order for Industry Categories
    industry_order = [
        "Technology", "Consulting", "Finance / Investment / Private Equity", "All Else"
    ]
    
    # FIRST: Display diversity within circles IF we have matched circles
    if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
        if not (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty):
            render_industry_diversity_histogram()
        else:
            st.info("Run the matching algorithm to see the Industry diversity within circles.")
    else:
        st.info("Run the matching algorithm to see the Industry diversity within circles.")
    
    # SECOND: Display Distribution of Industry
    st.subheader("Distribution of Industry")
    
    # Count by Industry Category
    industry_counts = df['Industry_Category'].value_counts().reindex(industry_order).fillna(0).astype(int)
    
    # Create a DataFrame for plotting
    industry_df = pd.DataFrame({
        'Industry Category': industry_counts.index,
        'Count': industry_counts.values
    })
    
    # Create histogram using plotly with Stanford cardinal red color
    fig = px.bar(
        industry_df,
        x='Industry Category',
        y='Count',
        title='Distribution of Industry Categories',
        text='Count',  # Display count values on bars
        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
    )
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': industry_order},
        xaxis_title="Industry Category",
        yaxis_title="Count of Participants"
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True, key="plot_985")
    
    # Create a breakdown by Status if Status column exists
    if 'Status' in df.columns:
        st.subheader("Industry by Status")
        
        # Create a crosstab of Industry Category vs Status
        status_industry = pd.crosstab(
            df['Industry_Category'], 
            df['Status'],
            rownames=['Industry Category'],
            colnames=['Status']
        ).reindex(industry_order)
        
        # Add a Total column
        status_industry['Total'] = status_industry.sum(axis=1)
        
        # Calculate percentages
        for col in status_industry.columns:
            if col != 'Total':
                status_industry[f'{col} %'] = (status_industry[col] / status_industry['Total'] * 100).round(1)
        
        # Reorder columns to group counts with percentages
        cols = []
        for status in status_industry.columns:
            if status != 'Total' and not status.endswith(' %'):
                cols.append(status)
                cols.append(f'{status} %')
        cols.append('Total')
        
        # Show the table
        st.dataframe(status_industry[cols], use_container_width=True)
        
        # Create a stacked bar chart
        fig = px.bar(
            status_industry.reset_index(),
            x='Industry Category',
            y=[col for col in status_industry.columns if col != 'Total' and not col.endswith(' %')],
            title='Industry Distribution by Status',
            barmode='stack',
            color_discrete_sequence=['#8C1515', '#2E2D29']  # Stanford colors
        )
        
        # Customize layout
        fig.update_layout(
            xaxis={'categoryorder': 'array', 'categoryarray': industry_order},
            xaxis_title="Industry Category",
            yaxis_title="Count of Participants",
            legend_title="Status"
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True, key="plot_1037")

def render_class_vintage_analysis(data):
    """Render the Class Vintage analysis visualizations"""
    st.subheader("Class Vintage Analysis")
    
    # Check if data is None
    if data is None:
        st.warning("No data available for analysis. Please upload participant data.")
        return
    
    # Create a copy to work with
    df = data.copy()
    
    # Debug: show column names
    st.caption("Debugging information:")
    with st.expander("Show data columns and sample values"):
        st.write("Available columns:")
        for col in df.columns:
            st.text(f"- {col}")
        
        # Try to find GSB Class column
        gsb_class_col = None
        for col in df.columns:
            if any(term in col.lower().replace(" ", "") for term in ['gsbclass', 'gsb class']):
                gsb_class_col = col
                break
        
        if gsb_class_col:
            st.write(f"Found GSB Class column: {gsb_class_col}")
            # Show some sample values
            sample_values = df[gsb_class_col].dropna().head(5).tolist()
            st.write(f"Sample values: {sample_values}")
        else:
            st.write("No GSB Class column found in the data")
    
    # Check if we have Class Vintage data
    if 'Class_Vintage' not in data.columns:
        st.warning("Class Vintage data is not available. Please ensure GSB Class data was included in the uploaded file.")
        
        # Try to calculate Class Vintage on-the-fly if there's a GSB Class column
        if gsb_class_col:
            st.info(f"Attempting to calculate Class Vintage from {gsb_class_col}...")
            try:
                from modules.data_processor import calculate_class_vintage
                
                # Convert to numeric and calculate vintage
                df['GSB_Class_Numeric'] = pd.to_numeric(df[gsb_class_col], errors='coerce')
                df['Class_Vintage'] = df['GSB_Class_Numeric'].apply(calculate_class_vintage)
                
                vintage_counts = df['Class_Vintage'].value_counts()
                st.success(f"Successfully calculated Class Vintage for {len(df[df['Class_Vintage'].notna()])} records")
                st.write(f"Distribution: {vintage_counts}")
                
                # Update session state with the new Class_Vintage
                if 'results' in st.session_state and st.session_state.results is not None:
                    # First, create a dictionary mapping Encoded ID to Class_Vintage
                    vintage_mapping = dict(zip(df['Encoded ID'], df['Class_Vintage']))
                    
                    # Then apply this mapping to the results DataFrame
                    if 'Encoded ID' in st.session_state.results.columns:
                        st.session_state.results['Class_Vintage'] = st.session_state.results['Encoded ID'].map(vintage_mapping)
                        st.info("Updated results data with Class Vintage")
            except Exception as e:
                st.error(f"Error calculating Class Vintage: {str(e)}")
                return
        else:
            return
    
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
    
    # FIRST: Display diversity within circles IF we have matched circles
    if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
        if not (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty):
            render_vintage_diversity_histogram()
        else:
            st.info("Run the matching algorithm to see the Class Vintage diversity within circles.")
    else:
        st.info("Run the matching algorithm to see the Class Vintage diversity within circles.")
    
    # SECOND: Display Distribution of Class Vintage
    st.subheader("Distribution of Class Vintage")
    
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
    st.plotly_chart(fig, use_container_width=True, key="plot_1159")
    
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
        st.plotly_chart(fig, use_container_width=True, key="plot_1211")
    
    # We have removed the "Class Vintage by Match Status", "Class Vintage Distribution by Match Status", 
    # and "Match Rate by Class Vintage" sections per user request

def render_debug_tab():
    """Render the debug tab content"""
    st.subheader("Debug Information")

    # Add special Houston debug section at the top
    st.write("### Houston Circles Debug")
    
    # Show a powerful data verification section FIRST
    st.subheader("🔴 Data Verification & Variable Tracking")
    st.write("This section directly inspects input data and model variables to diagnose issues")
    
    if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        st.success(f"✅ Processed data available with {len(df)} participants")
        
        # Check directly for our test participant
        test_id = '72549701782'
        if test_id in df['Encoded ID'].values:
            st.success(f"✅ TEST PARTICIPANT FOUND: {test_id}")
            
            # Show test participant details
            test_row = df[df['Encoded ID'] == test_id].iloc[0]
            test_info = {
                "ID": test_id,
                "Status": test_row.get('Status', 'N/A'),
                "Current Region": test_row.get('Current_Region', 'N/A'),
                "Location Preferences": [
                    test_row.get('first_choice_location', 'N/A'),
                    test_row.get('second_choice_location', 'N/A'),
                    test_row.get('third_choice_location', 'N/A')
                ],
                "Time Preferences": [
                    test_row.get('first_choice_time', 'N/A'),
                    test_row.get('second_choice_time', 'N/A'),
                    test_row.get('third_choice_time', 'N/A')
                ]
            }
            st.json(test_info)
        else:
            st.error(f"❌ TEST PARTICIPANT NOT FOUND: {test_id}")
            st.info("The test participant should be in the data but isn't - this is a critical issue")
        
        # Check for Houston circles
        houston_filter = df['Current_Circle_ID'].astype(str).str.contains('HOU', na=False)
        houston_circles = df[houston_filter]
        
        if len(houston_circles) > 0:
            st.success(f"✅ Found {len(houston_circles)} participants in Houston circles")
            
            # Get unique circle IDs
            circle_ids = houston_circles['Current_Circle_ID'].unique()
            st.write(f"Houston circles: {', '.join(circle_ids)}")
            
            # Check specifically for IP-HOU-02
            ip_hou_02 = df[df['Current_Circle_ID'] == 'IP-HOU-02']
            if len(ip_hou_02) > 0:
                st.success(f"✅ IP-HOU-02 circle found with {len(ip_hou_02)} members")
                
                # Show detailed info about this circle
                with st.expander("IP-HOU-02 Details"):
                    st.dataframe(ip_hou_02)
                    
                    # Try to extract meeting time with multiple column name options
                    day_column = None
                    time_column = None
                    
                    # Try different potential column names for meeting day - same logic as in optimizer
                    for col_name in ['Current_Meeting_Day', 'Current Meeting Day', 'Current/ Continuing Meeting Day']:
                        if col_name in ip_hou_02.columns:
                            day_column = col_name
                            break
                    
                    # Try different potential column names for meeting time
                    for col_name in ['Current_Meeting_Time', 'Current Meeting Time', 'Current/ Continuing Meeting Time']:
                        if col_name in ip_hou_02.columns:
                            time_column = col_name
                            break
                    
                    # Get meeting day and time, defaulting to 'Not available' if not found
                    meeting_day = ip_hou_02[day_column].iloc[0] if day_column else 'Not available'
                    meeting_time = ip_hou_02[time_column].iloc[0] if time_column else 'Not available'
                    
                    # Format and display
                    st.write(f"Meeting day: {meeting_day}")
                    st.write(f"Meeting time: {meeting_time}")
                    st.write(f"Formatted meeting time: {meeting_day} ({meeting_time})")
            else:
                st.error("❌ IP-HOU-02 circle not found in the data")
                st.info("The test circle should be in the data but isn't - this is a critical issue")
        else:
            st.error("❌ No Houston circles found in the data")
    else:
        st.error("❌ No processed data available in session state")
    
    # Now check results if available
    if 'results' in st.session_state and st.session_state.results is not None:
        results_df = st.session_state.results
        st.success(f"✅ Results data available with {len(results_df)} participants")
        
        # Check unmatched counts
        if 'proposed_NEW_circles_id' in results_df.columns:
            unmatched = results_df[results_df['proposed_NEW_circles_id'] == 'UNMATCHED']
            st.info(f"Unmatched participants: {len(unmatched)} out of {len(results_df)} ({len(unmatched)/len(results_df)*100:.2f}%)")
        
        # Check for missing preference data
        missing_prefs_cols = []
        for pref_type in ['location', 'time']:
            for choice_num in range(1, 4):
                col_name = f'first_choice_{pref_type}'
                if col_name in results_df.columns:
                    missing_count = results_df[col_name].isna().sum()
                    if missing_count > 0:
                        missing_prefs_cols.append(f"{col_name}: {missing_count} missing")
        
        if missing_prefs_cols:
            st.warning("⚠️ Missing preference data:")
            for col_info in missing_prefs_cols:
                st.text(f"- {col_info}")
        else:
            st.success("✅ All preference columns are populated")
        
        # Examine unmatched reasons
        if 'unmatched_reason' in results_df.columns:
            unmatched_reasons = results_df['unmatched_reason'].value_counts()
            if not unmatched_reasons.empty:
                st.subheader("Unmatched Reasons")
                
                # Create a DataFrame for plotting
                reason_df = pd.DataFrame({
                    'Reason': unmatched_reasons.index,
                    'Count': unmatched_reasons.values
                })
                
                # Create a bar chart
                fig = px.bar(
                    reason_df,
                    x='Reason',
                    y='Count',
                    title='Distribution of Unmatched Reasons',
                    text='Count',
                    color_discrete_sequence=['#8C1515']
                )
                
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    xaxis_title="Unmatched Reason",
                    yaxis_title="Count of Participants"
                )
                
                st.plotly_chart(fig, use_container_width=True, key="plot_1365")
                
                # Create a table
                st.dataframe(reason_df)
                
                # Examine specific reasons
                if 'NO_LOCATION_MATCH' in unmatched_reasons:
                    no_location_match = results_df[results_df['unmatched_reason'] == 'NO_LOCATION_MATCH']
                    
                    with st.expander(f"NO_LOCATION_MATCH ({len(no_location_match)} participants)"):
                        # Extract regions
                        regions = []
                        if 'Current_Region' in no_location_match.columns:
                            regions = no_location_match['Current_Region'].value_counts()
                        elif 'Region' in no_location_match.columns:
                            regions = no_location_match['Region'].value_counts()
                        
                        if not regions.empty:
                            st.write("Regions of participants with NO_LOCATION_MATCH:")
                            st.dataframe(pd.DataFrame({
                                'Region': regions.index,
                                'Count': regions.values
                            }))
                
                # Add a debugging function to check time format issues
                with st.expander("Time Format Debugging"):
                    st.subheader("🔍 Time Format Investigation")
                    st.write("This section helps identify any issues with time format parsing")
                    
                    # Find relevant time columns
                    time_columns = []
                    for col in results_df.columns:
                        if col.endswith('_time') and 'choice' in col:
                            time_columns.append(col)
                    
                    if time_columns:
                        # Select a column to examine
                        selected_time_col = st.selectbox("Select time column to examine", 
                                                        options=time_columns,
                                                        index=0)
                        
                        # Define a demonstration function to show time parsing
                        def extract_days_and_period_demo(time_str):
                            """Demonstration version that returns detailed info for display"""
                            if pd.isna(time_str) or not time_str:
                                return {
                                    "original": str(time_str),
                                    "is_valid": False,
                                    "days_found": [],
                                    "error": "Empty or null value"
                                }
                            
                            # Extract day information
                            days_mapping = {
                                'mon': 'Monday',
                                'tue': 'Tuesday', 
                                'wed': 'Wednesday',
                                'thu': 'Thursday',
                                'fri': 'Friday',
                                'sat': 'Saturday',
                                'sun': 'Sunday'
                            }
                            
                            # Convert to lowercase for easier matching
                            time_str_lower = str(time_str).lower()
                            
                            # Find days mentioned in the string
                            days_found = []
                            for day_abbr, day_full in days_mapping.items():
                                if day_abbr in time_str_lower:
                                    days_found.append(day_full)
                            
                            # Check for morning/afternoon/evening
                            period_found = None
                            if any(term in time_str_lower for term in ['am', 'morning']):
                                period_found = 'Morning'
                            elif any(term in time_str_lower for term in ['pm', 'afternoon']):
                                period_found = 'Afternoon'
                            elif 'evening' in time_str_lower:
                                period_found = 'Evening'
                            
                            # Return detailed parsing results
                            return {
                                "original": time_str,
                                "is_valid": bool(days_found),
                                "days_found": days_found,
                                "period_found": period_found,
                                "error": None if days_found else "No valid days found"
                            }
                        
                        # Simplified version that just returns extracted days
                        def extract_days_demo(time_str):
                            """Simplified version of extract_days_and_period_demo specifically for UI display"""
                            result = extract_days_and_period_demo(time_str)
                            return result["days_found"]
                        
                        # Sample a few values to show parsing
                        sample_df = results_df[selected_time_col].dropna().sample(min(5, len(results_df))).tolist()
                        
                        st.write("Sample parsing results:")
                        for sample in sample_df:
                            parsing_result = extract_days_and_period_demo(sample)
                            st.write(f"Original: '{parsing_result['original']}'")
                            st.write(f"Valid: {parsing_result['is_valid']}")
                            st.write(f"Days found: {', '.join(parsing_result['days_found']) if parsing_result['days_found'] else 'None'}")
                            st.write(f"Period: {parsing_result['period_found'] or 'Not found'}")
                            st.write("---")
                        
                        # Count parsing successes and failures
                        time_values = results_df[selected_time_col].dropna()
                        parsing_success = sum(1 for t in time_values if extract_days_demo(t))
                        parsing_failure = len(time_values) - parsing_success
                        
                        st.metric("Time values successfully parsed", 
                                f"{parsing_success} / {len(time_values)} ({parsing_success/len(time_values)*100:.1f}%)")
                        
                        if parsing_failure > 0:
                            st.warning(f"⚠️ {parsing_failure} time values could not be parsed")
                            
                            # Show problematic values
                            problem_values = [t for t in time_values if not extract_days_demo(t)]
                            if problem_values:
                                st.write("Problematic time values:")
                                for idx, val in enumerate(problem_values[:10]):  # Show at most 10
                                    st.text(f"{idx+1}. '{val}'")
                                if len(problem_values) > 10:
                                    st.text(f"... and {len(problem_values) - 10} more")
                    else:
                        st.info("No time-related columns found in the data")
    else:
        st.error("❌ No results data available in session state")
    
    # Check for matched circles
    if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
        circles_df = st.session_state.matched_circles
        if hasattr(circles_df, 'empty') and circles_df.empty:
            st.error("❌ Matched circles dataframe is empty")
        else:
            st.success(f"✅ Matched circles data available with {len(circles_df)} circles")
            
            # Count total members
            total_members = circles_df['member_count'].sum() if 'member_count' in circles_df.columns else "N/A"
            st.metric("Total matched participants", total_members)
            
            # Circle size distribution
            if 'member_count' in circles_df.columns:
                circle_sizes = circles_df['member_count'].value_counts().sort_index()
                
                st.subheader("Circle Size Distribution")
                
                # Create a DataFrame for plotting
                size_df = pd.DataFrame({
                    'Circle Size': circle_sizes.index,
                    'Number of Circles': circle_sizes.values
                })
                
                # Create a bar chart
                fig = px.bar(
                    size_df,
                    x='Circle Size',
                    y='Number of Circles',
                    title='Distribution of Circle Sizes',
                    text='Number of Circles',
                    color_discrete_sequence=['#8C1515']
                )
                
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    xaxis_title="Number of Members in Circle",
                    yaxis_title="Number of Circles"
                )
                
                st.plotly_chart(fig, use_container_width=True, key="plot_1537")
    else:
        st.error("❌ No matched circles data available in session state")
    
    # Show information about current session state
    st.subheader("🔢 Session State Variables")
    session_keys = list(st.session_state.keys())
    selected_keys = st.multiselect("Select session state variables to inspect", 
                                 options=session_keys,
                                 default=session_keys[:3] if len(session_keys) > 3 else session_keys)
    
    if selected_keys:
        for key in selected_keys:
            value = st.session_state[key]
            st.write(f"**{key}**:")
            if isinstance(value, pd.DataFrame):
                st.write(f"DataFrame with {len(value)} rows and {len(value.columns)} columns")
                st.dataframe(value.head(5))
            elif value is None:
                st.write("None")
            elif isinstance(value, (list, tuple, set)):
                st.write(f"{type(value).__name__} with {len(value)} items")
                st.write(str(value)[:1000] + "..." if len(str(value)) > 1000 else str(value))
            else:
                st.write(str(value)[:1000] + "..." if len(str(value)) > 1000 else str(value))


def render_results_overview():
    """Render the results overview section"""
    if ('matched_circles' not in st.session_state or 
        st.session_state.matched_circles is None or 
        (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty)):
        st.warning("No matching results available. Please run the matching algorithm first.")
        return
    
    st.subheader("Matching Results Overview")
    
    # Get the data
    matched_df = st.session_state.matched_circles
    results_df = st.session_state.results if 'results' in st.session_state else None
    
    # Create columns for the metrics
    col1, col2, col3 = st.columns(3)
    
    # Column 1: Circle stats
    with col1:
        st.metric("Number of Circles", len(matched_df))
        
        # Average circle size
        if 'member_count' in matched_df.columns:
            avg_size = matched_df['member_count'].mean()
            st.metric("Average Circle Size", f"{avg_size:.1f}")
    
    # Column 2: Participant stats
    with col2:
        # Total participants
        total_matched = matched_df['member_count'].sum() if 'member_count' in matched_df.columns else 0
        st.metric("Participants Matched", total_matched)
        
        # Total unmatched
        if results_df is not None and 'proposed_NEW_circles_id' in results_df.columns:
            unmatched_count = len(results_df[results_df['proposed_NEW_circles_id'] == 'UNMATCHED'])
            st.metric("Participants Unmatched", unmatched_count)
    
    # Column 3: Success rates
    with col3:
        # Match success rate
        if results_df is not None and 'proposed_NEW_circles_id' in results_df.columns:
            total_participants = len(results_df)
            match_rate = (total_matched / total_participants) * 100 if total_participants > 0 else 0
            st.metric("Match Success Rate", f"{match_rate:.1f}%")
        
        # Circles with target size
        if 'member_count' in matched_df.columns and 'target_size' in matched_df.columns:
            target_count = len(matched_df[matched_df['member_count'] == matched_df['target_size']])
            target_pct = (target_count / len(matched_df)) * 100 if len(matched_df) > 0 else 0
            st.metric("Circles at Target Size", f"{target_count} ({target_pct:.1f}%)")
    
    # Distribution of circle sizes
    if 'member_count' in matched_df.columns:
        st.subheader("Circle Size Distribution")
        
        # Count circles by size
        size_counts = matched_df['member_count'].value_counts().sort_index()
        
        # Create a DataFrame for plotting
        size_df = pd.DataFrame({
            'Circle Size': size_counts.index,
            'Count': size_counts.values
        })
        
        # Create a bar chart
        fig = px.bar(
            size_df,
            x='Circle Size',
            y='Count',
            title='Distribution of Circle Sizes',
            text='Count',
            color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
        )
        
        # Format
        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis=dict(
                title="Number of Members",
                tickmode='linear',
                dtick=1
            ),
            yaxis_title="Number of Circles"
        )
        
        # Plot with unique key
        st.plotly_chart(fig, use_container_width=True, key="results_overview_circle_size_dist")
    
    # Show unmatched reasons if available
    render_unmatched_table()
    
    # Show circle composition table
    render_circle_table()


def render_circle_table():
    """Render the circle composition table"""
    if ('matched_circles' not in st.session_state or 
        st.session_state.matched_circles is None or
        (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty)):
        return
    
    st.subheader("Circle Composition")
    
    # Get the data
    circles_df = st.session_state.matched_circles.copy()
    results_df = st.session_state.results.copy() if 'results' in st.session_state else None
    
    # Show the table
    if 'circle_id' in circles_df.columns and 'meeting_time' in circles_df.columns:
        # Create a display table with key information
        display_cols = ['circle_id', 'meeting_time', 'meeting_location', 'member_count']
        display_cols = [col for col in display_cols if col in circles_df.columns]
        
        if display_cols:
            display_df = circles_df[display_cols].copy()
            
            # Rename columns for display
            display_df.columns = [col.replace('_', ' ').title() for col in display_cols]
            
            # Sort by circle ID
            if 'Circle Id' in display_df.columns:
                display_df = display_df.sort_values('Circle Id')
            
            # Show the table
            st.dataframe(display_df, use_container_width=True)
            
            # Add an export option
            if st.button("Export Circle Data to CSV"):
                # Convert DataFrame to CSV
                csv = circles_df.to_csv(index=False)
                
                # Create a download link
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="circle_data.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("Circle data doesn't contain the expected columns.")
    else:
        st.warning("Circle data doesn't contain circle_id or meeting_time columns.")


def render_unmatched_table():
    """Render the unmatched participants table"""
    if 'results' not in st.session_state or st.session_state.results is None:
        return
    
    # Get unmatched participants
    results_df = st.session_state.results.copy()
    
    if 'proposed_NEW_circles_id' not in results_df.columns:
        return
    
    unmatched_df = results_df[results_df['proposed_NEW_circles_id'] == 'UNMATCHED']
    
    if len(unmatched_df) == 0:
        st.success("All participants were successfully matched!")
        return
    
    st.subheader(f"Unmatched Participants ({len(unmatched_df)})")
    
    # Show reasons for being unmatched
    if 'unmatched_reason' in unmatched_df.columns:
        reasons = unmatched_df['unmatched_reason'].value_counts()
        
        # Create a DataFrame for plotting
        reason_df = pd.DataFrame({
            'Reason': reasons.index,
            'Count': reasons.values
        })
        
        # Create a bar chart
        fig = px.bar(
            reason_df,
            x='Reason',
            y='Count',
            title='Reasons for Unmatched Participants',
            text='Count',
            color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
        )
        
        # Format
        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis_title="Unmatched Reason",
            yaxis_title="Number of Participants"
        )
        
        # Plot with unique key
        st.plotly_chart(fig, use_container_width=True, key="unmatched_reasons_chart")
    
    # Show the table of unmatched participants
    display_cols = ['Last Family Name', 'First Given Name', 'Encoded ID', 
                    'Current_Region', 'Status', 'unmatched_reason']
    
    # Filter to available columns
    display_cols = [col for col in display_cols if col in unmatched_df.columns]
    
    if display_cols:
        st.dataframe(unmatched_df[display_cols], use_container_width=True)
    else:
        st.warning("Unmatched participant data doesn't contain the expected columns.")


def render_circle_details():
    """Render detailed information about each circle"""
    if ('matched_circles' not in st.session_state or 
        st.session_state.matched_circles is None or
        (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty)):
        st.warning("No matching results available. Please run the matching algorithm first.")
        return
    
    # Get the data
    circles_df = st.session_state.matched_circles.copy()
    results_df = st.session_state.results.copy() if 'results' in st.session_state else None
    
    # If results aren't available, we can't show member details
    if results_df is None:
        st.warning("Participant data is not available. Cannot show detailed circle information.")
        return
    
    # Get all circle IDs
    circle_ids = circles_df['circle_id'].tolist() if 'circle_id' in circles_df.columns else []
    
    if not circle_ids:
        st.warning("No circle IDs found in the matching results.")
        return
    
    # Sort circle IDs for better UX
    circle_ids.sort()
    
    # Create a selection widget to choose a circle
    selected_circle = st.selectbox("Select a circle to view details", options=circle_ids)
    
    # Get the selected circle's data
    circle_row = circles_df[circles_df['circle_id'] == selected_circle].iloc[0]
    
    # Create columns for the display
    col1, col2 = st.columns([1, 2])
    
    # First column: Circle metadata
    with col1:
        st.subheader(f"Circle: {selected_circle}")
        
        # Show circle metadata
        metadata = {
            "Meeting Time": circle_row.get('meeting_time', 'Not specified'),
            "Meeting Location": circle_row.get('meeting_location', 'Not specified'),
            "Member Count": circle_row.get('member_count', 'Unknown'),
            "Target Size": circle_row.get('target_size', 'Unknown'),
        }
        
        for key, value in metadata.items():
            st.write(f"**{key}:** {value}")
    
    # Second column: Members table
    with col2:
        st.subheader("Members")
        
        # Get member IDs
        member_ids = []
        if 'members' in circle_row:
            # Handle both list and string representations
            if isinstance(circle_row['members'], list):
                member_ids = circle_row['members']
            elif isinstance(circle_row['members'], str):
                try:
                    # Try to evaluate if it's a string representation of a list
                    if circle_row['members'].startswith('['):
                        member_ids = eval(circle_row['members'])
                    else:
                        member_ids = [circle_row['members']]
                except:
                    st.error(f"Could not parse member list: {circle_row['members']}")
        
        if not member_ids:
            st.warning("No members found for this circle.")
            return
        
        # Get member data
        members_df = results_df[results_df['Encoded ID'].isin(member_ids)]
        
        # Create a display table
        display_cols = ['Last Family Name', 'First Given Name', 'Encoded ID', 
                        'Current_Region', 'Status', 'first_choice_time']
        
        # Filter to available columns
        display_cols = [col for col in display_cols if col in members_df.columns]
        
        if display_cols:
            st.dataframe(members_df[display_cols], use_container_width=True)
        else:
            st.warning("Member data doesn't contain the expected columns.")
    
    # Show visualizations specific to this circle
    st.subheader("Circle Analysis")
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2 = st.tabs(["Time Preferences", "Location Preferences"])
    
    with viz_tab1:
        # Time preference visualization
        st.write("Time Preferences Distribution")
        
        # Extract time preferences
        time_prefs = []
        for choice_num in range(1, 4):
            col_name = f'first_choice_time'
            if col_name in members_df.columns:
                prefs = members_df[col_name].dropna().tolist()
                time_prefs.extend(prefs)
        
        if time_prefs:
            # Create a simple display of common times
            from collections import Counter
            time_counts = Counter(time_prefs)
            
            # Create a DataFrame for plotting
            time_df = pd.DataFrame({
                'Time Preference': list(time_counts.keys()),
                'Count': list(time_counts.values())
            }).sort_values('Count', ascending=False)
            
            # Show as a table
            st.dataframe(time_df, use_container_width=True)
        else:
            st.info("No time preference data available for this circle.")
    
    with viz_tab2:
        # Location preference visualization
        st.write("Location Preferences Distribution")
        
        # Extract location preferences
        loc_prefs = []
        for choice_num in range(1, 4):
            col_name = f'first_choice_location'
            if col_name in members_df.columns:
                prefs = members_df[col_name].dropna().tolist()
                loc_prefs.extend(prefs)
        
        if loc_prefs:
            # Create a simple display of common locations
            from collections import Counter
            loc_counts = Counter(loc_prefs)
            
            # Create a DataFrame for plotting
            loc_df = pd.DataFrame({
                'Location Preference': list(loc_counts.keys()),
                'Count': list(loc_counts.values())
            }).sort_values('Count', ascending=False)
            
            # Show as a table
            st.dataframe(loc_df, use_container_width=True)
        else:
            st.info("No location preference data available for this circle.")


def render_participant_details():
    """Render detailed information about individual participants"""
    if 'results' not in st.session_state or st.session_state.results is None:
        st.warning("No participant data available. Please run the matching algorithm first.")
        return
    
    # Get the data
    results_df = st.session_state.results.copy()
    
    # Create a search box for participants
    search_type = st.radio("Search by", ["Name", "ID"], horizontal=True)
    
    if search_type == "Name":
        # Create search boxes for first and last name
        col1, col2 = st.columns(2)
        
        with col1:
            first_name_col = 'First Given Name' if 'First Given Name' in results_df.columns else None
            if first_name_col:
                first_name_options = [''] + sorted(results_df[first_name_col].dropna().unique().tolist())
                first_name = st.selectbox("First Name", options=first_name_options)
            else:
                st.warning("First name column not found in data")
                first_name = None
        
        with col2:
            last_name_col = 'Last Family Name' if 'Last Family Name' in results_df.columns else None
            if last_name_col:
                last_name_options = [''] + sorted(results_df[last_name_col].dropna().unique().tolist())
                last_name = st.selectbox("Last Name", options=last_name_options)
            else:
                st.warning("Last name column not found in data")
                last_name = None
        
        # Filter based on name
        filtered_df = results_df.copy()
        if first_name and first_name_col:
            filtered_df = filtered_df[filtered_df[first_name_col] == first_name]
        if last_name and last_name_col:
            filtered_df = filtered_df[filtered_df[last_name_col] == last_name]
    else:
        # Search by ID
        id_col = 'Encoded ID' if 'Encoded ID' in results_df.columns else None
        if id_col:
            id_options = [''] + sorted(results_df[id_col].dropna().unique().tolist())
            participant_id = st.selectbox("Participant ID", options=id_options)
            
            # Filter based on ID
            filtered_df = results_df[results_df[id_col] == participant_id] if participant_id else results_df
        else:
            st.warning("ID column not found in data")
            filtered_df = results_df
    
    # Show the filtered results
    if len(filtered_df) == 0:
        st.info("No participants found matching the search criteria.")
        return
    elif len(filtered_df) > 10:
        st.warning(f"Found {len(filtered_df)} participants matching the search criteria. Please refine your search.")
        
        # Show a sample of the results
        st.dataframe(filtered_df.head(10), use_container_width=True)
        return
    
    # Show detailed information for each matching participant
    for idx, participant in filtered_df.iterrows():
        # Get the participant ID
        participant_id = participant.get('Encoded ID', f"Participant {idx}")
        
        # Create an expander for this participant
        with st.expander(f"Participant: {participant_id}", expanded=True):
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            # Column 1: Basic information
            with col1:
                st.subheader("Basic Information")
                
                # Show basic participant info
                basic_fields = ['First Given Name', 'Last Family Name', 'Encoded ID', 
                                'Current_Region', 'Status']
                
                for field in basic_fields:
                    if field in participant:
                        st.write(f"**{field.replace('_', ' ').title()}:** {participant[field]}")
            
            # Column 2: Match information
            with col2:
                st.subheader("Match Information")
                
                # Show circle assignment if matched
                if 'proposed_NEW_circles_id' in participant:
                    circle_id = participant['proposed_NEW_circles_id']
                    
                    if circle_id == 'UNMATCHED':
                        st.error("**Status:** Unmatched")
                        
                        # Show reason if available
                        if 'unmatched_reason' in participant:
                            st.write(f"**Reason:** {participant['unmatched_reason']}")
                    else:
                        st.success(f"**Assigned Circle:** {circle_id}")
                        
                        # Show circle details if available
                        if ('matched_circles' in st.session_state and 
                            st.session_state.matched_circles is not None and
                            'circle_id' in st.session_state.matched_circles.columns):
                            
                            circles_df = st.session_state.matched_circles
                            circle_info = circles_df[circles_df['circle_id'] == circle_id]
                            
                            if not circle_info.empty:
                                circle_row = circle_info.iloc[0]
                                st.write(f"**Meeting Time:** {circle_row.get('meeting_time', 'Not specified')}")
                                st.write(f"**Meeting Location:** {circle_row.get('meeting_location', 'Not specified')}")
                                st.write(f"**Circle Size:** {circle_row.get('member_count', 'Unknown')}")
            
            # Show preferences section
            st.subheader("Preferences")
            pref_cols = st.columns(2)
            
            with pref_cols[0]:
                st.write("**Time Preferences**")
                for i in range(1, 4):
                    time_col = f"{['first', 'second', 'third'][i-1]}_choice_time"
                    if time_col in participant and pd.notna(participant[time_col]):
                        st.write(f"{i}. {participant[time_col]}")
            
            with pref_cols[1]:
                st.write("**Location Preferences**")
                for i in range(1, 4):
                    loc_col = f"{['first', 'second', 'third'][i-1]}_choice_location"
                    if loc_col in participant and pd.notna(participant[loc_col]):
                        st.write(f"{i}. {participant[loc_col]}")


def render_visualizations():
    """Render visualizations of the matching results"""
    if ('matched_circles' not in st.session_state or 
        st.session_state.matched_circles is None or
        (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty) or
        'results' not in st.session_state or 
        st.session_state.results is None):
        st.warning("No matching results available. Please run the matching algorithm first.")
        return
    
    st.subheader("Matching Visualizations")
    
    # Get the data
    circles_df = st.session_state.matched_circles.copy()
    results_df = st.session_state.results.copy()
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Circle Composition", "Regional Distribution", "Preference Satisfaction"])
    
    with viz_tab1:
        # Circle composition visualization
        st.write("Circle Composition Analysis")
        
        # Circle size distribution
        if 'member_count' in circles_df.columns:
            # Count circles by size
            size_counts = circles_df['member_count'].value_counts().sort_index()
            
            # Create a DataFrame for plotting
            size_df = pd.DataFrame({
                'Circle Size': size_counts.index,
                'Number of Circles': size_counts.values
            })
            
            # Create a bar chart
            fig = px.bar(
                size_df,
                x='Circle Size',
                y='Number of Circles',
                title='Distribution of Circle Sizes',
                text='Number of Circles',
                color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
            )
            
            # Format
            fig.update_traces(textposition='outside')
            fig.update_layout(
                xaxis=dict(
                    title="Number of Members",
                    tickmode='linear',
                    dtick=1
                ),
                yaxis_title="Number of Circles"
            )
            
            # Plot
            st.plotly_chart(fig, use_container_width=True, key="plot_2113")
            
            # Calculate statistics
            total_circles = len(circles_df)
            total_members = circles_df['member_count'].sum()
            avg_size = circles_df['member_count'].mean()
            
            # Show metrics
            cols = st.columns(3)
            cols[0].metric("Total Circles", total_circles)
            cols[1].metric("Total Matched Participants", total_members)
            cols[2].metric("Average Circle Size", f"{avg_size:.1f}")
        else:
            st.warning("Circle size data not available.")
    
    with viz_tab2:
        # Regional distribution visualization
        st.write("Regional Distribution Analysis")
        
        # Check for region column
        region_col = None
        if 'Current_Region' in results_df.columns:
            region_col = 'Current_Region'
        elif 'Region' in results_df.columns:
            region_col = 'Region'
        
        if region_col:
            # Count participants by region
            region_counts = results_df[region_col].value_counts()
            
            # Create a DataFrame for plotting
            region_df = pd.DataFrame({
                'Region': region_counts.index,
                'Count': region_counts.values
            }).sort_values('Count', ascending=False)
            
            # Create a bar chart
            fig = px.bar(
                region_df,
                x='Region',
                y='Count',
                title='Participant Distribution by Region',
                text='Count',
                color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
            )
            
            # Format
            fig.update_traces(textposition='outside')
            fig.update_layout(
                xaxis_title="Region",
                yaxis_title="Number of Participants"
            )
            
            # Plot
            st.plotly_chart(fig, use_container_width=True, key="plot_2167")
            
            # Show table
            st.dataframe(region_df, use_container_width=True)
            
            # Create a matched vs unmatched breakdown by region
            if 'proposed_NEW_circles_id' in results_df.columns:
                # Create a function to determine match status
                def get_match_status(circle_id):
                    return "Unmatched" if circle_id == "UNMATCHED" else "Matched"
                
                # Add match status column
                results_df['Match Status'] = results_df['proposed_NEW_circles_id'].apply(get_match_status)
                
                # Create a crosstab
                region_match = pd.crosstab(
                    results_df[region_col], 
                    results_df['Match Status'],
                    normalize='index'
                ) * 100
                
                # Sort by match rate
                if 'Matched' in region_match.columns:
                    region_match = region_match.sort_values('Matched', ascending=False)
                
                # Create a bar chart
                fig = px.bar(
                    region_match.reset_index(),
                    x=region_col,
                    y=['Matched', 'Unmatched'] if 'Matched' in region_match.columns and 'Unmatched' in region_match.columns else region_match.columns,
                    title='Match Success Rate by Region (%)',
                    barmode='stack',
                    color_discrete_sequence=['#175E54', '#820000']  # Stanford secondary colors
                )
                
                # Format
                fig.update_layout(
                    xaxis_title="Region",
                    yaxis_title="Percentage",
                    yaxis=dict(ticksuffix='%')
                )
                
                # Plot
                st.plotly_chart(fig, use_container_width=True, key="plot_2210")
        else:
            st.warning("Region data not available.")
    
    with viz_tab3:
        # Preference satisfaction visualization
        st.write("Preference Satisfaction Analysis")
        
        # Check if we have the necessary preference data
        if ('proposed_NEW_circles_id' in results_df.columns and 
            'matched_circles' in st.session_state):
            
            # Get only matched participants
            matched_df = results_df[results_df['proposed_NEW_circles_id'] != 'UNMATCHED'].copy()
            
            # Time preference satisfaction
            st.subheader("Time Preference Satisfaction")
            
            # Find columns with time choices
            time_cols = [col for col in matched_df.columns if 'choice_time' in col]
            
            if time_cols and 'proposed_NEW_circles_id' in matched_df.columns:
                # Function to check if assigned time matches preferences
                def check_time_match(row):
                    circle_id = row['proposed_NEW_circles_id']
                    if circle_id == 'UNMATCHED' or pd.isna(circle_id):
                        return None
                    
                    # Get the assigned circle's time
                    circle_info = circles_df[circles_df['circle_id'] == circle_id]
                    if circle_info.empty or 'meeting_time' not in circle_info.columns:
                        return None
                    
                    assigned_time = circle_info.iloc[0]['meeting_time']
                    if pd.isna(assigned_time):
                        return None
                    
                    # Check each preference
                    for i, col in enumerate(time_cols):
                        if col in row and pd.notna(row[col]):
                            preferred_time = row[col]
                            # Very simple check - just look for exact match or substring
                            if preferred_time == assigned_time or preferred_time in assigned_time or assigned_time in preferred_time:
                                return i+1  # Return the preference number (1, 2, 3)
                    
                    return 0  # No match
                
                # Apply the function
                matched_df['time_preference_match'] = matched_df.apply(check_time_match, axis=1)
                
                # Count by match result
                time_match_counts = matched_df['time_preference_match'].value_counts().sort_index()
                
                # Create labels
                match_labels = {
                    0: "No Match",
                    1: "1st Choice",
                    2: "2nd Choice",
                    3: "3rd Choice",
                    None: "Not Available"
                }
                
                # Create a DataFrame for plotting
                match_df = pd.DataFrame({
                    'Preference Match': [match_labels.get(idx, str(idx)) for idx in time_match_counts.index],
                    'Count': time_match_counts.values
                })
                
                # Create a pie chart
                fig = px.pie(
                    match_df,
                    names='Preference Match',
                    values='Count',
                    title='Time Preference Satisfaction',
                    color_discrete_sequence=px.colors.qualitative.D3
                )
                
                # Format
                fig.update_traces(textposition='inside', textinfo='percent+label')
                
                # Plot
                st.plotly_chart(fig, use_container_width=True, key="plot_2291")
                
                # Calculate the percentage of participants who got one of their preferences
                matched_count = sum(time_match_counts.get(i, 0) for i in [1, 2, 3])
                total_count = sum(time_match_counts.values)
                match_pct = (matched_count / total_count) * 100 if total_count > 0 else 0
                
                st.metric("Participants receiving a preferred time", 
                         f"{matched_count} out of {total_count} ({match_pct:.1f}%)")
            else:
                st.warning("Time preference data not available.")
            
            # Location preference satisfaction
            st.subheader("Location Preference Satisfaction")
            
            # Find columns with location choices
            loc_cols = [col for col in matched_df.columns if 'choice_location' in col]
            
            if loc_cols and 'proposed_NEW_circles_id' in matched_df.columns:
                # Function to check if assigned location matches preferences
                def check_location_match(row):
                    circle_id = row['proposed_NEW_circles_id']
                    if circle_id == 'UNMATCHED' or pd.isna(circle_id):
                        return None
                    
                    # Get the assigned circle's location
                    circle_info = circles_df[circles_df['circle_id'] == circle_id]
                    if circle_info.empty or 'meeting_location' not in circle_info.columns:
                        return None
                    
                    assigned_loc = circle_info.iloc[0]['meeting_location']
                    if pd.isna(assigned_loc):
                        return None
                    
                    # Check each preference
                    for i, col in enumerate(loc_cols):
                        if col in row and pd.notna(row[col]):
                            preferred_loc = row[col]
                            # Very simple check - just look for exact match or substring
                            if preferred_loc == assigned_loc or preferred_loc in assigned_loc or assigned_loc in preferred_loc:
                                return i+1  # Return the preference number (1, 2, 3)
                    
                    return 0  # No match
                
                # Apply the function
                matched_df['location_preference_match'] = matched_df.apply(check_location_match, axis=1)
                
                # Count by match result
                loc_match_counts = matched_df['location_preference_match'].value_counts().sort_index()
                
                # Create a DataFrame for plotting
                match_df = pd.DataFrame({
                    'Preference Match': [match_labels.get(idx, str(idx)) for idx in loc_match_counts.index],
                    'Count': loc_match_counts.values
                })
                
                # Create a pie chart
                fig = px.pie(
                    match_df,
                    names='Preference Match',
                    values='Count',
                    title='Location Preference Satisfaction',
                    color_discrete_sequence=px.colors.qualitative.D3
                )
                
                # Format
                fig.update_traces(textposition='inside', textinfo='percent+label')
                
                # Plot
                st.plotly_chart(fig, use_container_width=True, key="plot_2360")
                
                # Calculate the percentage of participants who got one of their preferences
                matched_count = sum(loc_match_counts.get(i, 0) for i in [1, 2, 3])
                total_count = sum(loc_match_counts.values)
                match_pct = (matched_count / total_count) * 100 if total_count > 0 else 0
                
                st.metric("Participants receiving a preferred location", 
                         f"{matched_count} out of {total_count} ({match_pct:.1f}%)")
            else:
                st.warning("Location preference data not available.")
        else:
            st.warning("Preference satisfaction analysis requires matched circle data.")
            
def render_racial_identity_diversity_histogram():
    """
    Create a histogram showing the distribution of circles based on 
    the number of different racial identity categories they contain
    """
    # First check if we have the necessary data
    if 'matched_circles' not in st.session_state or st.session_state.matched_circles is None:
        st.warning("No matched circles data available to analyze racial identity diversity.")
        return
        
    # Check if the matched_circles dataframe is empty
    if hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty:
        st.warning("No matched circles data available (empty dataframe).")
        return
        
    if 'results' not in st.session_state or st.session_state.results is None:
        st.warning("No results data available to analyze racial identity diversity.")
        return
    
    # Get the circle data
    circles_df = st.session_state.matched_circles.copy()
    results_df = st.session_state.results.copy()
    
    # Filter out circles with no members
    if 'member_count' not in circles_df.columns:
        st.warning("Circles data does not have member_count column")
        return
    
    circles_df = circles_df[circles_df['member_count'] > 0]
    
    if len(circles_df) == 0:
        st.warning("No circles with members available for analysis.")
        return
    
    # Dictionary to track unique racial identities per circle
    circle_racial_identity_counts = {}
    circle_racial_identity_diversity_scores = {}
    
    # Get racial identity data for each member of each circle
    for _, circle_row in circles_df.iterrows():
        circle_id = circle_row['circle_id']
        
        # Initialize empty set to track unique racial identities
        unique_racial_identities = set()
        
        # Get the list of members for this circle
        if 'members' in circle_row and circle_row['members']:
            # For list representation
            if isinstance(circle_row['members'], list):
                member_ids = circle_row['members']
            # For string representation - convert to list
            elif isinstance(circle_row['members'], str):
                try:
                    if circle_row['members'].startswith('['):
                        member_ids = eval(circle_row['members'])
                    else:
                        member_ids = [circle_row['members']]
                except Exception as e:
                    # Skip if parsing fails
                    continue
            else:
                # Skip if members data is not in expected format
                continue
                
            # For each member, look up their racial identity category in results_df
            for member_id in member_ids:
                member_data = results_df[results_df['Encoded ID'] == member_id]
                
                if not member_data.empty and 'Racial_Identity_Category' in member_data.columns:
                    racial_identity = member_data['Racial_Identity_Category'].iloc[0]
                    if pd.notna(racial_identity):
                        unique_racial_identities.add(racial_identity)
        
        # Store the count of unique racial identities for this circle
        if unique_racial_identities:  # Only include if there's at least one valid category
            count = len(unique_racial_identities)
            circle_racial_identity_counts[circle_id] = count
            # Calculate diversity score: 1 point if everyone is in the same category,
            # 2 points if two categories, 3 points if three categories
            circle_racial_identity_diversity_scores[circle_id] = count
    
    # Create histogram data from the racial identity counts
    if not circle_racial_identity_counts:
        st.warning("No racial identity data available for circles.")
        return
        
    # Count circles by number of unique racial identity categories
    diversity_counts = pd.Series(circle_racial_identity_counts).value_counts().sort_index()
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Number of Racial Identity Categories': diversity_counts.index,
        'Number of Circles': diversity_counts.values
    })
    
    st.subheader("Racial Identity Diversity Within Circles")
    
    # Create histogram using plotly with Stanford cardinal red color
    fig = px.bar(
        plot_df,
        x='Number of Racial Identity Categories',
        y='Number of Circles',
        title='Distribution of Circles by Number of Racial Identity Categories',
        text='Number of Circles',  # Display count values on bars
        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
    )
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis=dict(
            title="Number of Different Racial Identity Categories",
            tickmode='linear',
            dtick=1,  # Force integer labels
            range=[0.5, 3.5]  # Since we have 3 categories max
        ),
        yaxis_title="Number of Circles"
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True, key="racial_diversity_histogram")
    
    # Show a table with the data
    st.caption("Data table:")
    st.dataframe(plot_df, hide_index=True)
    
    # Display the Racial Identity diversity score
    st.subheader("Racial Identity Diversity Score")
    st.write("""
    For each circle, the racial identity diversity score is calculated as follows:
    - 1 point: All members in the same racial identity category
    - 2 points: Members from two different racial identity categories
    - 3 points: Members from all three racial identity categories
    """)
    
    # Calculate average and total diversity scores
    total_diversity_score = sum(circle_racial_identity_diversity_scores.values()) if circle_racial_identity_diversity_scores else 0
    avg_diversity_score = total_diversity_score / len(circle_racial_identity_diversity_scores) if circle_racial_identity_diversity_scores else 0
    
    # Create two columns for the metrics
    col1, col2 = st.columns(2)
    
    # Show average and total scores
    with col1:
        st.metric("Average Racial Identity Diversity Score", f"{avg_diversity_score:.2f}")
    with col2:
        st.metric("Total Racial Identity Diversity Score", f"{total_diversity_score}")
    
    # Add a brief explanation
    total_circles = sum(diversity_counts.values)
    diverse_circles = sum(diversity_counts[diversity_counts.index > 1].values)
    diverse_pct = (diverse_circles / total_circles * 100) if total_circles > 0 else 0
    
    st.write(f"Out of {total_circles} total circles, {diverse_circles} ({diverse_pct:.1f}%) contain members from multiple racial identity categories.")

def render_racial_identity_analysis(data):
    """Render the Racial Identity analysis visualizations"""
    st.subheader("Racial Identity Analysis")
    
    # Check if data is None
    if data is None:
        st.warning("No data available for analysis. Please upload participant data.")
        return
    
    # Create a copy to work with
    df = data.copy()
    
    # Debug: show column names
    st.caption("Debugging information:")
    with st.expander("Show data columns and sample values"):
        st.write("Available columns:")
        for col in df.columns:
            st.text(f"- {col}")
        
        # Try to find Racial Identity column
        racial_identity_col = None
        for col in df.columns:
            if "racial identity" in col.lower():
                racial_identity_col = col
                break
        
        if racial_identity_col:
            st.write(f"Found Racial Identity column: {racial_identity_col}")
            # Show some sample values
            sample_values = df[racial_identity_col].dropna().head(5).tolist()
            st.write(f"Sample values: {sample_values}")
        else:
            st.write("No Racial Identity column found in the data")
    
    # Check if we need to create Racial Identity Category
    if 'Racial_Identity_Category' not in df.columns:
        if racial_identity_col:
            st.info(f"Creating Racial Identity Category from {racial_identity_col}...")
            
            # Define function to categorize racial identity
            def categorize_racial_identity(identity):
                if pd.isna(identity):
                    return None
                
                # Convert to string in case it's not
                identity_str = str(identity)
                
                # Apply categorization rules
                if identity_str.startswith("White"):
                    return "White"
                elif "Asian" in identity_str:
                    return "Asian"
                else:
                    return "All Else"
            
            # Apply the categorization function
            df['Racial_Identity_Category'] = df[racial_identity_col].apply(categorize_racial_identity)
            
            # Update session state with the new Racial_Identity_Category
            if 'results' in st.session_state and st.session_state.results is not None:
                # Copy the newly created Racial_Identity_Category to the results DataFrame
                # First, create a dictionary mapping Encoded ID to Racial_Identity_Category
                race_cat_mapping = dict(zip(df['Encoded ID'], df['Racial_Identity_Category']))
                
                # Then apply this mapping to the results DataFrame
                if 'Encoded ID' in st.session_state.results.columns:
                    st.session_state.results['Racial_Identity_Category'] = st.session_state.results['Encoded ID'].map(race_cat_mapping)
                    st.info("Updated results data with Racial Identity Categories")
        else:
            st.warning("Racial Identity data is not available. Please ensure Racial Identity data was included in the uploaded file.")
            return
    
    # Filter out rows with missing Racial Identity Category
    df = df[df['Racial_Identity_Category'].notna()]
    
    if len(df) == 0:
        st.warning("No Racial Identity Category data is available after filtering.")
        return
    
    # Define the proper order for Racial Identity Categories
    racial_identity_order = ["White", "Asian", "All Else"]
    
    # FIRST: Display diversity within circles IF we have matched circles
    if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
        if not (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty):
            render_racial_identity_diversity_histogram()
        else:
            st.info("Run the matching algorithm to see the Racial Identity diversity within circles.")
    else:
        st.info("Run the matching algorithm to see the Racial Identity diversity within circles.")
    
    # SECOND: Display Distribution of Racial Identity
    st.subheader("Distribution of Racial Identity")
    
    # Count by Racial Identity Category
    racial_identity_counts = df['Racial_Identity_Category'].value_counts().reindex(racial_identity_order).fillna(0).astype(int)
    
    # Create a DataFrame for plotting
    racial_identity_df = pd.DataFrame({
        'Racial Identity Category': racial_identity_counts.index,
        'Count': racial_identity_counts.values
    })
    
    # Create histogram using plotly with Stanford cardinal red color
    fig = px.bar(
        racial_identity_df,
        x='Racial Identity Category',
        y='Count',
        title='Distribution of Racial Identity Categories',
        text='Count',  # Display count values on bars
        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
    )
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': racial_identity_order},
        xaxis_title="Racial Identity Category",
        yaxis_title="Count of Participants"
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True, key="racial_identity_distribution")
    
    # Create a breakdown by Status if Status column exists
    if 'Status' in df.columns:
        st.subheader("Racial Identity by Status")
        
        # Create a crosstab of Racial Identity Category vs Status
        status_racial_identity = pd.crosstab(
            df['Racial_Identity_Category'], 
            df['Status'],
            rownames=['Racial Identity Category'],
            colnames=['Status']
        ).reindex(racial_identity_order)
        
        # Add a Total column
        status_racial_identity['Total'] = status_racial_identity.sum(axis=1)
        
        # Calculate percentages
        for col in status_racial_identity.columns:
            if col != 'Total':
                status_racial_identity[f'{col} %'] = (status_racial_identity[col] / status_racial_identity['Total'] * 100).round(1)
        
        # Reorder columns to group counts with percentages
        cols = []
        for status in status_racial_identity.columns:
            if status != 'Total' and not status.endswith(' %'):
                cols.append(status)
                cols.append(f'{status} %')
        cols.append('Total')
        
        # Show the table
        st.dataframe(status_racial_identity[cols], use_container_width=True)
        
        # Create a stacked bar chart
        fig = px.bar(
            status_racial_identity.reset_index(),
            x='Racial Identity Category',
            y=[col for col in status_racial_identity.columns if col != 'Total' and not col.endswith(' %')],
            title='Racial Identity Distribution by Status',
            barmode='stack',
            color_discrete_sequence=['#8C1515', '#2E2D29']  # Stanford colors
        )
        
        # Customize layout
        fig.update_layout(
            xaxis={'categoryorder': 'array', 'categoryarray': racial_identity_order},
            xaxis_title="Racial Identity Category",
            yaxis_title="Count of Participants",
            legend_title="Status"
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True, key="racial_identity_status_distribution")