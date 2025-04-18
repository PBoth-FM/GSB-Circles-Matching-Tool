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