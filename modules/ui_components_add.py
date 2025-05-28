# Duplicate function removed - now importing from main ui_components module to avoid duplicates

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
            # Import and call the main histogram function to avoid duplicates
            from modules.ui_components import render_racial_identity_diversity_histogram
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