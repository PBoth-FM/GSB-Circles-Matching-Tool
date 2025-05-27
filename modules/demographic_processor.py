
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

def ensure_demographic_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all demographic categories exist in the DataFrame.
    Creates categories from raw data if they don't exist.
    
    Args:
        df: DataFrame containing participant data
        
    Returns:
        DataFrame with all demographic categories ensured
    """
    result_df = df.copy()
    
    # Class Vintage Category
    if 'Class_Vintage' not in result_df.columns:
        if 'GSB Class Year' in result_df.columns:
            result_df['Class_Vintage'] = categorize_class_vintage(result_df['GSB Class Year'])
        else:
            result_df['Class_Vintage'] = None
    
    # Employment Category
    if 'Employment_Category' not in result_df.columns:
        if 'Employment Status' in result_df.columns:
            result_df['Employment_Category'] = categorize_employment_status(result_df['Employment Status'])
        else:
            result_df['Employment_Category'] = None
    
    # Industry Category
    if 'Industry_Category' not in result_df.columns:
        if 'Industry Sector' in result_df.columns:
            result_df['Industry_Category'] = categorize_industry_sector(result_df['Industry Sector'])
        else:
            result_df['Industry_Category'] = None
    
    # Racial Identity Category
    if 'Racial_Identity_Category' not in result_df.columns:
        if 'Racial Identity' in result_df.columns:
            result_df['Racial_Identity_Category'] = categorize_racial_identity(result_df['Racial Identity'])
        else:
            result_df['Racial_Identity_Category'] = None
    
    # Children Category
    if 'Children_Category' not in result_df.columns:
        if 'Children' in result_df.columns:
            result_df['Children_Category'] = categorize_children(result_df['Children'])
        else:
            result_df['Children_Category'] = None
    
    return result_df

def categorize_class_vintage(class_year_series: pd.Series) -> pd.Series:
    """Categorize GSB Class Year into vintage buckets"""
    def get_vintage_category(year):
        if pd.isna(year):
            return None
        
        try:
            year_num = int(float(year))
            current_year = 2024  # Adjust as needed
            years_since = current_year - year_num
            
            if years_since <= 10:
                return "01-10 yrs"
            elif years_since <= 20:
                return "11-20 yrs"
            elif years_since <= 30:
                return "21-30 yrs"
            elif years_since <= 40:
                return "31-40 yrs"
            else:
                return "40+ yrs"
        except (ValueError, TypeError):
            return None
    
    return class_year_series.apply(get_vintage_category)

def categorize_employment_status(employment_series: pd.Series) -> pd.Series:
    """Categorize employment status into standard categories"""
    def get_employment_category(status):
        if pd.isna(status):
            return None
        
        status_str = str(status).lower().strip()
        
        if any(term in status_str for term in ['entrepreneur', 'founder', 'startup', 'self-employed']):
            return "Entrepreneur"
        elif any(term in status_str for term in ['employed', 'full-time', 'part-time', 'working']):
            return "Employed"
        elif any(term in status_str for term in ['student', 'mba', 'school']):
            return "Student"
        else:
            return "Other"
    
    return employment_series.apply(get_employment_category)

def categorize_industry_sector(industry_series: pd.Series) -> pd.Series:
    """Categorize industry sector into standard categories"""
    def get_industry_category(industry):
        if pd.isna(industry):
            return None
        
        industry_str = str(industry).lower().strip()
        
        if any(term in industry_str for term in ['tech', 'software', 'technology', 'it']):
            return "Technology"
        elif any(term in industry_str for term in ['finance', 'banking', 'investment']):
            return "Finance"
        elif any(term in industry_str for term in ['consulting']):
            return "Consulting"
        elif any(term in industry_str for term in ['healthcare', 'medical', 'pharma']):
            return "Healthcare"
        else:
            return "Other"
    
    return industry_series.apply(get_industry_category)

def categorize_racial_identity(racial_identity_series: pd.Series) -> pd.Series:
    """Categorize racial identity into standard categories"""
    def get_racial_identity_category(identity):
        if pd.isna(identity):
            return None
        
        identity_str = str(identity).lower().strip()
        
        if identity_str.startswith("white"):
            return "White"
        elif "asian" in identity_str:
            return "Asian"
        else:
            return "All Else"
    
    return racial_identity_series.apply(get_racial_identity_category)

def categorize_children(children_series: pd.Series) -> pd.Series:
    """Categorize children status into standard categories"""
    def get_children_category(children):
        if pd.isna(children):
            return None
        
        children_str = str(children).lower().strip()
        
        if children_str in ['yes', 'true', '1'] or 'yes' in children_str:
            return "Has Children"
        elif children_str in ['no', 'false', '0'] or 'no' in children_str:
            return "No Children"
        else:
            return "Unknown"
    
    return children_series.apply(get_children_category)

def get_circle_members_from_results(results_df: pd.DataFrame, circle_id: str) -> List[str]:
    """
    Get member IDs for a circle directly from results DataFrame.
    This is the single source of truth approach.
    
    Args:
        results_df: Results DataFrame containing participant data
        circle_id: ID of the circle to get members for
        
    Returns:
        List of member Encoded IDs
    """
    if 'proposed_NEW_circles_id' not in results_df.columns:
        return []
    
    circle_members = results_df[results_df['proposed_NEW_circles_id'] == circle_id]
    if circle_members.empty:
        return []
    
    # Return non-null Encoded IDs
    member_ids = circle_members['Encoded ID'].dropna().tolist()
    return member_ids

def calculate_circle_diversity_from_results(results_df: pd.DataFrame, circle_id: str, 
                                          category_column: str) -> int:
    """
    Calculate diversity score for a circle in a specific category using Results DataFrame.
    
    Args:
        results_df: Results DataFrame containing participant data
        circle_id: ID of the circle
        category_column: Name of the demographic category column
        
    Returns:
        Number of unique categories in this circle (diversity score)
    """
    # Get member IDs for this circle
    member_ids = get_circle_members_from_results(results_df, circle_id)
    
    if not member_ids:
        return 0
    
    # Get member data
    member_data = results_df[results_df['Encoded ID'].isin(member_ids)]
    
    if member_data.empty or category_column not in member_data.columns:
        return 0
    
    # Count unique non-null categories
    unique_categories = member_data[category_column].dropna().unique()
    return len(unique_categories)

def get_all_circles_from_results(results_df: pd.DataFrame) -> List[str]:
    """
    Get all unique circle IDs from Results DataFrame.
    
    Args:
        results_df: Results DataFrame containing participant data
        
    Returns:
        List of unique circle IDs (excluding UNMATCHED)
    """
    if 'proposed_NEW_circles_id' not in results_df.columns:
        return []
    
    # Get unique circle IDs, excluding UNMATCHED and null values
    circle_ids = results_df['proposed_NEW_circles_id'].dropna()
    unique_circles = circle_ids[circle_ids != 'UNMATCHED'].unique().tolist()
    return unique_circles
