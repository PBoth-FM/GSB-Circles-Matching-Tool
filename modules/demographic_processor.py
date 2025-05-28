
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
        
        # No children category
        if "no children" in children_str or children_str in ['no', 'false', '0']:
            return "No children"
        
        # First time <=5s
        if any(term in children_str for term in ["children 5 and under", "children under 5", "pregnant", "first time <5"]):
            return "First time <5"
        
        # First time 6-18s  
        if any(term in children_str for term in ["children 6 through 17", "children 6-18", "first time 6-18"]) and not any(term in children_str for term in ["adult children", "adult"]):
            return "First time 6-18"
        
        # Adult children / all else
        if any(term in children_str for term in ["adult children", "adult", "all else"]) or children_str in ['yes', 'true', '1'] or 'yes' in children_str:
            return "Adult children / all else"
        
        # Default to adult children / all else for any other children data
        return "Adult children / all else"
    
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

def calculate_all_diversity_scores_from_results(results_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Calculate diversity scores for all circles and all categories using Results DataFrame.
    
    Args:
        results_df: Results DataFrame containing participant data
        
    Returns:
        Dictionary mapping circle IDs to their diversity scores by category
    """
    # Ensure demographic categories exist
    results_df = ensure_demographic_categories(results_df)
    
    # Get all circles
    circle_ids = get_all_circles_from_results(results_df)
    
    # Categories to calculate
    categories = {
        'vintage': 'Class_Vintage',
        'employment': 'Employment_Category', 
        'industry': 'Industry_Category',
        'racial_identity': 'Racial_Identity_Category',
        'children': 'Children_Category'
    }
    
    # Calculate scores for each circle
    diversity_scores = {}
    for circle_id in circle_ids:
        circle_scores = {}
        for category_name, column_name in categories.items():
            score = calculate_circle_diversity_from_results(results_df, circle_id, column_name)
            circle_scores[category_name] = score
        diversity_scores[circle_id] = circle_scores
    
    return diversity_scores

def reconstruct_circle_metadata_from_results(results_df: pd.DataFrame, circle_id: str) -> Dict[str, Any]:
    """
    Reconstruct comprehensive circle metadata from Results DataFrame.
    
    Args:
        results_df: Results DataFrame containing participant data
        circle_id: ID of the circle to reconstruct metadata for
        
    Returns:
        Dictionary containing comprehensive circle metadata
    """
    # Get members for this circle
    member_ids = get_circle_members_from_results(results_df, circle_id)
    
    if not member_ids:
        return {
            'circle_id': circle_id,
            'member_count': 0,
            'members': [],
            'region': 'Unknown',
            'subregion': 'Unknown', 
            'meeting_time': 'Unknown',
            'always_hosts': 0,
            'sometimes_hosts': 0,
            'max_additions': 0
        }
    
    # Get member data
    member_data = results_df[results_df['Encoded ID'].isin(member_ids)]
    
    # Extract metadata from the first member (assuming consistent within circle)
    first_member = member_data.iloc[0]
    
    # Get region and subregion
    region = first_member.get('proposed_NEW_Region', first_member.get('Current_Region', 'Unknown'))
    subregion = first_member.get('proposed_NEW_Subregion', first_member.get('Current_Subregion', 'Unknown'))
    meeting_time = first_member.get('proposed_NEW_DayTime', first_member.get('Current_Meeting_Time', 'Unknown'))
    
    # Calculate host counts using standardized host status
    from utils.data_standardization import normalize_host_status
    
    always_hosts = 0
    sometimes_hosts = 0
    
    # Find host column
    host_col = None
    for col in ['host', 'Host', 'willing_to_host']:
        if col in member_data.columns:
            host_col = col
            break
    
    if host_col:
        for _, member in member_data.iterrows():
            host_status = normalize_host_status(member[host_col])
            if host_status == 'ALWAYS':
                always_hosts += 1
            elif host_status == 'SOMETIMES':
                sometimes_hosts += 1
    
    # Determine max_additions (simplified logic for now)
    max_additions = max(0, 10 - len(member_ids))  # Assume target size of 10
    
    return {
        'circle_id': circle_id,
        'member_count': len(member_ids),
        'members': member_ids,
        'region': region,
        'subregion': subregion,
        'meeting_time': meeting_time,
        'always_hosts': always_hosts,
        'sometimes_hosts': sometimes_hosts,
        'max_additions': max_additions,
        'new_members': 0,  # Will be calculated based on status
        'is_new_circle': False  # Will be determined by circle naming
    }

def create_circles_dataframe_from_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a comprehensive circles DataFrame from Results DataFrame.
    This serves as a complete replacement for matched_circles when it's empty.
    
    Args:
        results_df: Results DataFrame containing participant data
        
    Returns:
        DataFrame with comprehensive circle data
    """
    # Ensure demographic categories exist
    results_df = ensure_demographic_categories(results_df)
    
    # Get all circles
    circle_ids = get_all_circles_from_results(results_df)
    
    if not circle_ids:
        return pd.DataFrame()
    
    # Calculate diversity scores for all circles
    all_diversity_scores = calculate_all_diversity_scores_from_results(results_df)
    
    # Build circle data
    circles_data = []
    for circle_id in circle_ids:
        # Get comprehensive metadata
        metadata = reconstruct_circle_metadata_from_results(results_df, circle_id)
        
        # Add diversity scores
        if circle_id in all_diversity_scores:
            scores = all_diversity_scores[circle_id]
            metadata.update({
                'vintage_score': scores.get('vintage', 0),
                'employment_score': scores.get('employment', 0),
                'industry_score': scores.get('industry', 0),
                'racial_identity_score': scores.get('racial_identity', 0),
                'children_score': scores.get('children', 0)
            })
        
        # Add metadata source
        metadata['metadata_source'] = 'results_dataframe'
        
        circles_data.append(metadata)
    
    return pd.DataFrame(circles_data)
