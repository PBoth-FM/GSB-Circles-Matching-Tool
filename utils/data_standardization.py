import pandas as pd
import re
import logging
from typing import List, Dict, Any, Union

# Configure logging
logger = logging.getLogger('data_standardization')

# Track normalization statistics for debugging
_normalization_stats = {
    'host_status': {
        'total': 0,
        'normalized_to_always': 0,
        'normalized_to_sometimes': 0,
        'normalized_to_never': 0,
        'already_standardized': 0,
        'examples': {}
    },
    'member_lists': {
        'total': 0,
        'from_string': 0,
        'from_list': 0,
        'from_nan': 0,
        'from_other': 0,
        'examples': {}
    },
    'encoded_ids': {
        'total': 0,
        'from_string': 0,
        'from_numeric': 0,
        'from_nan': 0,
        'examples': {}
    }
}

def normalize_host_status(host_value: Any) -> str:
    """
    Normalize host status to standard values: ALWAYS, SOMETIMES, or NEVER
    
    Args:
        host_value: The host value to normalize (could be string, boolean, number, or None)
        
    Returns:
        Standardized host status: "ALWAYS", "SOMETIMES", or "NEVER"
    """
    # Track statistics
    _normalization_stats['host_status']['total'] += 1
    
    # Handle None/NaN values
    if pd.isna(host_value) or host_value is None or host_value == '':
        _normalization_stats['host_status']['normalized_to_never'] += 1
        _track_example('host_status', host_value, 'NEVER', 'null_or_empty')
        return "NEVER"
    
    # Handle boolean values
    if isinstance(host_value, bool):
        if host_value:
            _normalization_stats['host_status']['normalized_to_always'] += 1
            _track_example('host_status', host_value, 'ALWAYS', 'boolean_true')
            return "ALWAYS"
        else:
            _normalization_stats['host_status']['normalized_to_never'] += 1
            _track_example('host_status', host_value, 'NEVER', 'boolean_false')
            return "NEVER"
    
    # Handle numeric values
    if isinstance(host_value, (int, float)):
        if host_value == 1:
            _normalization_stats['host_status']['normalized_to_always'] += 1
            _track_example('host_status', host_value, 'ALWAYS', 'numeric_one')
            return "ALWAYS"
        else:
            _normalization_stats['host_status']['normalized_to_never'] += 1
            _track_example('host_status', host_value, 'NEVER', 'numeric_zero')
            return "NEVER"
    
    # Now we know it's a string or convertible to string
    host_str = str(host_value).strip().lower()
    
    # Check if already in standard format
    if host_str in ['always', 'sometimes', 'never']:
        standardized = host_str.upper()
        _normalization_stats['host_status']['already_standardized'] += 1
        return standardized
    
    # Match patterns for "always" hosts
    always_patterns = ['always', 'yes', 'true', 'definitely', 'certainly', 'happy to', '1', 'willing']
    for pattern in always_patterns:
        if pattern in host_str or host_str == pattern:
            _normalization_stats['host_status']['normalized_to_always'] += 1
            _track_example('host_status', host_value, 'ALWAYS', f'pattern_{pattern}')
            return "ALWAYS"
    
    # Match patterns for "sometimes" hosts
    sometimes_patterns = ['sometimes', 'maybe', 'possibly', 'occasionally', 'can', 'could', 
                         'willing to but', 'able to but', 'if needed', 'if necessary']
    for pattern in sometimes_patterns:
        if pattern in host_str:
            _normalization_stats['host_status']['normalized_to_sometimes'] += 1
            _track_example('host_status', host_value, 'SOMETIMES', f'pattern_{pattern}')
            return "SOMETIMES"
    
    # Match patterns for "never" hosts
    never_patterns = ['never', 'no', 'not', 'cannot', "can't", 'unable', 'impossible', '0', 'don\'t']
    for pattern in never_patterns:
        if pattern in host_str:
            _normalization_stats['host_status']['normalized_to_never'] += 1
            _track_example('host_status', host_value, 'NEVER', f'pattern_{pattern}')
            return "NEVER"
    
    # Default to NEVER for unmatched patterns
    _normalization_stats['host_status']['normalized_to_never'] += 1
    _track_example('host_status', host_value, 'NEVER', 'default_unmatched')
    return "NEVER"

def normalize_member_list(value: Any) -> List[str]:
    """
    Normalize member list to a consistent List[str] format
    
    Args:
        value: The member list value to normalize (could be list, string, None, etc.)
        
    Returns:
        List of member IDs as strings
    """
    # Track statistics
    _normalization_stats['member_lists']['total'] += 1
    
    # Handle None/NaN values
    if pd.isna(value) or value is None:
        _normalization_stats['member_lists']['from_nan'] += 1
        _track_example('member_lists', value, [], 'null_or_none')
        return []
    
    # Handle already-list values
    if isinstance(value, list):
        _normalization_stats['member_lists']['from_list'] += 1
        
        # Ensure all elements are strings and handle potential None/NaN values within the list
        normalized_list = []
        for item in value:
            if pd.isna(item) or item is None:
                continue  # Skip None/NaN values
            
            # Convert to string
            normalized_list.append(str(item).strip())
        
        _track_example('member_lists', value, normalized_list, 'from_list')
        return normalized_list
    
    # Handle string values - potentially representing a list
    if isinstance(value, str):
        _normalization_stats['member_lists']['from_string'] += 1
        
        # Check if it's a string representation of a list
        if value.strip().startswith('[') and value.strip().endswith(']'):
            try:
                # Try to parse as Python literal
                import ast
                parsed_list = ast.literal_eval(value)
                
                # Ensure it's actually a list
                if isinstance(parsed_list, list):
                    # Recursively normalize to handle nested structures
                    _track_example('member_lists', value, parsed_list, 'string_as_list')
                    return normalize_member_list(parsed_list)
            except (ValueError, SyntaxError) as e:
                # If parsing fails, treat as a single string item
                logger.warning(f"Failed to parse member list string: {e}")
        
        # If we get here, treat as a single string value
        if value.strip():  # Only add non-empty strings
            _track_example('member_lists', value, [value.strip()], 'string_as_item')
            return [value.strip()]
        else:
            _track_example('member_lists', value, [], 'empty_string')
            return []
    
    # Handle other types (convert to string and treat as single item)
    _normalization_stats['member_lists']['from_other'] += 1
    
    try:
        # Try to convert to string and use as single item
        str_value = str(value).strip()
        if str_value:
            _track_example('member_lists', value, [str_value], f'from_{type(value).__name__}')
            return [str_value]
    except Exception as e:
        logger.warning(f"Failed to convert member list value to string: {e}")
    
    # If all else fails, return empty list
    _track_example('member_lists', value, [], 'conversion_failed')
    return []

def normalize_encoded_id(value: Any) -> str:
    """
    Normalize encoded ID to string format
    
    Args:
        value: The encoded ID value to normalize
        
    Returns:
        Encoded ID as string
    """
    # Track statistics
    _normalization_stats['encoded_ids']['total'] += 1
    
    # Handle None/NaN values
    if pd.isna(value) or value is None:
        _normalization_stats['encoded_ids']['from_nan'] += 1
        _track_example('encoded_ids', value, '', 'null_or_none')
        return ""
    
    # Handle string values
    if isinstance(value, str):
        _normalization_stats['encoded_ids']['from_string'] += 1
        _track_example('encoded_ids', value, value.strip(), 'from_string')
        return value.strip()
    
    # Handle numeric values
    if isinstance(value, (int, float)):
        _normalization_stats['encoded_ids']['from_numeric'] += 1
        
        # Convert to string, being careful with floating point
        if isinstance(value, float) and value.is_integer():
            # Convert to int first to remove decimal point
            str_value = str(int(value))
        else:
            str_value = str(value)
        
        _track_example('encoded_ids', value, str_value, 'from_numeric')
        return str_value
    
    # For any other type, convert to string
    try:
        str_value = str(value).strip()
        _track_example('encoded_ids', value, str_value, f'from_{type(value).__name__}')
        return str_value
    except Exception as e:
        logger.warning(f"Failed to convert encoded ID to string: {e}")
        return ""

def _track_example(category: str, original: Any, normalized: Any, rule: str) -> None:
    """
    Track example conversions for debugging
    
    Args:
        category: The normalization category (host_status, member_lists, encoded_ids)
        original: The original value
        normalized: The normalized value
        rule: The rule that triggered this normalization
    """
    examples = _normalization_stats[category]['examples']
    
    # Create rule entry if it doesn't exist
    if rule not in examples:
        examples[rule] = []
    
    # Add example if we don't have too many already
    if len(examples[rule]) < 5:  # Limit to 5 examples per rule
        examples[rule].append({
            'original': original,
            'original_type': type(original).__name__,
            'normalized': normalized
        })

def get_normalization_stats() -> Dict[str, Any]:
    """
    Get current normalization statistics
    
    Returns:
        Dictionary with normalization statistics
    """
    return _normalization_stats

def reset_normalization_stats() -> None:
    """
    Reset normalization statistics
    """
    global _normalization_stats
    _normalization_stats = {
        'host_status': {
            'total': 0,
            'normalized_to_always': 0,
            'normalized_to_sometimes': 0,
            'normalized_to_never': 0,
            'already_standardized': 0,
            'examples': {}
        },
        'member_lists': {
            'total': 0,
            'from_string': 0,
            'from_list': 0,
            'from_nan': 0,
            'from_other': 0,
            'examples': {}
        },
        'encoded_ids': {
            'total': 0,
            'from_string': 0,
            'from_numeric': 0,
            'from_nan': 0,
            'examples': {}
        }
    }

def print_normalization_logs() -> None:
    """
    Print detailed normalization logs
    """
    stats = get_normalization_stats()
    
    print("\n====== DATA STANDARDIZATION LOGS ======")
    
    # Host status logs
    host_stats = stats['host_status']
    print(f"\nHOST STATUS NORMALIZATION ({host_stats['total']} total):")
    print(f"  - Normalized to ALWAYS:    {host_stats['normalized_to_always']}")
    print(f"  - Normalized to SOMETIMES: {host_stats['normalized_to_sometimes']}")
    print(f"  - Normalized to NEVER:     {host_stats['normalized_to_never']}")
    print(f"  - Already standardized:    {host_stats['already_standardized']}")
    
    # Print examples for host status
    print("\n  SAMPLE CONVERSIONS:")
    for rule, examples in host_stats['examples'].items():
        print(f"  Rule: {rule}")
        for ex in examples:
            print(f"    '{ex['original']}' ({ex['original_type']}) → '{ex['normalized']}'")
    
    # Member lists logs
    member_stats = stats['member_lists']
    print(f"\nMEMBER LIST NORMALIZATION ({member_stats['total']} total):")
    print(f"  - From string:  {member_stats['from_string']}")
    print(f"  - From list:    {member_stats['from_list']}")
    print(f"  - From NaN:     {member_stats['from_nan']}")
    print(f"  - From other:   {member_stats['from_other']}")
    
    # Encoded IDs logs
    id_stats = stats['encoded_ids']
    print(f"\nENCODED ID NORMALIZATION ({id_stats['total']} total):")
    print(f"  - From string:  {id_stats['from_string']}")
    print(f"  - From numeric: {id_stats['from_numeric']}")
    print(f"  - From NaN:     {id_stats['from_nan']}")
    
    print("\n========================================")
