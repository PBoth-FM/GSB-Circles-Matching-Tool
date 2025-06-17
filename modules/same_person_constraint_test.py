"""
Test module for validating the same-person constraint functionality.
This ensures participants with the same base Encoded ID (e.g., 78171198108 and 78171198108A)
are never placed in the same circle.
"""

import pandas as pd
import re


def get_base_encoded_id(encoded_id):
    """
    Extract base Encoded ID by removing alphabetical suffixes like A, B, C
    This is the same function used in the optimizer.
    
    Args:
        encoded_id: The encoded ID which may have alphabetical suffixes
        
    Returns:
        Base encoded ID without alphabetical suffixes
    """
    if not encoded_id:
        return encoded_id
    # Remove trailing alphabetical characters (case insensitive)
    # Match pattern: digits followed by optional alphabetical suffix
    match = re.match(r'^(\d+)[A-Za-z]*$', str(encoded_id))
    if match:
        return match.group(1)
    return str(encoded_id)  # Return as-is if no pattern match


def validate_same_person_constraint(results_df):
    """
    Validate that no circle contains multiple participants with the same base Encoded ID.
    
    Args:
        results_df: DataFrame with optimization results containing 'Encoded ID' and 'proposed_NEW_circles_id'
        
    Returns:
        Dictionary with validation results including any violations found
    """
    if results_df.empty:
        return {
            'valid': True,
            'violations': [],
            'message': 'No results to validate'
        }
    
    # Only check matched participants (not UNMATCHED)
    matched_df = results_df[results_df['proposed_NEW_circles_id'] != 'UNMATCHED'].copy()
    
    if matched_df.empty:
        return {
            'valid': True,
            'violations': [],
            'message': 'No matched participants to validate'
        }
    
    # Add base ID column for analysis
    matched_df['base_encoded_id'] = matched_df['Encoded ID'].apply(get_base_encoded_id)
    
    violations = []
    
    # Group by circle and check for duplicate base IDs within each circle
    for circle_id, circle_group in matched_df.groupby('proposed_NEW_circles_id'):
        base_id_counts = circle_group['base_encoded_id'].value_counts()
        
        # Find base IDs that appear more than once in this circle
        duplicate_base_ids = base_id_counts[base_id_counts > 1]
        
        for base_id, count in duplicate_base_ids.items():
            # Get all participants with this base ID in this circle
            duplicate_participants = circle_group[circle_group['base_encoded_id'] == base_id]['Encoded ID'].tolist()
            
            violations.append({
                'circle_id': circle_id,
                'base_encoded_id': base_id,
                'duplicate_participants': duplicate_participants,
                'count': count
            })
    
    # Prepare summary
    is_valid = len(violations) == 0
    
    if is_valid:
        message = f"✅ Validation passed: No same-person violations found across {matched_df['proposed_NEW_circles_id'].nunique()} circles"
    else:
        message = f"❌ Validation failed: Found {len(violations)} same-person violations"
    
    return {
        'valid': is_valid,
        'violations': violations,
        'message': message,
        'total_circles_checked': matched_df['proposed_NEW_circles_id'].nunique(),
        'total_participants_checked': len(matched_df)
    }


def test_base_encoded_id_extraction():
    """Test the base encoded ID extraction function with various inputs."""
    test_cases = [
        ('78171198108', '78171198108'),
        ('78171198108A', '78171198108'),
        ('78171198108B', '78171198108'),
        ('12345678901AA', '12345678901'),
        ('12345678901xyz', '12345678901'),
        ('12345678901', '12345678901'),
        ('', ''),
        (None, None),
        ('abc123def', 'abc123def'),  # No leading digits, should return as-is
        ('123', '123'),
    ]
    
    results = []
    for input_id, expected in test_cases:
        actual = get_base_encoded_id(input_id)
        passed = actual == expected
        results.append({
            'input': input_id,
            'expected': expected,
            'actual': actual,
            'passed': passed
        })
        
    all_passed = all(r['passed'] for r in results)
    
    return {
        'all_passed': all_passed,
        'test_results': results,
        'message': f"Base ID extraction test: {'✅ PASSED' if all_passed else '❌ FAILED'}"
    }


def create_test_data_with_duplicates():
    """Create sample test data with duplicate base IDs to validate constraint functionality."""
    test_data = [
        {'Encoded ID': '78171198108', 'proposed_NEW_circles_id': 'IP-NYC-01'},
        {'Encoded ID': '78171198108A', 'proposed_NEW_circles_id': 'IP-NYC-02'},  # Different circle - OK
        {'Encoded ID': '78171198108B', 'proposed_NEW_circles_id': 'IP-NYC-01'},  # Same circle as first - VIOLATION
        {'Encoded ID': '12345678901', 'proposed_NEW_circles_id': 'IP-SF-01'},
        {'Encoded ID': '12345678901A', 'proposed_NEW_circles_id': 'IP-SF-02'},   # Different circle - OK
        {'Encoded ID': '99999999999', 'proposed_NEW_circles_id': 'IP-LA-01'},   # No duplicates - OK
        {'Encoded ID': '88888888888', 'proposed_NEW_circles_id': 'UNMATCHED'},  # Unmatched - ignore
        {'Encoded ID': '88888888888A', 'proposed_NEW_circles_id': 'UNMATCHED'}, # Unmatched - ignore
    ]
    
    return pd.DataFrame(test_data)


def run_constraint_validation_tests():
    """Run comprehensive tests for the same-person constraint validation."""
    print("🧪 Running same-person constraint validation tests...")
    
    # Test 1: Base ID extraction
    print("\n📋 Test 1: Base Encoded ID extraction")
    extraction_results = test_base_encoded_id_extraction()
    print(f"  {extraction_results['message']}")
    
    if not extraction_results['all_passed']:
        print("  Failed test cases:")
        for result in extraction_results['test_results']:
            if not result['passed']:
                print(f"    Input: {result['input']} | Expected: {result['expected']} | Got: {result['actual']}")
    
    # Test 2: Validation with test data containing violations
    print("\n📋 Test 2: Constraint validation with violation test data")
    test_df = create_test_data_with_duplicates()
    validation_results = validate_same_person_constraint(test_df)
    
    print(f"  {validation_results['message']}")
    print(f"  Total circles checked: {validation_results['total_circles_checked']}")
    print(f"  Total participants checked: {validation_results['total_participants_checked']}")
    
    if validation_results['violations']:
        print("  Violations found (expected for this test):")
        for violation in validation_results['violations']:
            print(f"    Circle {violation['circle_id']}: Base ID {violation['base_encoded_id']} appears {violation['count']} times")
            print(f"      Participants: {violation['duplicate_participants']}")
    
    # Test 3: Validation with clean test data (no violations)
    print("\n📋 Test 3: Constraint validation with clean test data")
    clean_test_data = [
        {'Encoded ID': '78171198108', 'proposed_NEW_circles_id': 'IP-NYC-01'},
        {'Encoded ID': '78171198108A', 'proposed_NEW_circles_id': 'IP-NYC-02'},  # Different circle - OK
        {'Encoded ID': '12345678901', 'proposed_NEW_circles_id': 'IP-SF-01'},
        {'Encoded ID': '12345678901A', 'proposed_NEW_circles_id': 'IP-SF-02'},   # Different circle - OK
        {'Encoded ID': '99999999999', 'proposed_NEW_circles_id': 'IP-LA-01'},   # No duplicates - OK
    ]
    clean_df = pd.DataFrame(clean_test_data)
    clean_validation = validate_same_person_constraint(clean_df)
    
    print(f"  {clean_validation['message']}")
    
    # Overall test summary
    all_tests_passed = (
        extraction_results['all_passed'] and 
        len(validation_results['violations']) == 1 and  # Expected 1 violation in test data
        clean_validation['valid']  # Clean data should have no violations
    )
    
    print(f"\n🏁 Overall test result: {'✅ ALL TESTS PASSED' if all_tests_passed else '❌ SOME TESTS FAILED'}")
    
    return {
        'all_passed': all_tests_passed,
        'extraction_test': extraction_results,
        'violation_test': validation_results,
        'clean_test': clean_validation
    }


if __name__ == "__main__":
    # Run tests when script is executed directly
    run_constraint_validation_tests()