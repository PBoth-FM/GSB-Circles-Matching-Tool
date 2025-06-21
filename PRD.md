# Product Requirements Document (PRD)
## GSB Circles Matching Tool

### Version: 2.0
### Date: June 21, 2025

---

## Code Review Section

### Architecture Overview and Design Patterns

The GSB Circles Matching Tool is built using a **modular, multi-layered architecture** with the following design patterns:

#### **1. Layered Architecture**
- **Presentation Layer**: Streamlit-based UI components (`app.py`, `modules/ui_components.py`)
- **Business Logic Layer**: Optimization algorithms, co-leader assignment, data processing
- **Data Access Layer**: File loaders, validators, and normalization utilities
- **Infrastructure Layer**: Feature flags, metadata management, debugging tools

#### **2. Module Design Patterns**
- **Strategy Pattern**: Multiple optimization strategies in `modules/optimizer.py` and `modules/optimizer_new.py`
- **Factory Pattern**: Dynamic circle creation and reconstruction in `modules/circle_reconstruction.py`
- **Observer Pattern**: Session state management for real-time UI updates
- **Command Pattern**: Configurable optimization parameters and feature flags

### Key Components and Relationships

#### **Core Modules**
```
app.py (Main Entry Point)
├── modules/
│   ├── data_loader.py          # CSV processing and validation
│   ├── data_processor.py       # Data cleaning and normalization
│   ├── optimizer.py            # Primary optimization engine
│   ├── optimizer_new.py        # Enhanced optimization with ID-based matching
│   ├── co_leader_assignment.py # Business rule implementation
│   ├── circle_reconstruction.py # Circle metadata management
│   └── ui_components.py        # Streamlit interface components
├── utils/
│   ├── feature_flags.py        # Configuration management
│   ├── normalization.py        # Geographic data standardization
│   ├── validators.py           # Data integrity checks
│   └── circle_metadata_manager.py # Centralized metadata handling
```

#### **Data Flow Architecture**
1. **Data Ingestion**: CSV upload → validation → deduplication
2. **Data Processing**: Column mapping → normalization → standardization
3. **Optimization**: Constraint-based matching using PuLP linear programming
4. **Post-Processing**: Co-leader assignment → constraint validation → circle reconstruction
5. **Output Generation**: Results display → demographics analysis → CSV export

### Technical Dependencies and Integrations

#### **Core Dependencies**
- **Streamlit**: Web application framework (v1.x)
- **PuLP**: Linear programming optimization library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive data visualization

#### **External Data Sources**
- **Normalization Tables**: CSV-based geographic standardization
  - `Circles-RegionNormalization.csv`
  - `Circles-SubregionNormalization.csv` 
  - `Circles-RegionSubregionCodeMapping.csv`

### Code Quality Assessment

#### **Strengths**
- **Modular Design**: Well-separated concerns with clear module boundaries
- **Comprehensive Logging**: Extensive debug output for troubleshooting
- **Feature Flag System**: Configurable functionality for gradual rollouts
- **Data Validation**: Multi-layer validation with detailed error reporting
- **Constraint Handling**: Robust business rule enforcement

#### **Areas for Improvement**
- **Error Handling**: Some modules lack comprehensive exception handling
- **Performance**: Large dataset processing could benefit from optimization
- **Testing**: Limited automated test coverage
- **Documentation**: Code comments could be more comprehensive

---

## 1. Executive Summary

### Product Overview
The **GSB Circles Matching Tool** is a sophisticated web-based application designed to optimize the assignment of Stanford Graduate School of Business (GSB) alumni into discussion circles. The system uses advanced constraint-based optimization algorithms to match participants based on geographic preferences, time availability, hosting capabilities, and co-leadership requirements.

### Primary Purpose
- **Automated Circle Formation**: Replace manual assignment processes with algorithmic optimization
- **Preference Satisfaction**: Maximize participant satisfaction through location and time matching
- **Business Rule Enforcement**: Ensure compliance with circle formation policies
- **Continuing Circle Management**: Preserve existing circles while accommodating new members

### Target Users
- **Primary Users**: GSB Alumni Relations staff and circle coordinators
- **Secondary Users**: Circle co-leaders and administrative assistants
- **End Beneficiaries**: GSB alumni participating in discussion circles

### Key Value Propositions
1. **Efficiency**: Reduces manual assignment time from weeks to minutes
2. **Optimization**: Maximizes overall participant satisfaction through algorithmic matching
3. **Scalability**: Handles hundreds of participants across global regions
4. **Transparency**: Provides detailed assignment rationale and demographic analysis
5. **Flexibility**: Configurable parameters for different assignment scenarios

---

## 2. Functional Requirements

### 2.1 Data Management

#### **CSV Data Upload and Processing**
- **File Format Support**: CSV files with standardized column structures
- **Column Mapping**: Automatic detection and mapping of input columns to system fields
- **Data Validation**: Multi-tier validation including:
  - Required field presence validation
  - Data type consistency checks
  - Status value normalization (`CURRENT-CONTINUING`, `NEW`, `MOVING OUT`, `WAITLIST`)
  - Geographic data standardization

#### **Deduplication and Data Cleaning**
```python
# Implementation in modules/data_loader.py
def deduplicate_encoded_ids(df):
    # Adds alphabetical suffixes (A, B, C) to duplicate Encoded IDs
    # Returns: (clean_df, deduplication_messages)
```

#### **Data Normalization Pipeline**
- **Geographic Normalization**: Region and subregion standardization using lookup tables
- **Time Preference Standardization**: Converts various time formats to canonical form
- **Class Vintage Calculation**: Automatic calculation based on GSB graduation year
- **Host Status Processing**: Normalization of hosting preferences

### 2.2 Optimization Algorithm

#### **Core Matching Logic**
The system implements a **two-phase optimization approach**:

**Phase 1: Constraint-Based Linear Programming**
```python
# Implementation in modules/optimizer_new.py
def optimize_region_v2(region_df, region, existing_circles, config):
    # Uses PuLP library for constraint satisfaction
    # Maximizes: (1000 * matched_participants) + preference_scores
```

**Phase 2: Post-Processing and Validation**
- Same-person constraint validation (prevents duplicates in circles)
- Co-leader assignment business rules
- Circle size balancing

#### **Business Constraints Implemented**
1. **Circle Size Constraints**:
   - Minimum: 5 participants per new circle
   - Maximum: Configurable (default 8, up to 10)
   - Existing circles: Respect current membership + max_additions

2. **Geographic Compatibility**:
   - Location preference matching with weighted scoring
   - Cross-region assignments allowed for specific scenarios

3. **Time Compatibility**:
   - Day-of-week matching (Monday-Sunday)
   - Time-of-day matching (Morning, Afternoon, Evening)
   - Range support (e.g., "Monday-Wednesday")

4. **Host Requirements**:
   - Each circle requires hosting capability
   - "Always" hosts or minimum 2 "Sometimes" hosts per circle

5. **Continuing Member Preservation**:
   - CURRENT-CONTINUING members stay in existing circles
   - New members can join existing circles with capacity

### 2.3 Co-Leader Assignment

#### **Business Rules Implementation**
```python
# Implementation in modules/co_leader_assignment.py
def assign_co_leaders(results_df, debug_mode=False):
    """
    Business Rules:
    1. CURRENT-CONTINUING + Current Co-Leader:
       - If "Co-Leader Response: CL in 2025?" = "No" → proposed_NEW_Coleader = "No"
       - Otherwise → proposed_NEW_Coleader = "Yes"
    
    2. Non-CURRENT-CONTINUING participants:
       - If "(Non CLs) Volunteering to Co-Lead?" = "Yes" → proposed_NEW_Coleader = "Yes"
       - Otherwise → proposed_NEW_Coleader = "No"
    
    3. Minimum Co-Leader Requirement:
       - Each circle must have ≥2 co-leaders
       - If <2, set ALL participants as co-leaders
    """
```

### 2.4 User Interface Features

#### **Multi-Tab Navigation**
1. **Match Tab**: Primary workflow interface
   - File upload and validation
   - Configuration parameters
   - Optimization execution
   - Results overview

2. **Demographics Tab**: Post-analysis visualization
   - Circle size distribution
   - Geographic distribution analysis
   - Demographic diversity metrics

3. **Documentation Tab**: User guidance and resources

4. **Debug Tab**: Advanced diagnostics and troubleshooting
   - Feature flag management
   - Metadata validation
   - Constraint violation reporting

#### **Configuration Options**
- **Circle Size Preference**: Balanced, Larger Circles, Smaller Circles
- **Location Match Weight**: 1.0-10.0 (configurable preference weighting)
- **Time Match Weight**: 1.0-10.0 (configurable preference weighting)
- **Maximum Circle Size**: 5-10 participants
- **Debug Mode**: Enhanced logging and diagnostics

### 2.5 Output Generation

#### **Results Export**
- **CSV Download**: Complete assignment results with all participant data
- **Circle Summary**: Metadata for each formed circle
- **Unmatched Report**: Participants who couldn't be assigned with reasons

#### **Analytics and Reporting**
- **Match Rate Statistics**: Overall and demographic-specific success rates
- **Preference Satisfaction Metrics**: Location and time match effectiveness
- **Circle Distribution Analysis**: Size, geographic, and demographic balance

---

## 3. Technical Specifications

### 3.1 System Architecture

#### **Technology Stack**
- **Backend Framework**: Streamlit (Python web application framework)
- **Optimization Engine**: PuLP (Python linear programming library)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly Express
- **Session Management**: Streamlit session state

#### **Deployment Architecture**
```
Web Browser ↔ Streamlit Server (Port 5000) ↔ Python Application
                    ↓
            File System (CSV Storage)
                    ↓
            Session State (In-Memory)
```

#### **Configuration Management**
```python
# Implementation in utils/feature_flags.py
class FeatureFlags:
    - enable_feature_flags_ui: Control debug interface visibility
    - use_standardized_host_status: Normalize host values
    - use_standardized_member_lists: Enforce list format consistency
    - use_optimizer_metadata: Use optimizer-generated metadata
    - enable_metadata_validation: Show validation reports
```

### 3.2 Data Models

#### **Participant Data Schema**
```python
required_columns = [
    'Encoded ID',           # Unique participant identifier
    'Status',              # CURRENT-CONTINUING, NEW, MOVING OUT, WAITLIST
    'Current_Region',      # Geographic region preference
    'Current_Subregion',   # Sub-regional preference
    'Time_Preference',     # Meeting time availability
    'Host_Preference',     # Hosting capability (Always/Sometimes/Never)
    'GSB_Class',          # Graduation year for vintage calculation
]

optional_columns = [
    'Current_Circle_ID',           # For continuing members
    'Co-Leader Response: CL in 2025?',  # Co-leader continuation
    '(Non CLs) Volunteering to Co-Lead?', # New co-leader volunteers
    'Current Co-Leader?',          # Current co-leader status
]
```

#### **Circle Metadata Schema**
```python
circle_metadata = {
    'circle_id': str,          # Unique circle identifier (e.g., 'IP-BOS-01')
    'region': str,             # Geographic region
    'subregion': str,          # Sub-regional classification
    'meeting_time': str,       # Standardized time slot
    'members': List[str],      # List of participant Encoded IDs
    'member_count': int,       # Current member count
    'max_additions': int,      # Maximum new members allowed
    'is_existing': bool,       # True for continuing circles
    'always_hosts': int,       # Count of "Always" hosts
    'sometimes_hosts': int,    # Count of "Sometimes" hosts
    'new_members': int,        # Count of new participants
    'continuing_members': int, # Count of continuing participants
}
```

### 3.3 Algorithm Specifications

#### **Optimization Objective Function**
```
Maximize: 1000 * (total_matched_participants) + Σ(preference_scores)

Where preference_scores = location_weight * location_match + time_weight * time_match
```

#### **Constraint Equations**
1. **Assignment Constraint**: `Σ(x[p,c]) ≤ 1` for each participant p
2. **Capacity Constraint**: `Σ(x[p,c]) ≤ max_capacity[c]` for each circle c
3. **Minimum Size**: `Σ(x[p,c]) ≥ min_size * y[c]` for new circles
4. **Host Requirement**: `always_hosts + (sometimes_hosts ≥ 2) ≥ y[c]` for new circles
5. **Compatibility**: `x[p,c] = 0` if participant p incompatible with circle c

### 3.4 Performance Requirements

#### **Processing Constraints**
- **Dataset Size**: Up to 1,000 participants
- **Optimization Time**: <60 seconds for typical datasets
- **Memory Usage**: <2GB RAM for standard operations
- **File Upload**: CSV files up to 10MB

#### **Scalability Considerations**
- **Geographic Regions**: Optimized per-region to improve performance
- **Incremental Processing**: Support for batch processing of large datasets
- **Caching**: Normalization table caching for improved response times

---

## 4. User Experience Requirements

### 4.1 Interface Design Principles

#### **Streamlit Native Components**
- **Clean, Minimalist Design**: Leverages Streamlit's built-in styling
- **Progressive Disclosure**: Advanced options hidden in expandable sections
- **Real-time Feedback**: Progress indicators and status updates
- **Error Communication**: Clear, actionable error messages

#### **Navigation Structure**
```
Tab 1: Match (Primary Workflow)
├── File Upload Section
├── Data Validation Results
├── Configuration Panel
├── Optimization Controls
└── Results Overview

Tab 2: Demographics (Analysis)
├── Circle Size Distribution
├── Geographic Distribution
└── Diversity Metrics

Tab 3: Documentation
└── User Guide and Resources

Tab 4: Debug (Advanced Users)
├── Feature Flags
├── Metadata Validation
└── Constraint Reports
```

### 4.2 User Journey Mapping

#### **Primary Workflow**
1. **Data Preparation**: Upload CSV file with participant data
2. **Validation**: Review and resolve data quality issues
3. **Configuration**: Set optimization parameters and preferences
4. **Execution**: Run matching algorithm and monitor progress
5. **Review**: Analyze results and demographic distribution
6. **Export**: Download final assignments and circle metadata

#### **Error Recovery Paths**
- **Data Validation Failures**: Clear error messages with resolution guidance
- **Optimization Failures**: Fallback algorithms and manual intervention options
- **Constraint Violations**: Detailed reporting with suggested fixes

### 4.3 Accessibility Considerations

#### **Streamlit Accessibility Features**
- **Keyboard Navigation**: Full keyboard accessibility through Streamlit
- **Screen Reader Support**: Semantic HTML structure
- **Color Contrast**: Default Streamlit theme meets WCAG guidelines
- **Responsive Design**: Automatic mobile and tablet adaptation

---

## 5. Business Logic Documentation

### 5.1 Core Algorithms

#### **Time Compatibility Logic**
```python
# Implementation in modules/data_processor.py
def is_time_compatible(time1, time2, is_important=False):
    """
    Compatibility Rules:
    - Exact matches: "Monday Evening" matches "Monday Evening"
    - Day ranges: "Monday-Wednesday" matches "Tuesday"
    - Time flexibility: "Evening" matches any day + "Evening"
    - Cross-compatibility: Partial overlaps score proportionally
    """
```

#### **Location Preference Scoring**
```python
def calculate_preference_scores(df):
    """
    Location Scoring:
    - Exact subregion match: 3 points
    - Same region, different subregion: 2 points
    - Different region: 1 point
    - No preference specified: 0 points
    """
```

### 5.2 Decision Trees

#### **Circle Assignment Decision Flow**
```
Participant Input
├── Status = "CURRENT-CONTINUING"?
│   ├── Yes → Find existing circle → Force assignment
│   └── No → Continue to optimization
├── Compatible circles available?
│   ├── Yes → Apply preference scoring
│   └── No → Mark as UNMATCHED
└── Optimization result
    ├── Assigned → Apply co-leader rules
    └── Unmatched → Generate reason code
```

#### **Co-Leader Assignment Logic**
```
For each participant in assigned circle:
├── Status = "CURRENT-CONTINUING" AND Current Co-Leader = "Yes"?
│   ├── Yes → Check "CL in 2025?" response
│   │   ├── "No" → proposed_NEW_Coleader = "No"
│   │   └── Other → proposed_NEW_Coleader = "Yes"
│   └── No → Check "Volunteering to Co-Lead?"
│       ├── "Yes" → proposed_NEW_Coleader = "Yes"
│       └── Other → proposed_NEW_Coleader = "No"
├── Count co-leaders in circle
└── If < 2 co-leaders → Set ALL participants as co-leaders
```

### 5.3 Validation Rules

#### **Data Integrity Constraints**
1. **Unique Participant IDs**: Automatic deduplication with suffix generation
2. **Valid Status Values**: Must be in approved list with normalization
3. **Geographic Consistency**: Region/subregion combinations validated against lookup tables
4. **Time Format Standardization**: Automatic parsing and normalization
5. **Same-Person Prevention**: Base ID extraction prevents multiple instances in same circle

#### **Business Rule Validation**
```python
# Implementation in modules/same_person_constraint_test.py
def validate_same_person_constraint(results):
    """
    Ensures no circle contains multiple participants with the same base Encoded ID
    Base ID extraction: "12345A" and "12345B" both have base ID "12345"
    """
```

### 5.4 Calculation Methods

#### **Class Vintage Calculation**
```python
# Implementation in modules/data_processor.py
def calculate_class_vintage(gsb_class):
    """
    Current year: 2025
    Vintage Categories:
    - "01-10 yrs": 2015-2024 graduates
    - "11-20 yrs": 2005-2014 graduates  
    - "21-30 yrs": 1995-2004 graduates
    - "31+ yrs": Pre-1995 graduates
    """
```

#### **Preference Score Aggregation**
```
total_score = (location_weight * location_score) + (time_weight * time_score)
final_objective = 1000 * matched_count + Σ(total_scores)
```

### 5.5 State Management

#### **Session State Variables**
```python
session_state = {
    'df': None,                    # Original uploaded data
    'processed_data': None,        # Cleaned and normalized data
    'results': None,               # Optimization results
    'matched_circles': None,       # Circle metadata
    'unmatched_participants': None, # Unassigned participants
    'config': {                    # Configuration parameters
        'min_circle_size': 5,
        'max_circle_size': 8,
        'existing_circle_handling': 'optimize',
        'optimization_weight_location': 5.0,
        'optimization_weight_time': 5.0,
        'debug_mode': True
    },
    'feature_flags': {},           # Feature toggles
    'circle_manager': None,        # Metadata manager instance
}
```

---

## 6. Integration Requirements

### 6.1 File System Integration

#### **Data Sources**
- **Input Files**: CSV uploads via Streamlit file uploader
- **Normalization Tables**: Static CSV files in `attached_assets/` directory
- **Debug Logs**: JSON files in `debug_data/` directory for troubleshooting

#### **File Processing Pipeline**
1. **Upload Validation**: File type, size, and format checks
2. **Column Detection**: Automatic mapping to expected schema
3. **Data Cleaning**: Normalization and standardization
4. **Export Generation**: CSV and JSON output formats

### 6.2 Third-Party Dependencies

#### **Python Libraries**
```python
dependencies = {
    'streamlit': '^1.0.0',     # Web framework
    'pandas': '^1.5.0',        # Data manipulation
    'numpy': '^1.24.0',        # Numerical computing
    'pulp': '^2.7.0',          # Linear programming
    'plotly': '^5.15.0',       # Data visualization
}
```

#### **Data Exchange Formats**
- **Input**: CSV with UTF-8 encoding
- **Output**: CSV with complete results and metadata
- **Internal**: JSON for session state and debug logs
- **Configuration**: Python dictionaries and feature flags

---

## 7. Testing & Quality Assurance

### 7.1 Current Test Coverage

#### **Manual Testing Scenarios**
- **Happy Path**: Complete workflow with valid data
- **Data Validation**: Various CSV format and content issues
- **Optimization Edge Cases**: Insufficient participants, no compatible circles
- **Business Rule Validation**: Co-leader assignment and constraint checking

#### **Debug and Diagnostics**
```python
# Comprehensive logging throughout the application
debug_features = [
    'Circle eligibility tracking',
    'Participant assignment tracing', 
    'Constraint violation reporting',
    'Metadata consistency validation',
    'Performance timing measurements'
]
```

### 7.2 Quality Metrics

#### **Performance Benchmarks**
- **Small Dataset** (50 participants): <5 seconds
- **Medium Dataset** (200 participants): <15 seconds  
- **Large Dataset** (500+ participants): <60 seconds

#### **Accuracy Metrics**
- **Match Rate**: Percentage of participants successfully assigned
- **Preference Satisfaction**: Average location and time preference scores
- **Constraint Compliance**: Zero tolerance for business rule violations
- **Data Integrity**: 100% preservation of participant information

### 7.3 Deployment Considerations

#### **Environment Requirements**
```bash
# Streamlit configuration in .streamlit/config.toml
[server]
headless = true
address = "0.0.0.0" 
port = 5000
```

#### **Operational Monitoring**
- **Application Logs**: Comprehensive debug output for troubleshooting
- **Performance Metrics**: Optimization timing and memory usage
- **Error Tracking**: Detailed exception handling and user feedback
- **Data Validation**: Real-time integrity checking and reporting

---

## 8. Future Enhancement Opportunities

### 8.1 Technical Improvements
- **Automated Testing**: Unit and integration test development
- **Performance Optimization**: Algorithm tuning for larger datasets  
- **Database Integration**: Persistent storage for historical data
- **API Development**: RESTful interface for programmatic access

### 8.2 Functional Enhancements
- **Multi-Round Optimization**: Iterative improvement capabilities
- **Advanced Demographics**: Enhanced diversity and inclusion metrics
- **Scheduling Integration**: Calendar and meeting coordination
- **Feedback Loop**: Post-assignment satisfaction tracking

---

## Appendix

### Contact Information
**Product Owner**: Patricia Bothwell (GSB '05)  
**Email**: pbothwell@fastmail.com

### Related Documentation
- **Deployment Guide**: https://docs.google.com/document/d/1yPtqwDfTzHZ4WIy5fPZbtUJUpXKwsacIWdwi2bCWwmM/edit
- **Detailed Requirements**: https://docs.google.com/document/d/1N0rjRWCO_aKpYxlUXbSXEPEWSMWHUABktOzc4RAejHI/edit

---

*Document Version: 2.0*  
*Last Updated: June 21, 2025*  
*Generated from comprehensive code analysis*