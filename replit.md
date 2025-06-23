# GSB Circles Matching Tool

## Overview

The GSB Circles Matching Tool is a Streamlit-based web application designed to optimize participant matching for Stanford Graduate School of Business alumni circles. The application uses linear programming to assign participants to circles based on preferences, constraints, and optimization criteria while maintaining existing circle continuity and demographic diversity.

## System Architecture

### Layered Architecture
The application follows a **modular, multi-layered architecture**:

- **Presentation Layer**: Streamlit UI (`app.py`, `modules/ui_components.py`)
- **Business Logic Layer**: Optimization algorithms, co-leader assignment, data processing
- **Data Access Layer**: File loaders, validators, and normalization utilities  
- **Infrastructure Layer**: Feature flags, metadata management, debugging tools

### Design Patterns
- **Strategy Pattern**: Multiple optimization strategies in `modules/optimizer.py` and `modules/optimizer_new.py`
- **Factory Pattern**: Dynamic circle creation and reconstruction
- **Observer Pattern**: Session state management for real-time UI updates
- **Command Pattern**: Configurable optimization parameters via feature flags

## Key Components

### Core Modules
- `app.py` - Main Streamlit application entry point with session state management
- `modules/data_loader.py` - CSV processing, validation, and deduplication
- `modules/data_processor.py` - Data cleaning, normalization, and standardization
- `modules/optimizer.py` - Primary optimization engine using PuLP linear programming
- `modules/optimizer_new.py` - Enhanced optimization with ID-based matching
- `modules/co_leader_assignment.py` - Business rule implementation for leadership roles
- `modules/ui_components.py` - Streamlit interface components and visualizations

### Utilities
- `utils/feature_flags.py` - Configuration management system
- `utils/normalization.py` - Geographic data standardization
- `utils/validators.py` - Data integrity validation
- `utils/circle_metadata_manager.py` - Centralized metadata handling

### Optimization Engine
- **Technology**: PuLP (Python Linear Programming)
- **Problem Type**: Binary Integer Linear Programming (BILP)
- **Decision Variables**: Binary assignment variables for participant-circle pairs
- **Objective Function**: Multi-component scoring including base assignment, circle size incentives, preference matching, and diversity bonuses

## Data Flow

1. **Data Ingestion**: CSV upload → validation → deduplication using alphabetical suffixes
2. **Data Processing**: Column mapping → geographic normalization → standardization
3. **Optimization**: Constraint-based matching using linear programming with multiple objectives
4. **Post-processing**: Co-leader assignment → circle reconstruction → results validation
5. **Output Generation**: Interactive visualizations → CSV export → demographic analysis

## External Dependencies

### Python Packages
- `streamlit` - Web application framework
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `pulp` - Linear programming optimization
- `plotly` - Interactive visualizations

### Data Sources
- CSV participant data with required columns: `Encoded ID`, `Status`
- Normalization tables: Region/subregion mappings in `attached_assets/`
- Circle metadata for continuing participants

## Deployment Strategy

### Replit Configuration
- **Runtime**: Python 3.11 with Nix package management
- **Deployment Target**: Autoscale
- **Port Configuration**: Internal 5000 → External 80
- **Startup Command**: `streamlit run app.py --server.port 5000`

### Environment Setup
- Streamlit server configured for headless operation
- CORS enabled for external access
- Custom theme with GSB branding colors
- Session state persistence for multi-step workflows

## Changelog

```
Changelog:
- June 23, 2025. Initial setup
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```