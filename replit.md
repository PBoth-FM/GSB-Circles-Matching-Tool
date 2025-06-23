# replit.md

## Overview

The GSB Circles Matching Tool is a sophisticated web application built with Streamlit that optimizes the assignment of Stanford Graduate School of Business alumni to discussion circles. The application uses linear programming optimization to match participants based on their geographic preferences, time availability, and other criteria while respecting business constraints around circle sizes and continuing member assignments.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web interface
- **UI Components**: Modular design with dedicated components for different tabs (Match, Debug, Demographics)
- **Data Visualization**: Plotly charts for demographics analysis and results visualization
- **Session Management**: Streamlit session state for maintaining application state across interactions

### Backend Architecture
- **Optimization Engine**: PuLP library for linear programming optimization
- **Data Processing Pipeline**: Multi-stage data validation, normalization, and preprocessing
- **Modular Design**: Separate modules for data loading, processing, optimization, and UI components
- **Feature Flag System**: Configurable features for enabling/disabling functionality

## Key Components

### Data Processing Layer
- **Data Loader** (`modules/data_loader.py`): CSV file processing with deduplication and validation
- **Data Processor** (`modules/data_processor.py`): Data cleaning, normalization, and demographic categorization
- **Validators** (`utils/validators.py`): Data integrity checks and column validation

### Optimization Engine
- **Primary Optimizer** (`modules/optimizer.py`): Core matching algorithm using linear programming
- **Enhanced Optimizer** (`modules/optimizer_new.py`): Advanced optimization with ID-based matching
- **Co-Leader Assignment** (`modules/co_leader_assignment.py`): Business logic for assigning circle leaders

### Utility Layer
- **Normalization** (`utils/normalization.py`): Geographic region and subregion standardization
- **Region Mapping** (`utils/region_mapper.py`): Region code extraction and mapping
- **Feature Flags** (`utils/feature_flags.py`): Configuration management system
- **Metadata Management** (`utils/circle_metadata_manager.py`): Centralized circle data handling

## Data Flow

1. **Data Ingestion**: CSV file upload through Streamlit interface
2. **Validation**: Required column checks and data type validation
3. **Preprocessing**: Data cleaning, normalization, and deduplication
4. **Optimization**: Constraint-based matching using linear programming
5. **Post-processing**: Circle reconstruction and co-leader assignment
6. **Results Display**: Interactive tables and visualizations of matching results

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **PuLP**: Linear programming optimization library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive data visualization

### Data Sources
- CSV files containing participant data and preferences
- Normalization tables for geographic regions and subregions
- Circle metadata and existing member information

## Deployment Strategy

### Platform Configuration
- **Deployment Target**: Autoscale on Replit
- **Runtime**: Python 3.11 with Streamlit server
- **Port Configuration**: Internal port 5000, external port 80
- **Nix Environment**: Stable channel with glibc locales

### Application Startup
- Main entry point: `app.py`
- Server command: `streamlit run app.py --server.port 5000`
- Configuration: Custom Streamlit config with CORS enabled

## Changelog

- June 23, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.