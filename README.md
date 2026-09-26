# Data-Pilot

**AI-powered dataset health analyzer and preprocessing agent for detecting data-quality issues and preparing datasets for machine learning.**

## 📌 Project Overview

Data-Pilot is a collaborative project focused on analyzing dataset quality, identifying common data-quality issues, and preparing datasets for machine-learning workflows.

The project is being developed as a modular system covering dataset profiling, quality checks, preprocessing, validation, structured reporting, and AI-agent integration.

## 🎯 Objectives

- Analyze the overall health and structure of datasets
- Detect common data-quality issues
- Profile columns and datasets automatically
- Identify potentially problematic values and relationships
- Provide preprocessing capabilities for machine-learning datasets
- Generate structured quality information for further processing
- Integrate dataset analysis with an AI-powered workflow

## 🔍 Dataset Health Analysis

The project currently covers the following dataset analysis areas:

- Dataset statistics
- Column profiling
- Dataset profiling
- Missing-value detection
- Duplicate detection
- Data-type analysis
- Outlier detection
- Categorical-data analysis
- Invalid-value detection
- Cardinality analysis
- Constant-column detection
- Correlation analysis
- ID-column detection
- Data-leakage detection

## ✅ Development Progress

### Completed

- [x] Project setup and repository structure
- [x] Dataset statistics
- [x] Column profiling
- [x] Dataset profiling
- [x] Sample CSV and Excel datasets
- [x] Missing-value detection
- [x] Duplicate detection
- [x] Data-type analysis
- [x] Outlier detection
- [x] Categorical-data analysis
- [x] Invalid-value detection
- [x] Cardinality analysis
- [x] Constant-column detection
- [x] Correlation analysis
- [x] ID-column detection

### Currently Being Tested

- [ ] Data-leakage detection

### Upcoming

- [ ] Dataset and quality schemas
- [ ] Data preprocessing modules
- [ ] Input and output validators
- [ ] Shared test suite
- [ ] Structured report generation
- [ ] AI-agent integration
- [ ] End-to-end testing
- [ ] Final documentation and project release

## 🧩 Planned Components

### Schemas

Structured schemas for representing dataset information and detected quality issues.

- `dataset_schema.py`
- `quality_schema.py`

### Preprocessing

The project will include preprocessing capabilities for preparing datasets after quality analysis.

- Data cleaning
- Data transformation
- Dataset preparation
- Machine-learning-ready data preparation

### Validators

Validation utilities will be added to ensure consistent inputs and outputs across the project.

- `utils/validators.py`

### Testing

A shared test suite will be developed to validate individual components and their integration.

- `tests/`

### AI-Agent Integration

The structured dataset-quality report will later be connected to the AI agent to support automated interpretation and further dataset-preparation decisions.

## 📂 Project Structure

```text
Data-Pilot/
│
├── profiler/
├── preprocessing/
├── utils/
├── tests/
├── data/
├── README.md
├── requirements.txt
└── .gitignore
