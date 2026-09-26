# Data-Pilot

**AI-powered dataset health analyzer and preprocessing agent for detecting data-quality issues and preparing datasets for machine learning.**

## 📌 Project Overview

Data-Pilot is a collaborative project focused on analyzing dataset quality, identifying common data-quality issues, and preparing datasets for machine-learning workflows.

The project is developed as a modular system covering:

- Dataset profiling
- Quality checks
- Preprocessing
- Validation
- Structured reporting
- AI-agent integration

## 🎯 Objectives

- Analyze the overall health and structure of datasets
- Detect common data-quality issues
- Profile columns and datasets automatically
- Identify potentially problematic values and relationships
- Provide deterministic preprocessing capabilities
- Generate structured quality reports
- Prepare datasets for machine-learning workflows
- Integrate dataset analysis with an AI-powered workflow

## 🔍 Dataset Health Analysis

The project currently includes:

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

## 🧹 Preprocessing

The preprocessing system currently supports:

- Mean, median, and mode missing-value handling
- Missing-row removal
- Duplicate removal
- Outlier removal
- Outlier clipping
- Categorical-value normalization
- Label encoding
- One-hot encoding
- Standard scaling
- Min-max scaling
- Sequential preprocessing pipelines

## 📊 Structured Quality Reports

Quality checks generate structured reports containing:

- Dataset dimensions
- Memory usage
- Column information
- Detected issues
- Issue counts
- Issue percentages
- Severity
- Supporting evidence

Pydantic schemas are used to provide a consistent data structure between the analysis and AI-agent layers.

## 🛡️ Validation

The project includes validation utilities for:

- DataFrame validation
- Column validation
- Numeric-column validation
- Categorical-column validation

## 🧪 Testing

The project currently contains **55 automated tests** covering profiling, quality checks, preprocessing, quality-engine integration, schemas, and validation.

```text
55 passed

## ✅ Development Progress

### Completed

- [x] Project setup and repository structure
- [x] Dataset statistics
- [x] Column profiling
- [x] Dataset profiling
- [x] CSV and Excel dataset support
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
- [x] Data-leakage detection
- [x] Dataset and quality schemas
- [x] Data preprocessing modules
- [x] Input and output validators
- [x] Structured quality reports
- [x] Preprocessing pipeline
- [x] Automated test suite
- [x] 55 passing tests

### Upcoming

- [ ] AI-agent integration
- [ ] Recommendation workflow
- [ ] User approval workflow
- [ ] Application/UI integration
- [ ] Visualization layer
- [ ] End-to-end integration
- [ ] Final project release

## 🧩 Planned Components

### AI-Agent Integration

The structured dataset-quality report will be connected to the AI agent to support automated interpretation and preprocessing recommendations.

### Application

The application layer will provide:

- Dataset upload
- AI-assisted recommendations
- User approval
- Visualization
- Final reporting

## 📂 Project Structure

```text
Data-Pilot/
│
├── agent/
├── config/
├── profiler/
├── quality_checks/
├── preprocessing/
├── schemas/
├── utils/
├── tests/
├── data/
│   ├── sample/
│   ├── uploads/
│   └── processed/
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore


## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Pydantic
- OpenPyXL
- Pytest

## 👥 Contributors

### Krish Kothari

**Data Analysis & Preprocessing**

- Dataset profiling
- Statistics
- Quality checks
- Structured quality reports
- Preprocessing
- Validation
- Testing

### Dhruv

**AI Agent & Application**

- Application interface
- AI agent
- Recommendations
- User approval workflow
- Visualizations
- Final reporting
