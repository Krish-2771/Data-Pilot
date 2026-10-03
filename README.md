# Data-Pilot
### Developed By
- **Dhruv Pophale** — AI Agent & Application
- **Krish Kothari** — Data Analysis & Preprocessing
  
**AI-powered dataset health analyzer and preprocessing agent for detecting data-quality issues and preparing datasets for machine learning.**

## 📌 Project Overview

  
Data-Pilot is a collaborative project focused on analyzing dataset quality, identifying common data-quality issues, and preparing datasets for machine-learning workflows.

The project is developed as a modular system covering:

- Dataset profiling
- Quality checks
- Preprocessing
- Validation
- Structured reporting
- **AI-agent integration (NVIDIA NIM)**

## 🎯 Objectives

- Analyze the overall health and structure of datasets
- Detect common data-quality issues
- Profile columns and datasets automatically
- Identify potentially problematic values and relationships
- Provide deterministic preprocessing capabilities
- Generate structured quality reports
- Prepare datasets for machine-learning workflows
- Integrate dataset analysis with an AI-powered workflow using NVIDIA NIM

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
- **String quality checks (empty strings, whitespace)**
- **Datetime quality checks (invalid dates)**

## 🧹 Preprocessing

The preprocessing system currently supports:

- Mean, median, mode, and **fill_unknown** missing-value handling
- Missing-row removal
- Duplicate removal
- Outlier removal
- Outlier clipping
- Categorical-value normalization
- **String cleaning (trim_whitespace, empty_strings_to_missing, normalize_categorical_case)**
- **Type conversion (convert_to_numeric, convert_to_datetime, convert_to_categorical)**
- Label encoding
- One-hot encoding
- Standard scaling
- Min-max scaling
- Sequential preprocessing pipelines
- **Post-preprocessing validation**

## 📊 Structured Quality Reports

Quality checks generate structured reports containing:

- Dataset dimensions
- Memory usage
- Column information
- Detected issues
- Issue counts
- Issue percentages
- Severity (error/warning/info)
- Supporting evidence
- **Structured issue categories**

Pydantic schemas are used to provide a consistent data structure between the analysis and AI-agent layers.

## 🛡️ Validation

The project includes validation utilities for:

- DataFrame validation
- Column validation
- Numeric-column validation
- Categorical-column validation
- **Post-preprocessing validation (PASS/WARNING/FAIL)**

## 🤖 AI-Agent Integration (NVIDIA NIM)

Data-Pilot includes an AI-powered recommendation layer that sits on top of the deterministic quality analysis:

- **Model**: nvidia/nemotron-3.5-lightning-30b-a3b (configurable via env vars)
- **Input**: Structured DatasetQualityReportSchema
- **Output**: Structured preprocessing recommendations (action, columns, reason, confidence)
- **Actions**: 19 supported preprocessing actions from the pipeline
- **Validation**: All AI recommendations validated against supported actions and dataset columns
- **Approval**: All actions require explicit user approval before execution
- **No direct DataFrame modification**: AI only recommends, never executes

### NVIDIA NIM Setup

1. Get API key from https://build.nvidia.com/
2. Copy `.env.example` to `.env` and add your key:
   ```bash
   cp .env.example .env
   # Edit .env and add NVIDIA_API_KEY
   ```
3. Test connection:
   ```bash
   python scripts/test_nim_connection.py
   ```

### Usage

```python
from ai import DataPilotAgent
import pandas as pd

# Load data
df = pd.read_csv("data.csv")

# Create agent (reads config from .env)
agent = DataPilotAgent()

# Full pipeline: profile → quality checks → AI recommendations
recommendations = agent.recommend_preprocessing(df)

# Or use existing quality report
from quality_checks.quality_engine import build_dataset_quality_report
report = build_dataset_quality_report(df)
recommendations = agent.analyze_quality_report(report)

# View recommendations
for rec in result.recommendations:
    print(f"Action: {rec.action.value} | Columns: {rec.columns} | Confidence: {rec.confidence}")
    print(f"Reason: {rec.reason}")
```

## 🧪 Testing

The project currently contains **103 automated tests** covering profiling, quality checks, preprocessing, quality-engine integration, schemas, validation, and AI layer.

```text
103 passed (55 original + 48 new)
```

Run tests:
```bash
pytest tests/ -v
```

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
- [x] **String quality checks**
- [x] **Datetime quality checks**
- [x] **New preprocessing actions (7)**
- [x] **Post-preprocessing validation**
- [x] **NVIDIA NIM AI client**
- [x] **AI recommendation engine**
- [x] **DataPilotAgent high-level interface**
- [x] **AI unit tests (mocked)**
- [x] **103 passing tests**

### Upcoming

- [ ] User approval workflow UI
- [ ] Application/UI integration
- [ ] Visualization layer
- [ ] End-to-end integration
- [ ] Final project release

## 🧩 Planned Components

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
├── ai/                      # NEW: AI agent module
│   ├── __init__.py
│   ├── agent.py             # DataPilotAgent high-level interface
│   ├── client.py            # NVIDIA NIM / OpenAI-compatible client
│   ├── recommender.py       # Recommendation engine
│   ├── schemas.py           # Pydantic schemas for AI I/O
│   ├── exceptions.py        # Custom exception hierarchy
│   └── prompts.py           # System prompts
│
├── config/
├── profiler/
├── quality_checks/
├── preprocessing/
├── schemas/
├── utils/
├── tests/
├── scripts/                 # NEW: Utility scripts
│   └── test_nim_connection.py
├── data/
│   ├── sample/
│   ├── uploads/
│   └── processed/
│
├── app.py
├── README.md
├── requirements.txt
├── .env.example             # NEW: Environment template
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
- **OpenAI Python SDK (for NIM)**
- **python-dotenv**

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

### Dhruv Pophale

**AI Agent & Application**

- Application interface
- AI agent (NVIDIA NIM integration)
- Recommendations
- User approval workflow
- Visualizations
- Final reporting
