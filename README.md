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
