# AptiConduiteML

Multimodal Machine Learning Framework for Neuro-Visual Impairment Assessment in Simulated Driving.

## Overview

AptiConduiteML is a research-oriented machine learning framework developed within the AptiConduite project at UVSQ / Université Paris-Saclay.

The project aims to support neuro-visual impairment assessment by combining:

- EEG (brain activity)
- ECG (heart activity)
- EDA (electrodermal activity)
- Eye tracking
- Head movement
- Driving simulator behavioral data
- Clinical and questionnaire information

The framework transforms heterogeneous physiological and behavioral signals into interpretable indicators that can support expert-driven assessment and diagnosis.

## Project Status

This repository contains only a **sanitized subset** of the original project.

The complete research codebase cannot be released because it contains:

- Medical and physiological data processing pipelines
- Human-subject experimental data
- Clinical assessment information
- Research partner proprietary components
- Infrastructure related to data collection and storage

To protect participant privacy and comply with research ethics requirements, the full dataset and complete implementation are not publicly available.

## My Contributions

### Machine Learning Pipeline

- Designed end-to-end multimodal ML workflow
- Built feature extraction pipelines for physiological and behavioral data
- Implemented feature selection and dimensionality reduction methods
- Developed classification and evaluation pipelines
- Integrated model explainability techniques

### Signal Processing

- EEG spectral feature extraction
- ECG and HRV analysis
- EDA feature engineering
- Eye-tracking behavioral metrics
- Driving simulator performance indicators

### Explainable AI

- SHAP-based model interpretation
- Prototype-based deviation analysis
- Human-readable feature reporting
- Clinical-oriented result visualization

### Data Engineering

- MongoDB data management tools
- Data export/import synchronization utilities
- Automated experiment pipelines
- Reproducible result generation

## Repository Structure

Current public version includes selected components such as:

- Feature extraction
- Feature selection
- Classification experiments
- Database utilities
- Signal processing modules

The following modules are currently experimental:

- Clustering analysis
- Pattern mining / subgroup discovery
- Prototype deviation analysis

These modules were developed as exploratory research tools and should not be considered validated clinical methods.

## Work in Progress

Some research directions remain incomplete or under evaluation:

- Anomaly detection framework (prototype stage)
- Advanced multimodal representation learning
- Automated report generation
- Retrieval-Augmented Generation (RAG) integration for explainable decision support

Current development efforts are increasingly focused on combining multimodal machine learning with RAG-based knowledge retrieval and explainable AI systems.

## Technology Stack

### Machine Learning

- Python
- Scikit-learn
- SHAP
- Imbalanced-learn

### Data Processing

- Pandas
- NumPy
- SciPy

### Signal Processing

- NeuroKit2
- MNE

### Data Storage

- MongoDB
- PyMongo

### Pattern Discovery

- PySubgroup
- MLXtend

## Research Disclaimer

This repository is intended for research and educational purposes only.

The methods presented here are experimental and are not intended for medical diagnosis or clinical decision-making without appropriate expert validation.

## Future Plans

Potential future releases may include:

- Synthetic demonstration datasets
- Simplified reproducible examples
- Public benchmark pipelines
- RAG-based explainable AI prototypes
- Educational notebooks and tutorials

---

© AptiConduite Research Project

Partial public release for demonstration and research portfolio purposes.
