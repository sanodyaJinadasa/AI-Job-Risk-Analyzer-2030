# AI Automation Risk Analyzer on Jobs 2030

**Predict job automation risk and probability by 2030 using machine learning.**

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Data](#data)  
- [Usage](#usage)  
- [Machine Learning Pipeline](#machine-learning-pipeline)  
- [Evaluation Metrics](#evaluation-metrics)  
- [Model Explainability](#model-explainability)  
- [Saved Models](#saved-models)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Project Overview

This project builds a **full machine learning pipeline** to predict:

1. **Job automation risk category**: Low / Medium / High  
2. **Automation probability (0–1)**: Likelihood of job being automated by 2030  

The pipeline supports both **classification and regression tasks**, along with **feature importance analysis and SHAP explanations**.

---

## Features

- Preprocess numeric and categorical features:
  - Numeric: median imputation + standard scaling  
  - Categorical: constant imputation + one-hot encoding  
- Train multiple models:
  - Classification: Logistic Regression, Ridge Classifier, Decision Tree  
  - Regression: Linear Regression, Ridge, Decision Tree Regressor  
- Evaluate model performance with standard metrics
- Extract **feature importance** from tree-based models
- Explain predictions using **SHAP values**
- Save and load trained pipelines for future predictions

---

## Installation

Install required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap joblib
