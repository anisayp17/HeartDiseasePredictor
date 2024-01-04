# Heart Disease Predictor App

## Overview

This repository contains code for a Streamlit web application that predicts heart disease based on input parameters. The application includes three tabs: Single-predict, Multi-predict, and Description.

## Requirements
- Python 3.x
- pandas==2.1.1
- numpy==1.26.0
- imbalanced-learn==0.11.0
- scikit-learn==1.3.1
- streamlit==1.22.0
- xgboost==2.0.2


## Instructions

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit app with `streamlit run app.py`.

## Code Structure

- `app.py`: Main file containing the Streamlit application code.
- `model/xgb_model.pkl`: Pre-trained XGBoost model file.
- `logo.png`: Logo file for the web application.

## Usage

- **Single-predict:** Predict heart disease for a single set of input parameters.
- **Multi-predict:** Predict heart disease for multiple sets of input parameters.
- **Description:** Provides a description of each feature in the dataset.

## Dataset

The application uses a hungarian heart disease dataset. For more details on the dataset, refer to the [original source](https://archive.ics.uci.edu/dataset/45/heart+disease).
