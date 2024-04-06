# Siftwell Coding Challenge Project

## Overview

This project was undertaken as part of a coding challenge issued by Siftwell, focusing on building supervised machine learning models to address two main objectives:

1. Regression Model: To predict the total future cost based on various features.
2. Binary Classification Model: To classify whether an individual is likely to receive mental health treatment, with the output being predicted probabilities.
3. The challenge involved comprehensive data preparation, exploratory data analysis (EDA), model development and validation, and generating predictions on unseen data.

## Project Structure

### The project is structured as follows:

1. data/raw/: Contains the raw CSV files used for the challenge.
- coding_challenge_train.csv: Training data with input features and target labels.
- coding_challenge_test_without_labels.csv: Test data with input features only.
2. src/models/: Contains the saved model files (post-training) for easy reuse.
- regression_model.joblib: Saved RandomForestRegressor model.
- classification_model.joblib: Saved RandomForestClassifier model.
3. notebooks/: Contains Jupyter notebooks used for EDA and initial model training experiments.
- eda.ipynb: Notebook containing exploratory data analysis.
4. src/: Contains Python scripts for training models and generating predictions.
- regression_model.py: Script for training the regression model.
- classification_model.py: Script for training the classification model.
- generate_predictions.py: Script that manages model training/reuse and generates predictions on the test dataset.

### Key Insights from EDA

- Significant correlations were found among features and between features and target variables, guiding the feature selection and engineering process.
- Non-linear relationships and outliers were identified, informing preprocessing steps.
- Data quality checks revealed missing values and categorical features requiring encoding and imputation strategies.

### Generating Predictions
#### Steps to Run:
1. Prepare Your Environment: Ensure you have Python 3.8+ installed along with necessary libraries (pandas, numpy, scikit-learn, matplotlib, seaborn, and joblib).
2. Load and Inspect Data: Use the eda.ipynb notebook for initial data exploration and to understand the dataset's structure and characteristics.
3. Generate Predictions: Execute the generate_predictions.py script. This script will:
- Check for existing models and offer an option to retrain.
- Load the test dataset.
- Use the trained models to generate predictions on the test dataset.
- Provide an option for incremental updates to the predictions file or a full refresh.
- Save the prediction results in the data/submission/ directory as per the format specified in sample_prediction_file.csv.
4. Submission: Submit the generated CSV file along with the Python code files as per the challenge's submission guidelines. All files and data will be stored in a public GitHub repo to be shared with Siftwell Analytics.

### Evaluation Criteria

- Model performance is evaluated based on accuracy in predicting unseen data.
- Code readability, structure, and adherence to best practices.
Innovative approaches and solutions to the data challenges presented.
This project underscores the importance of thorough EDA, careful model selection, and rigorous validation in tackling predictive modeling tasks. By adhering to a structured workflow and employing best practices, the project aims to develop robust models capable of generating reliable predictions for both regression and classification tasks.