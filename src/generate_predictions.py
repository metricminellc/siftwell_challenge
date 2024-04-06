# Description: This script manages the model training process and generates predictions

import pandas as pd
import os
from joblib import load
# Import model training functions for both regression and classification tasks
from regression_model import train_regression_model
from classification_model import train_classification_model


def train_model(model_name):
    """
    Manages the model training process or loads an existing model based on user input.
    Automatically trains a new model if none exists on disk. Provides an option for
    the user to retrain models to ensure the models are up-to-date with the latest data.

    Parameters:
    - model_name: A string indicating which model to train or load ('regression' or 'classification').

    Returns:
    - The trained or loaded model object.
    """
    # Check for an existing model file and train a new model if it doesn't exist
    model_path = f'../src/models/{model_name}_model.joblib'
    if not os.path.exists(model_path):
        print(f"No model exists. Training new {model_name} model...")
        if model_name == 'regression':
            return train_regression_model()
        elif model_name == 'classification':
            return train_classification_model()

    # If model exists, prompt the user to either retrain or use the existing model
    response = input(f"Do you want to retrain the {model_name} model? (y/n): ").strip().lower()
    if response == 'y':
        if model_name == 'regression':
            return train_regression_model()
        elif model_name == 'classification':
            return train_classification_model()
    elif response == 'n':
        return load(f'../src/models/{model_name}_model.joblib')
    else:
        print(f"Invalid input. Using existing {model_name} model.")
        return load(f'../src/models/{model_name}_model.joblib')


def update_submission_file(submission_df, incremental=False):
    """
    Updates the submission file with new predictions. It can perform a full refresh of the
    submission file or an incremental update, adding only new rows based on 'line_number'.

    Parameters:
    - submission_df: DataFrame containing the latest predictions.
    - incremental: A boolean flag indicating whether to perform an incremental update.
    """
    submission_path = '../data/submission/submission_prediction_file.csv'
    if incremental and os.path.exists(submission_path):
        submission_old_df = pd.read_csv(submission_path)
        submission_new_df = submission_df[~submission_df['line_number'].isin(submission_old_df['line_number'])]
        submission_inc_df = pd.concat([submission_old_df, submission_new_df])
        submission_inc_df.to_csv(submission_path, index=False)
        print(f"Incremental submission update completed with {len(submission_new_df)} new rows added.")
    else:
        submission_df.to_csv(submission_path, index=False)
        print(f"Full refresh of submission file completed with {len(submission_df)} rows.")


# Main logic for managing model training and generating submission predictions
# Train or load models for regression and classification tasks
regression_model = train_model('regression')
classification_model = train_model('classification')

# Load the test data
test_df = pd.read_csv('../data/raw/coding_challenge_test_without_labels.csv')

# Generate predictions using the trained models
regression_predictions = regression_model.predict(test_df)
classification_predictions = classification_model.predict_proba(test_df)[:, 1]

# Prepare the submission DataFrame
submission_df = pd.DataFrame({
    'line_number': test_df['line_number'],
    'prediction_total_cost_future': [f"{x:.2f}" for x in regression_predictions],
    'prediction_treatment__mental_health': [f"{x:.8f}" for x in classification_predictions]
})

# Determine if an incremental update is required
incremental = False
if os.path.exists('../data/submission/submission_prediction_file.csv'):
    user_input = input("Do you want to run an incremental update to the submission file? (y/n): ").strip().lower()
    incremental = user_input == 'y'

# Update the submission file as per user choice
update_submission_file(submission_df, incremental=incremental)
