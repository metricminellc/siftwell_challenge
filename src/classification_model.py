# Binary Classification Model: Build another supervised model, this time to classify whether an individual is likely to
# receive mental health treatment. Please generate predicted probabilities.

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def train_classification_model():
    """
    Trains a RandomForestClassifier for a binary classification task focused on predicting
    'treatment__mental_health'. The RandomForestClassifier is chosen for several key reasons,
    detailed below, which align with the insights gained during the exploratory data analysis (EDA)
    phase and the specific characteristics of our dataset.

    Model Selection Rationale:
    - Our EDA revealed significant correlations among features and between features and the target variable.
      Unlike simpler models that might assume independence among features (e.g., Naive Bayes), RandomForest
      is capable of capturing complex interactions between features without needing those interactions to be
      manually specified. This makes it particularly suited for our data, where such correlations are prominent.

    - RandomForest does not presuppose any specific data distribution or linearity in the relationship between
      features and the target, offering flexibility in handling the diverse and complex patterns present in our dataset.

    - The model's ensemble nature, aggregating decisions from multiple decision trees, naturally handles variance
      and bias, providing robustness against overfitting, which is crucial given the correlations and interactions
      observed during EDA.

    - It performs well on large datasets and maintains reasonable performance even with the inclusion of many features,
      as is the case in our dataset. This scalability is important for potential applications to larger datasets in the future.

    Configuration:
    - Adjustments to the model's hyperparameters (n_estimators, max_depth, min_samples_split, and min_samples_leaf)
      are made to fine-tune its performance, balancing model complexity with the need to accurately capture the underlying
      data structure. These settings aim to mitigate overfitting while preserving the model's ability to model the nuanced
      relationships identified during EDA.
    """
    # Load the training data
    train_df = pd.read_csv('../data/raw/coding_challenge_train.csv')

    # Prepare features and target
    X_clf = train_df.drop(columns=['total_cost_future', 'treatment__mental_health'])
    y_clf = train_df['treatment__mental_health']

    # Identify numeric and categorical columns
    numeric_features = X_clf.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_clf.select_dtypes(include=['object']).columns

    # Define preprocessing pipelines for numeric and categorical features
    numeric_transformer_clf = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer_clf = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine into a ColumnTransformer
    preprocessor_clf = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer_clf, numeric_features),
            ('cat', categorical_transformer_clf, categorical_features)])

    # Adjusted RandomForestClassifier settings
    classifier = RandomForestClassifier(
        n_estimators=150,  # Increased number of trees
        max_depth=10,  # Limited depth to control model complexity
        min_samples_split=4,  # Require more samples for a split
        min_samples_leaf=2,  # Require more samples for a leaf node
        max_features='sqrt',  # Limit the number of features considered at each split
        random_state=42,
        class_weight='balanced'
    )

    # Create the RandomForestClassifier pipeline
    classifier_pipeline = Pipeline(steps=[('preprocessor', preprocessor_clf),
                                          ('classifier', classifier)])

    # Split the data into training and validation sets
    X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    # Train the model
    classifier_pipeline.fit(X_train_clf, y_train_clf)

    # Save the model to a file
    model_path = '../src/models/classification_model.joblib'
    joblib.dump(classifier_pipeline, model_path)

    return classifier_pipeline
