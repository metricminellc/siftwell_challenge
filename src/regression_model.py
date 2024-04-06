# Regression Model: Develop a supervised machine learning model to predict the probability of receiving mental health
# treatment based on the features provided in the training file.

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def train_regression_model():
    """
    Train the regression model using RandomForestRegressor to predict 'total_cost_future'.
    This model is selected for its robustness and versatility across a wide range of datasets,
    including those with complex, nonlinear relationships and interactions between features.

    Model Selection Rationale:
    - RandomForestRegressor is capable of handling the intricate correlations observed between
      features and the target variable 'total_cost_future', as well as among features themselves.
      Unlike linear models, which may struggle with multicollinearity and interactions unless explicitly
      modeled, RandomForest can naturally capture these dynamics through its ensemble of decision trees.

    - This model does not assume that features are independent. Its tree-based structure allows it
      to consider various combinations of features, making it adept at uncovering the underlying
      structure in data where features are interrelated. This is crucial in our context, given the
      significant correlations and interactions identified during the EDA process.

    - It's particularly suited for larger datasets as it scales well with data size. While it requires
      more computational resources than simpler models, its ability to run in parallel and handle large,
      complex datasets effectively makes it a strong candidate for our regression task.

    - The default setting of n_estimators=100 provides a balance between model performance and
      computational efficiency. This number of trees is typically sufficient for achieving robust
      predictions while keeping the training time manageable.

    Implementation Notes:
    - The model pipeline integrates preprocessing steps, including imputation and encoding, to
      handle missing values and categorical features, ensuring the RandomForestRegressor works
      with cleaned and appropriately formatted data.
    - After training, the model is saved to a joblib file for persistence, enabling reuse without
      retraining for subsequent predictions on new data.
    """
    # Load the training data
    train_df = pd.read_csv('../data/raw/coding_challenge_train.csv')

    # Prepare features and target
    X = train_df.drop(columns=['total_cost_future', 'treatment__mental_health'])
    y = train_df['total_cost_future']

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Define numeric and categorical pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Now integrate this `preprocessor` in your model pipeline
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model
    model_pipeline.fit(X_train, y_train)

    # Save the model to a file
    model_path = '../src/models/regression_model.joblib'
    joblib.dump(model_pipeline, model_path)

    return model_pipeline
