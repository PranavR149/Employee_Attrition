#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

class FeatureSelector:
    """
    Feature Selector Class: Selects top features based on importance scores.

    Inputs:
    - n_estimators: int, Number of trees in the forest (default: 100).
    - random_state: int, Seed for random number generator (default: 42).

    Access Level:
    - Public
    """
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def select_features(self, X: pd.DataFrame, y: pd.Series, top_n: int = 28) -> list:
        """
        Selects top features based on importance scores.

        Inputs:
        - X: pd.DataFrame, Features DataFrame.
        - y: pd.Series, Target Series.
        - top_n: int, Number of top features to select (default: 28).

        Output:
        - List of selected top features.

        Access Level:
        - Public
        """
        # Train the model
        self.model.fit(X, y)

        # Get feature importances
        feature_importances = self.model.feature_importances_

        # Create a DataFrame of feature importances
        feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
        
        # Sort features by importance (descending order)
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        
        # Select top N features
        top_n_features = feature_importance_df.head(top_n)['Feature'].tolist()

        return top_n_features


class ModelEvaluator:
    """
    Model Evaluator Class: Evaluates model performance and sets threshold.

    Inputs:
    - model: Any, Machine learning model.
    - X_train, X_val, X_test: pd.DataFrame, Features for training, validation, and testing.
    - y_train, y_val, y_test: pd.Series, Target variables for training, validation, and testing.

    Access Level:
    - Public
    """
    def __init__(self, model: any, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series, y_val: pd.Series, y_test: pd.Series):
        self.model = model
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.best_threshold = None

    def train(self):
        """Trains the model.

        Access Level:
        - Public
        """
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """Evaluates model performance on validation data.

        Access Level:
        - Public
        """
        y_pred_val_proba = self.model.predict_proba(self.X_val)
        if self.best_threshold is None:
            threshold = 0.5  # Default threshold
        else:
            threshold = self.best_threshold
        y_pred_best_threshold = (y_pred_val_proba[:, 1] >= threshold).astype(int)
        return classification_report(self.y_val, y_pred_best_threshold)

    def evaluate_test(self):
        """Evaluates model performance on test data.

        Access Level:
        - Public
        """
        y_pred_test_proba = self.model.predict_proba(self.X_test)
        if self.best_threshold is None:
            threshold = 0.5  # Default threshold
        else:
            threshold = self.best_threshold
        y_pred_best_threshold = (y_pred_test_proba[:, 1] >= threshold).astype(int)
        return classification_report(self.y_test, y_pred_best_threshold)

    def set_threshold(self, threshold: float):
        """Sets the threshold for model classification.

        Access Level:
        - Public
        """
        self.best_threshold = threshold


class BaseModel:
    """Base Model Class: Defines common methods for all models.

    Access Level:
    - Public
    """
    def __init__(self, model: any):
        self.model = model

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits the model to the training data.

        Access Level:
        - Public
        """
        self.model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predicts probabilities for the given data.

        Access Level:
        - Public
        """
        return self.model.predict_proba(X)


class RandomForestModel(BaseModel):
    """
    Random Forest Model Class: Wrapper for RandomForestClassifier.

    Inputs:
    - n_estimators: int, Number of trees in the forest (default: 200).
    - random_state: int, Seed for random number generator (default: 42).

    Access Level:
    - Public
    """
    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        super().__init__(RandomForestClassifier(n_estimators=n_estimators, random_state=random_state))


class XGBModel(BaseModel):
    """
    XGBoost Model Class: Wrapper for XGBClassifier.

    Inputs:
    - n_estimators: int, Number of boosting rounds (default: 200).
    - random_state: int, Seed for random number generator (default: 42).

    Access Level:
    - Public
    """
    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        super().__init__(XGBClassifier(n_estimators=n_estimators, random_state=random_state))


class AdaBoostModel(BaseModel):
    """
    AdaBoost Model Class: Wrapper for AdaBoostClassifier.

    Inputs:
    - n_estimators: int, Number of boosting rounds (default: 180).
    - learning_rate: float, Learning rate shrinks the contribution of each classifier (default: 0.01).
    - random_state: int, Seed for random number generator (default: 42).

    Access Level:
    - Public
    """
    def __init__(self, n_estimators: int = 180, learning_rate: float = 0.01, random_state: int = 42):
        super().__init__(AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state))


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression Model Class: Wrapper for LogisticRegression.

    Inputs:
    - penalty: str, Regularization term (default: 'l1').
    - C: float, Inverse of regularization strength (default: 0.1).
    - solver: str, Algorithm to use in the optimization problem (default: 'liblinear').

    Access Level:
    - Public
    """
    def __init__(self, penalty: str = 'l1', C: float = 0.1, solver: str = 'liblinear'):
        super().__init__(LogisticRegression(penalty=penalty, C=C, solver=solver))


if __name__ == "__main__":
    # Load dataset
    dataset = pd.read_csv("Employee.csv")
    
    # Separate features (X) and target variable (y)
    columns_to_keep = ["MonthlyIncome",
        "Age",
        "DailyRate",
        "DistanceFromHome",
        "OverTime_Num",
        "TotalWorkingYears",
        "NumCompaniesWorked",
        "YearsAtCompany",
        "YearsWithCurrManager",
        "StockOptionLevel",
        "PercentSalaryHike",
        "YearsInCurrentRole",
        "EnvironmentSatisfaction",
        "New EducationField",
        "TrainingTimesLastYear",
        "New JobRole",
        "YearsSinceLastPromotion",
        "RelationshipSatisfaction",
        "JobInvolvement",
        "JobSatisfaction",
        "Education",
        "WorkLifeBalance",
        "Travel_Frequently",
        "Married",
        "Gender_Num",
        "Sales Representative",
        "Divorced",
        "Laboratory Technician",
        "Travel_Rarely",
        "Sales Executive",
        "Research Scientist",
        "PerformanceRating",
        "Non-Travel",
        "Human Resources",
        "Manufacturing Director",
        "Healthcare Representative",
        "Manager",
        "Research Director" ]
    X = dataset[columns_to_keep]
    y = dataset["Attrition_numeric"]
    
    # Train-Validation-Test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create FeatureSelector instance
    feature_selector = FeatureSelector()

    # Select top features
    selected_features = feature_selector.select_features(X_train, y_train)

    models = [
        RandomForestModel(),
        XGBModel(),
        AdaBoostModel(),
        LogisticRegressionModel()
    ]

    thresholds = [0.19, 0.013, 0.328, 0.18]  # Thresholds for each model

    for model, threshold in zip(models, thresholds):
        evaluator = ModelEvaluator(model.model, X_train, X_val, X_test, y_train, y_val, y_test)
        evaluator.train()
        evaluator.set_threshold(threshold)
        print("\033[1m" + type(model).__name__ + "\033[0m")  # Printing model name in bold
        print(evaluator.evaluate_test())  # Evaluating model on test data

