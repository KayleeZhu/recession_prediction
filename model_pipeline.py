import pandas as pd
import numpy as np

from timeseriescv import cross_validation as kfoldcv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV


def train_test_split(df, train_start, train_end, test_start, test_end):
    """
    Split the data into two parts by specified dates: training set & test set
    Training set used for cross validation
    Test set used for out sample performance
    """
    df_train = df.loc[(df.date >= train_start) & (df.date <= train_end)]
    df_test = df.loc[(df.date >= test_start) & (df.date <= test_end)]

    # Get the index of train & test set
    train_index = df_train.index
    test_index = df_test.index

    return df_train, df_test, (train_index, test_index)


def get_train_val_indices(data, lag, recession_period):
    """
    This function will split the data
    Get the indices of train & validation set
    :param data: The entire data which needed to be split
    :param lag: How long the lag you want to apply for validation set and training set
    :param recession_period: can be 3m, 6m or 12m
    :return: the index of each fold
    """
    data['date'] = pd.to_datetime(data['date'])
    cv = kfoldcv.CombPurgedKFoldCV(n_splits=5, n_test_splits=1, embargo_td=pd.Timedelta(days=lag + 10))
    data = data.sort_values('date')

    month = int(recession_period[:-1])
    data['evaluation_date'] = data['date'].shift(-month)
    fold_indices = list(cv.split(data, pred_times=data['date'], eval_times=data['evaluation_date']))

    return fold_indices


def get_parameter_distribution():
    """
    Define a dictionary of parameters distribution for each model
    """
    param_dist = {'logistic': {'classifier__penalty': ['elasticnet'],
                               'classifier__dual': [False],
                               'classifier__C': np.arange(0.3, 5, 0.5),
                               'classifier__multi_class': ['auto'],
                               'classifier__max_iter': [100, 500, 1000, 1500, 2000, 3000],
                               'classifier__solver': ['saga'],
                               'classifier__l1_ratio': np.arange(0, 1.2, 0.2)
                               },
                  'svc': {'classifier__C': np.arange(0.3, 5, 0.5),
                          'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                          'classifier__degree': [1, 2, 3],
                          'classifier__gamma': ['scale', 'auto'],
                          'classifier__probability': [True]
                          }
                  }
    return param_dist


class ModelPipeline:
    """
    This Model Pipeline defines the Machine Learning pipeline process:
    - Applying standard scaler
    - Set up algorithm: Logistic or SVC
    - Hyperparameter tuning through 5-fold CV
    - Fit model with given X
    - Get prediction with given X
    """

    def __init__(self, model_type, feature_list, y_var_name, recession_period,  score_method, random_state):
        self.model_type = model_type
        self.feature_list = feature_list
        self.y_var_name = y_var_name
        self.recession_period = recession_period
        self.score_method = score_method
        self.random_state = random_state

        self.param_dist = get_parameter_distribution()
        self.pipeline = self.set_up_pipeline()

    def set_up_pipeline(self):
        # Generate pipeline for the given model type
        if self.model_type == 'logistic':
            pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', LogisticRegression())])
        elif self.model_type == 'svc':
            pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', SVC())])
        return pipeline

    def get_features(self, X_data):
        X = X_data[self.feature_list].copy()
        return X

    def get_labels(self, y_data):
        return y_data[self.y_var_name].copy()

    def hyperparameter_tuning(self, X_data, y_data):
        # Get fold indicies:
        fold_indices = get_train_val_indices(X_data, lag=5, recession_period=self.recession_period)

        # Drop non-features / non-labels columns
        X = self.get_features(X_data)
        y = self.get_labels(y_data)

        # Search for best params
        cv = RandomizedSearchCV(self.pipeline,
                                self.param_dist[self.model_type],
                                random_state=self.random_state,
                                scoring=self.score_method,
                                n_jobs=-1,
                                cv=fold_indices,
                                n_iter=5)
        cv.fit(X, y)
        self.pipeline = cv.best_estimator_

    def model_training(self, X_data, y_data):
        # Drop non-features / non-labels columns
        X = self.get_features(X_data)
        y = self.get_labels(y_data)

        # Train the model
        self.pipeline.fit(X, y)

    def get_prediction(self, X_data, y_data):
        # Drop non-features columns
        X = self.get_features(X_data)

        # Get predicted labels and prediction probabilities
        y_pred = self.pipeline.predict(X)
        prob = self.pipeline.predict_proba(X)
        # Attach the prediction and prob to data
        y_data['y_pred'] = y_pred
        y_data[['prob_0', 'prob_1']] = prob
        return y_data
