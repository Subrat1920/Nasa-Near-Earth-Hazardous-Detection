from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    ExtraTreesClassifier, 
    AdaBoostClassifier
)

PARAMS = [
    ('Logistic Regression', LogisticRegression(), {
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'C': [0.1, 0.3, 0.5, 0.7, 0.9],
        'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
        'max_iter': [100, 300, 500, 700, 900]
    }),

    ('Decision Tree Classifier', DecisionTreeClassifier(), {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [x for x in range(1, 10, 2)],
        'min_samples_split': [x for x in range(2, 10, 2)],
        'min_samples_leaf': [x for x in range(1, 10, 3)],
        'max_features': ['sqrt', 'log2', None]
    }),

    ('Random Forest Classifier', RandomForestClassifier(), {
        'n_estimators': [x*100 for x in range(1, 10, 2)],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [x for x in range(3, 10, 2)],
        'min_samples_split': [x for x in range(2, 10, 2)],
        'min_samples_leaf': [x for x in range(1, 10, 3)],
        'max_features': ['sqrt', 'log2', None]
    }),

    ('Gradient Boosting Classifier', GradientBoostingClassifier(), {
        'loss': ['log_loss', 'exponential'],
        'learning_rate': [x/10 for x in range(1, 10, 3)],
        'n_estimators': [x*100 for x in range(1, 10, 2)],
        'subsample': [x/10 for x in range(1, 10, 3)],
        'criterion': ['friedman_mse', 'squared_error'],
        'max_depth': [x for x in range(3, 10, 2)],
        'max_features': ['sqrt', 'log2']
    }),

    ('K Nearest Neighbors Classifier', KNeighborsClassifier(), {
        'n_neighbors': [x for x in range(1, 10, 3)],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [x for x in range(30, 100, 20)],
        'p': [x for x in range(2, 10, 3)]
    }),

    ('Extra Trees Classifier', ExtraTreesClassifier(), {
        'n_estimators': [100, 200, 500, 1000],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2', None]
    }),

    ('AdaBoost Classifier', AdaBoostClassifier(), {
        'n_estimators': [50, 100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'algorithm': ['SAMME', 'SAMME.R']
    }),
]