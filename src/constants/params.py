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
    # Logistic Regression
    ('Logistic Regression', LogisticRegression(), {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],   # fixed None -> 'none'
        'C': [0.1, 0.3, 0.5, 0.7, 0.9],
        'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
        'max_iter': [100, 300, 500, 700, 900],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # required if penalty='elasticnet'
    }),

    # Decision Tree
    ('Decision Tree Classifier', DecisionTreeClassifier(), {
        'criterion': ['gini', 'entropy', 'log_loss'],   # added log_loss
        'splitter': ['best', 'random'],
        'max_depth': [x for x in range(1, 50, 2)],
        'min_samples_split': [x for x in range(2, 50, 2)],
        'min_samples_leaf': [x for x in range(1, 50, 3)],
        'max_features': ['sqrt', 'log2', None],
        'min_impurity_decrease': [x/10 for x in range(1, 50, 2)],
        'ccp_alpha': [x/10 for x in range(1, 50, 2)]
    }),

    # Random Forest
    ('Random Forest Classifier', RandomForestClassifier(), {
        'n_estimators': [x*100 for x in range(1, 50, 2)],
        'criterion': ['gini', 'entropy', 'log_loss'],   # log_loss supported
        'max_depth': [x for x in range(3, 50, 2)],
        'min_samples_split': [x for x in range(2, 50, 2)],
        'min_samples_leaf': [x for x in range(1, 50, 3)],
        'max_features': ['sqrt', 'log2', None],
        'min_impurity_decrease': [x/10 for x in range(1, 50, 2)],
        'ccp_alpha': [x/10 for x in range(1, 50, 2)]
    }),

    # Gradient Boosting
    ('Gradient Boosting Classifier', GradientBoostingClassifier(), {
        'loss': ['log_loss', 'exponential'],
        'learning_rate': [x/10 for x in range(1, 50, 3)],
        'n_estimators': [x*100 for x in range(1, 50, 2)],
        'subsample': [x/10 for x in range(1, 50, 3)],
        'criterion': ['friedman_mse', 'squared_error'],
        'max_depth': [x for x in range(3, 50, 2)],
        'max_features': ['sqrt', 'log2'],
        'warm_start': [False, True],
        'validation_fraction': [x/10 for x in range(1, 50, 2)],
        'n_iter_no_change': [x for x in range(1, 5)],
        'ccp_alpha': [x/10 for x in range(1, 50, 2)]
        # removed tol (invalid)
    }),

    # K Nearest Neighbors
    ('K Nearest Neighbors Classifier', KNeighborsClassifier(), {
        'n_neighbors': [x for x in range(1, 50, 3)],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [x for x in range(30, 100, 20)],
        'p': [x for x in range(2, 50, 3)]
    }),

    # Extra Trees
    ('Extra Trees Classifier', ExtraTreesClassifier(), {
        'n_estimators': [100, 200, 500, 1000],
        'criterion': ['gini', 'entropy', 'log_loss'],   # added log_loss
        'max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'max_features': ['sqrt', 'log2', None],
        'ccp_alpha': [x/10 for x in range(1, 50, 2)],
        'warm_start': [True, False],
        'bootstrap': [True, False],                     # added so max_samples works
        'max_samples': [x/10 for x in range(1, 10)],    # valid only if bootstrap=True
        # removed monotonic_cst (invalid)
    }),

    # AdaBoost
    ('AdaBoost Classifier', AdaBoostClassifier(), {
        'n_estimators': [50, 100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'algorithm': ['SAMME', 'SAMME.R']
    }),
]
