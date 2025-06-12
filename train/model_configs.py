from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier

MODEL_CONFIGS = {
    'dt': {
        'model': DecisionTreeClassifier,
        'hyperparameters': {
            'criterion': ['gini', 'entropy'],
            'max_depth': (3, 20),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10)
        }
    },
    'knn': {
        'model': KNeighborsClassifier,
        'hyperparameters': {
            'n_neighbors': (3, 20),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    },
    'gnb': {
        'model': GaussianNB,
        'hyperparameters': {
            'var_smoothing': (1e-10, 1e-8, 'log')
        }
    },
    'sgd': {
        'model': SGDClassifier,
        'hyperparameters': {
            'loss': ['hinge', 'log_loss', 'modified_huber'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': (1e-5, 1e-2, 'log'),
            'max_iter': (500, 2000)
        }
    },
    'bag': {
        'model': BaggingClassifier,
        'hyperparameters': {
            'n_estimators': (5, 50),
            'max_samples': (0.5, 1.0),
            'max_features': (0.5, 1.0)
        }
    }
}