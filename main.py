from metaflow import FlowSpec, step, card, environment, Parameter, current, resources, pypi_base
import pandas as pd
import os
from datetime import datetime

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score

# Sklearn models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
# from sklearn.ensemble import VotingClassifier


# Sklearn hyperopt
from hpsklearn import *
# from hyperopt import hp, STATUS_OK, Trials, fmin
from hyperopt import tpe

# MLflow
import mlflow
from mlflow.models import infer_signature
from mlflow.data.pandas_dataset import PandasDataset

# For drawing plot
import matplotlib.pyplot as plt
plt.style.use("ggplot")  #using style ggplot
from metaflow.cards import VegaChart
from metaflow.cards import Markdown

@pypi_base(
    python="3.13.3",
    packages={
        "hyperopt-sklearn": "1.1.1",
        "matplotlib": " 3.10.1",
        "mlflow": "2.22.0",
        "xgboost": "3.0.0",
        "metaflow": "2.15.10"
    }
)


class MLFlowPipeline(FlowSpec):
    """MLflow pipeline for training and evaluating wine quality models.
    
    This pipeline implements:
    - Data loading and preprocessing
    - Exploratory data analysis
    - Model training with cross-validation
    - Ensemble model creation
    """
    
    DATA_DIR = Parameter(
        "data_dir",
        type=str,
        default="./dataset/",
        help="The relative location of the folder containing the dataset.",
    )
    
    DATASET_URL = Parameter(
        "dataset_url",
        type=str,
        default="https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv",
        help="The URL of the dataset to be used.",
    )
    
    USE_HYPEROPT = Parameter(
        "use_hyperopt",
        type=bool,
        default=False,
        help="Whether to use hyperparameter optimization or default parameters.",
    )
    
    @environment(
        vars={
            "MLFLOW_TRACKING_URI": os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://127.0.0.1:5000",
            ),

            "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": os.getenv(
                "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING",
                "true",
            ),
        },
    )
        
    @card
    @resources(cpu=8, memory=4000)
    @step
    def start(self):
        '''
        Set up the MLflow tracking URI and define the models to be trained.
        '''
        self.mlflow_enable_system_metrics_logging = os.getenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING")
        
        if self.mlflow_enable_system_metrics_logging:
            # For example, if you set sample interval to 2 seconds and samples before logging to 3, 
            # then system metrics will be collected every 2 seconds, 
            # then after 3 samples are collected (2 * 3 = 6s), 
            # we aggregate the metrics and log to MLflow server.
            
            # For some reason, mlflow doesn't log any model's system metrics that run less than 10 seconds even when manually set below
            
            mlflow.set_system_metrics_sampling_interval(1)
            mlflow.set_system_metrics_samples_before_logging(1)
            mlflow.enable_system_metrics_logging()
            print("System metrics logging is enabled.")
        
        if not isinstance(self.DATA_DIR, str) or not isinstance(self.DATASET_URL, str):
            raise TypeError("DATA_DIR and DATASET_URL must be strings")

        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Define model configurations
        self.model_configs = [
            {'name': model_name, 'model': model_class} 
            for model_name, model_class in {
                'dt': DecisionTreeClassifier,
                # 'rf': RandomForestClassifier,
                # 'ada': AdaBoostClassifier,
                # 'et': ExtraTreesClassifier,
                'bag': BaggingClassifier,
                'knn': KNeighborsClassifier,
                # 'xgb': XGBClassifier,
                'gnb': GaussianNB,
                'sgd': SGDClassifier,
                # 'svc': SVC,
                # 'lsvc': LinearSVC,
                # 'gb': GradientBoostingClassifier,
                # 'mlp': MLPClassifier,
            }.items()
        ]

        self.next(self.load_dataset)
        
        
    @card
    @resources(cpu=8, memory=4000)
    @step
    def load_dataset(self):
        '''
        Download the dataset from the given URL and save it locally.
        '''
        # Get current date for versioning
        current_date = datetime.now().strftime("%Y%m%d")
        
        # if exist dataset in dataset folder then skip download
        if os.path.exists(self.DATA_DIR):
            print("Dataset already exists. Skipping download.")
            self.dataset_path = os.path.join(self.DATA_DIR, "winequality-white.csv")
            self.df = pd.read_csv(self.dataset_path)
            
            # Log dataset information with MLflow
            with mlflow.start_run(run_name="dataset_loading") as run:
                # pylint: disable=no-member
                dataset = mlflow.data.from_pandas( 
                    self.df, 
                    source=self.DATASET_URL, 
                    name="wine quality - white", 
                    targets="quality"
                ) 
                
                mlflow.log_input(
                    dataset=dataset,
                    context="training",
                    tags={
                        "dataset_name": "wine_quality",
                        "dataset_version": current_date,  # Using current date as version
                        "dataset_shape": str(self.df.shape),
                        "features": str(self.df.columns.tolist()),
                        "data_source": "local_storage",
                        "missing_values": str(self.df.isnull().sum().to_dict())
                    }
                )
                
                # Log dataset statistics
                mlflow.log_dict(
                    self.df.describe().to_dict(),
                    "raw_dataset_statistics.json"
                )
                
                # Log dataset artifact
                mlflow.log_artifact(self.dataset_path, "raw_dataset")
                
                # Set dataset-level tags
                mlflow.set_tag("pipeline_stage", "data_loading")
                mlflow.set_tag("total_samples", len(self.df))
                mlflow.set_tag("total_features", len(self.df.columns))
                
            self.next(self.eda)
            return
            
        # Create directory if it doesn't exist
        os.makedirs(self.DATA_DIR, exist_ok=True)
        
        # Download dataset
        try:
            raw_data = pd.read_csv(self.DATASET_URL, delimiter=";")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise
        
        # Save dataset to local directory
        self.dataset_path = os.path.join(self.DATA_DIR, "winequality-white.csv")
        raw_data.to_csv(self.dataset_path, index=False)
        print(f"Dataset downloaded and saved to {self.dataset_path}")
        
        self.df = pd.read_csv(self.dataset_path)
        
        # Log downloaded dataset information with MLflow
        with mlflow.start_run(run_name="dataset_loading") as run:
            # pylint: disable=no-member
            dataset = mlflow.data.from_pandas( 
                self.df, 
                source=self.DATASET_URL, 
                name="wine quality - white", 
                targets="quality"
            ) 
            
            mlflow.log_input(
                dataset=dataset,
                context="training",
                tags={
                    "dataset_name": "wine_quality",
                    "dataset_version": current_date,  # Using current date as version
                    "dataset_shape": str(self.df.shape),
                    "features": str(self.df.columns.tolist()),
                    "data_source": "local_storage",
                    "missing_values": str(self.df.isnull().sum().to_dict())
                }
            )
            
            # Log dataset statistics
            mlflow.log_dict(
                self.df.describe().to_dict(),
                "raw_dataset_statistics.json"
            )
            
            # Log dataset artifact
            mlflow.log_artifact(self.dataset_path, "raw_dataset")
            
            # Set dataset-level tags
            mlflow.set_tag("pipeline_stage", "data_loading")
            mlflow.set_tag("total_samples", len(self.df))
            mlflow.set_tag("total_features", len(self.df.columns))
            
        # Retrieve the run information
        logged_run = mlflow.get_run(run.info.run_id)

        # Retrieve the Dataset object
        logged_dataset = logged_run.inputs.dataset_inputs[0].dataset

        # View some of the recorded Dataset information
        print(f"Dataset name: {logged_dataset.name}")
        print(f"Dataset digest: {logged_dataset.digest}")
        print(f"Dataset profile: {logged_dataset.profile}")
        print(f"Dataset schema: {logged_dataset.schema}")    
        
        self.next(self.eda)
    
            
    @card
    @resources(cpu=8, memory=4000)  
    @step
    def eda(self):
        '''
        Perform exploratory data analysis (EDA) on the dataset.
        '''
        # Dataset statistics
        current.card.append(Markdown(
            "## Dataset Statistics\n" +
            f"* Number of samples: {len(self.df)}\n" +
            f"* Number of features: {len(self.df.columns) - 1}\n")
        )
        # current.card.append(Markdown(f"* Quality distribution:\n{self.df['quality'].value_counts().to_markdown()}\n"))
        quality_dist = self.df['quality'].value_counts().to_string()
        current.card.append(Markdown(f"* Quality distribution:\n```\n{quality_dist}\n```"))

        
        # Create visualizations for the card
        current.card.append(
            VegaChart(
                {
                    "data": {"values": self.df.to_dict(orient="records")},
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": "quality", "type": "nominal"},
                        "y": {"aggregate": "count", "type": "quantitative"}
                    },
                    "title": "Distribution of Wine Quality"
                }
            )
        )

        # Quality vs Features relationships
        features = ['volatile acidity', 'citric acid', 'chlorides', 'pH', 'sulphates']
        for feature in features:
            current.card.append(
                VegaChart(
                    {
                        "data": {"values": self.df.to_dict(orient="records")},
                        "mark": "line",
                        "encoding": {
                            "x": {"field": "quality", "type": "nominal"},
                            "y": {
                                "field": feature,
                                "type": "quantitative",
                                "aggregate": "mean"
                            }
                        },
                        "title": f"Average {feature.title()} by Quality"
                    }
                )
            )

        # Alcohol relationship
        current.card.append(
            VegaChart(
                {
                    "data": {"values": self.df.to_dict(orient="records")},
                    "mark": "line",
                    "encoding": {
                        "x": {"field": "quality", "type": "nominal"},
                        "y": {
                            "field": "alcohol",
                            "type": "quantitative",
                            "aggregate": "mean"
                        }
                    },
                    "title": "Average Alcohol Content by Quality"
                }
            )
        )

        # Sulfur dioxide relationships
        for sulfur_type in ['free sulfur dioxide', 'total sulfur dioxide']:
            current.card.append(
                VegaChart(
                    {
                        "data": {"values": self.df.to_dict(orient="records")},
                        "mark": "line",
                        "encoding": {
                        "x": {"field": "quality", "type": "nominal"},
                            "y": {
                                "field": sulfur_type,
                                "type": "quantitative",
                                "aggregate": "mean"
                            }
                        },
                        "title": f"Average {sulfur_type.title()} by Quality"
                    }
                )
            )

        # Correlation heatmap
        correlation_data = []
        corr_matrix = self.df.corr()
        for col1 in corr_matrix.columns:
            for col2 in corr_matrix.columns:
                correlation_data.append({
                    "var1": col1,
                    "var2": col2,
                    "correlation": corr_matrix.loc[col1, col2]
                })

        current.card.append(
            VegaChart(
                {
                    "data": {"values": correlation_data},
                    "mark": "rect",
                    "encoding": {
                        "x": {"field": "var1", "type": "nominal"},
                        "y": {"field": "var2", "type": "nominal"},
                        "color": {"field": "correlation", "type": "quantitative"}
                    },
                    "title": "Feature Correlation Heatmap"
                }
            )
        )


        self.next(self.load_data)

    @card
    @resources(cpu=8, memory=4000)
    @step
    def load_data(self):
        '''
        Split it into training and validation sets and log dataset information.
        '''
        X = self.df.drop('quality', axis=1).values
        y = self.df['quality'].values
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.next(self.train_models, foreach='model_configs')

    @card
    @resources(cpu=8, memory=4000)
    @step
    def train_models(self):
        '''
        Train models using either HyperoptEstimator or default scikit-learn models.
        '''
        import time
        self.trained_models = []
        model_config = self.input
        model_name = model_config['name']

        # Add label encoding for hyperopt
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(self.y_train)
        y_val_encoded = label_encoder.transform(self.y_val)

        # Map model names to their hpsklearn and default configurations
        model_mappings = {
            # Training Time: 8.49 seconds
            'dt': {
                'hp_model': lambda: decision_tree_classifier('dt_classifier'),
                'criterion': 'log_loss',
                'default_params': {'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1}
            },
            # # Training Time: 425.55 seconds
            # 'rf': {
            #     'hp_model': lambda: random_forest_classifier('rf_classifier'),
            #     'criterion': 'log_loss',
            #     'default_params': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2}
            # },
            # # Training Time: 111.14 seconds
            # 'ada': {
            #     'hp_model': lambda: ada_boost_classifier('ada_classifier'),
            #     'default_params': {'n_estimators': 50, 'learning_rate': 1.0}
            # },
            # # Training Time: 178.22 seconds
            # 'et': {
            #     'hp_model': lambda: extra_trees_classifier('et_classifier'),
            #     'default_params': {'n_estimators': 100, 'max_depth': 10}
            # },
            # Training Time: 41.07 seconds
            'bag': {
                'hp_model': lambda: bagging_classifier('bag_classifier'),
                'default_params': {'n_estimators': 10}
            },
            # Training Time: 33.31 seconds
            'knn': {
                'hp_model': lambda: k_neighbors_classifier('knn_classifier'),
                'default_params': {'n_neighbors': 5, 'weights': 'uniform'}
            },
            # # Training Time: 693.22 seconds
            # 'xgb': {
            #     'hp_model': lambda: xgboost_classification('xgb_classifier'),
            #     'default_params': {
            #         'n_estimators': 100,
            #         'learning_rate': 0.1,
            #         'max_depth': 3,
            #         'gamma': 0
            #     }
            # },
            # Training Time: 4.51 seconds
            'gnb': {
                'hp_model': lambda: gaussian_nb('gnb_classifier'),
                'default_params': {}
            },
            # Training Time: 21.67 seconds
            'sgd': {
                'hp_model': lambda: sgd_classifier('sgd_classifier'),
                'default_params': {'loss': 'hinge', 'penalty': 'l2', 'alpha': 0.0001}
            },
            # # Training Time: 909.88 seconds
            # 'svc': {
            #     'hp_model': lambda: svc('svc_classifier'),
            #     'default_params': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'probability': True}
            # },
            # # Training Time: 260.15 seconds
            # 'lsvc': {
            #     'hp_model': lambda: linear_svc('lsvc_classifier'),
            #     'default_params': {'C': 1.0, 'loss': 'squared_hinge'}
            # },
            # # Buggy models
            # https://github.com/hyperopt/hyperopt-sklearn/issues/141#issuecomment-548502709
            # 'gb': {
            #     'hp_model': lambda: gradient_boosting_classifier('gb_classifier'),
            #     'criterion': 'log_loss',
            #     'default_params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
            # },
            # 'mlp': {
            #     'hp_model': lambda: mlp_classifier('mlp_classifier'),
            #     'default_params': {'hidden_layer_sizes': (100,), 'activation': 'relu'}
            # },
        }

        with mlflow.start_run(run_name=f"model_{model_name}") as run:
            try:
                start_time = time.time()
                
                if self.USE_HYPEROPT and model_name in model_mappings:
                    print(f"[{model_name}] Using HyperoptEstimator...")
                    estimator = HyperoptEstimator(
                        classifier=model_mappings[model_name]['hp_model'](),
                        preprocessing=any_preprocessing('pre'),
                        algo=tpe.suggest,
                        max_evals=50,
                        trial_timeout=300,
                        seed=42
                    )
                    # Use encoded labels for training
                    model = estimator
                    
                    model.fit(self.X_train, y_train_encoded)
                    y_pred_encoded = model.predict(self.X_val)
                    
                    # Transform predictions back to original labels
                    y_pred = label_encoder.inverse_transform(y_pred_encoded)
                    
                    best_model_config = estimator.best_model()
                    
                    # # Log learner parameters
                    # learner = best_model_config.get('learner', {})
                    # if learner:
                    #     learner_params = learner.get_params()
                    #     mlflow.log_params({
                    #         f"learner_{key}": value 
                    #         for key, value in learner_params.items()
                    #     })
                    
                    # # Log preprocessor parameters
                    # preprocs = best_model_config.get('preprocs', ())
                    # for i, preproc in enumerate(preprocs):
                    #     preproc_params = preproc.get_params()
                    #     mlflow.log_params({
                    #         f"preproc_{i}_{key}": value 
                    #         for key, value in preproc_params.items()
                    #     })
                    
                    # # Log any extra preprocessor parameters
                    # ex_preprocs = best_model_config.get('ex_preprocs', ())
                    # for i, ex_preproc in enumerate(ex_preprocs):
                    #     ex_preproc_params = ex_preproc.get_params()
                    #     mlflow.log_params({
                    #         f"ex_preproc_{i}_{key}": value 
                    #         for key, value in ex_preproc_params.items()
                    #     })
                    
                    # Log best model configuration to MLflow
                    mlflow.log_params({
                        "best_model_config": str(best_model_config),
                        # "best_model_learner": str(best_model_config.get('learner', '')),
                        # "best_model_preprocessors": str(best_model_config.get('preprocs', '')),
                    })
                    
                    current.card.append(Markdown(
                        f"### Best Model Configuration\n" +
                        f"```python\n{best_model_config}\n```\n"
                    ))

                else:
                    print(f"[{model_name}] Using default parameters...")
                    params = model_mappings[model_name]['default_params']
                    model = model_config['model'](**params)
                    
                    model.fit(self.X_train, y_train_encoded)
                    y_pred_encoded = model.predict(self.X_val)
                    
                    y_pred = label_encoder.inverse_transform(y_pred_encoded)

                # Calculate training time
                training_time = time.time() - start_time
                print(f"[{model_name}] Training completed in {training_time:.2f} seconds")

                val_acc = accuracy_score(self.y_val, y_pred)

                # Log metrics including training time
                mlflow.log_params({
                    "method": "HyperoptEstimator" if self.USE_HYPEROPT else "default_parameters",
                    **model_mappings[model_name]['default_params']
                })
                
                if self.USE_HYPEROPT:
                    mlflow.log_params({
                        "label_mapping": str(dict(zip(
                            label_encoder.classes_,
                            label_encoder.transform(label_encoder.classes_)
                        )))
                    })
                    
                mlflow.log_metrics({
                    "validation_accuracy": val_acc,
                    "training_time_seconds": training_time
                })

                # Add to card with timing information
                current.card.append(Markdown(
                    f"## Model: {model_name}\n" +
                    f"* Method: {'HyperoptEstimator' if self.USE_HYPEROPT else 'Default Parameters'}\n" +
                    f"* Default Parameters: {model_mappings[model_name]['default_params']}\n" +
                    f"* Training Time: {training_time:.2f} seconds\n" +
                    f"* Validation Accuracy: {val_acc:.4f}\n"
                ))

                self.trained_models.append((model_name, model, val_acc, training_time))

            except Exception as e:
                print(f"[{model_name}] Error in training: {e}")
                raise

        self.next(self.join)

        
    @card    
    @resources(cpu=8, memory=4000)  
    @step
    def join(self, inputs):
        '''
        Compare model performances and select the best model.
        '''
        # Collect all models and their metrics
        self.trained_models = [input.trained_models for input in inputs]
        self.trained_models = [item for sublist in self.trained_models for item in sublist]  # Flatten the list
        
        # Sort models by validation accuracy
        sorted_models = sorted(self.trained_models, key=lambda x: x[2], reverse=True)
        best_model = sorted_models[0]
        
        # Start MLflow run for model comparison
        with mlflow.start_run(run_name="model_comparison") as run:
            # Log best model metrics
            mlflow.log_metrics({
                "best_model_accuracy": best_model[2],
                "best_model_training_time": best_model[3]
            })
            mlflow.log_param("best_model_name", best_model[0])
            
            # Create comparison visualization with timing information
            model_results = pd.DataFrame({
                'Model': [name for name, _, acc, time in sorted_models],
                'Accuracy': [acc for _, _, acc, time in sorted_models],
                'Training_Time': [time for _, _, acc, time in sorted_models]
            })

            # Add detailed metrics to card
            current.card.append(Markdown(
                "## Model Performance Comparison\n"
                f"### Best Model: {best_model[0]}\n"
                f"* Validation Accuracy: {best_model[2]:.4f}\n"
                f"* Training Time: {best_model[3]:.2f} seconds\n\n"
                "### All Models Performance:\n" +
                "\n".join([f"* {name}: {acc:.4f} ({time:.2f}s)" 
                          for name, _, acc, time in sorted_models])
            ))

            # Add bar chart visualization with color coding by training time
            current.card.append(
                VegaChart({
                    "data": {"values": model_results.to_dict(orient="records")},
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": "Model", "type": "nominal"},
                        "y": {
                            "field": "Accuracy",
                            "type": "quantitative",
                            "scale": {"domain": [0, 1]}
                        },
                        "color": {
                            "field": "Training_Time",
                            "type": "quantitative",
                            "title": "Training Time (s)"
                        }
                    },
                    "title": "Model Performance Comparison"
                })
            )

            # Log the best model
            print('Logging best model...')
            signature = infer_signature(inputs[0].X_train, inputs[0].y_val)
            mlflow.sklearn.log_model(
                best_model[1],
                f"best_model_{best_model[0]}",
                signature=signature,
                input_example=inputs[0].X_train[:5],
                registered_model_name="sk-learn-best-model",
            )
            print(f"Best model {best_model[0]} logged successfully.")

        self.next(self.end)

    # https://github.com/scikit-learn/scikit-learn/issues/12297
    # can't use sklearn voting classifier for pre-fitted models
    
    # @card    
    # @resources(cpu=8, memory=4000)  
    # @step
    # def join(self, inputs):
    #     '''
    #     Join the results of the parallel training steps.
    #     '''
    #     self.trained_models = [input.trained_models for input in inputs]
    #     self.trained_models = [item for sublist in self.trained_models for item in sublist]  # Flatten the list
    #     print(f"Trained models: {self.trained_models} \n")
    #     # Ensure variables are passed to the next step
    #     self.X_train = inputs[0].X_train  # Assuming all steps have the same data
    #     self.X_val = inputs[0].X_val
    #     self.y_train = inputs[0].y_train
    #     self.y_val = inputs[0].y_val
    #     self.next(self.ensemble)
    
    # @card
    # @resources(cpu=8, memory=4000)  
    # @step
    # def ensemble(self):
    #     '''
    #     Perform manual soft-voting ensemble using trained models.
    #     '''
    #     import numpy as np

    #     # Start MLflow run for ensemble model
    #     with mlflow.start_run(run_name="ensemble_model") as run:
            
    #         eclf = VotingClassifier(
    #             estimators=[(name, model) for name, model, _ in self.trained_models],
    #             voting='soft'
    #         )
    #         eclf.fit(self.X_train, self.y_train)
    #         y_pred = eclf.predict(self.X_val)
    #         final_accuracy = accuracy_score(self.y_val, y_pred)
            
    #         # Log ensemble metrics
    #         mlflow.log_metric("ensemble_accuracy", final_accuracy)
            
    #         # Infer model signature for ensemble
    #         signature = infer_signature(self.X_train, y_pred)
            
    #         # Log ensemble model
    #         mlflow.sklearn.log_model(
    #             eclf, 
    #             "ensemble_model",
    #             signature=signature,
    #             input_example=self.X_train[:5]
    #         )

    #         # Update metaflow card
    #         current.card.append(Markdown(
    #             "## Ensemble Model Results\n" +
    #             f"* Method: Manual Soft Voting\n" +
    #             f"* Base Models: {[name for name, _, _ in self.trained_models]}\n" +
    #             f"* Validation Accuracy: {final_accuracy:.4f}\n"
    #         ))

    #         # Add comparison visualization
    #         model_results = pd.DataFrame({
    #             'Model': [name for name, _, acc in self.trained_models] + ['Ensemble'],
    #             'Accuracy': [acc for _, _, acc in self.trained_models] + [final_accuracy]
    #         })

    #         current.card.append(
    #             VegaChart({
    #                 "data": {"values": model_results.to_dict(orient="records")},
    #                 "mark": "bar",
    #                 "encoding": {
    #                     "x": {"field": "Model", "type": "nominal"},
    #                     "y": {
    #                         "field": "Accuracy",
    #                         "type": "quantitative",
    #                         "scale": {"domain": [0, 1]}
    #                     }
    #                 },
    #                 "title": "Model Performance Comparison"
    #             })
    #         )

    #     self.next(self.end)


    @card    
    @step
    def end(self):
        '''
        Final step to conclude the pipeline.
        '''
        print("Pipeline completed successfully.")

if __name__ == '__main__':
    MLFlowPipeline()
