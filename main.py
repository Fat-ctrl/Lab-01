from metaflow import FlowSpec, step, card, environment, Parameter, current, resources
import numpy as np
import pandas as pd
import os
from datetime import datetime

# For MLflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import datasets
import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import cross_val_score, KFold
from mlflow.data.pandas_dataset import PandasDataset

# For drawing plot
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")  #using style ggplot
import plotly.graph_objects as go
import plotly.express as px
from metaflow.cards import VegaChart
from metaflow.cards import Markdown

# hyperparameters
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

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
    
    @card
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
    @step
    def start(self):
        '''
        Set up the MLflow tracking URI and define the models to be trained.
        '''
        if not isinstance(self.DATA_DIR, str) or not isinstance(self.DATASET_URL, str):
            raise TypeError("DATA_DIR and DATASET_URL must be strings")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        
        # Define model configurations with hyperparameter search spaces
        self.model_configs = [
            {
                'name': 'dt',
                'model': DecisionTreeClassifier,
                'space': {
                    'max_depth': hp.choice('dt_max_depth', range(1, 20)),
                    'min_samples_split': hp.choice('dt_min_samples_split', range(2, 10)),
                    'min_samples_leaf': hp.choice('dt_min_samples_leaf', range(1, 5))
                }
            },
            {
                'name': 'knn',
                'model': KNeighborsClassifier,
                'space': {
                    'n_neighbors': hp.choice('knn_n_neighbors', range(1, 20)),
                    'weights': hp.choice('knn_weights', ['uniform', 'distance']),
                    'p': hp.choice('knn_p', [1, 2])
                }
            },
            {
                'name': 'svc',
                'model': SVC,
                'space': {
                    'C': hp.loguniform('svc_C', np.log(0.01), np.log(100)),
                    'gamma': hp.loguniform('svc_gamma', np.log(0.0001), np.log(1)),
                    'kernel': hp.choice('svc_kernel', ['rbf', 'linear']),
                    'probability': True
                }
            }
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
    @resources(cpu=8, memory=4000)  # 2 CPU cores, 4GB memory
    @step
    def train_models(self):
        '''
        Do parallel training of models using MLflow with optional hyperparameter optimization.
        '''
        self.trained_models = []
        model_config = self.input
        
        # Define cross-validation strategy
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Start MLflow run for individual model
        with mlflow.start_run(run_name=f"model_{model_config['name']}") as run:
            try:
                if self.USE_HYPEROPT:
                    # Hyperparameter optimization code
                    def objective(params):
                        try:
                            if model_config['name'] == 'svc':
                                params['probability'] = True
                            model = model_config['model'](**params)
                            cv_scores = cross_val_score(
                                model, self.X_train, self.y_train, 
                                cv=cv, scoring='accuracy'
                            )
                            return {'loss': -cv_scores.mean(), 'status': STATUS_OK}
                        except Exception as e:
                            print(f"Error in objective function: {e}")
                            return {'loss': float('inf'), 'status': STATUS_OK}

                    # Run optimization
                    trials = Trials()
                    best = fmin(
                        fn=objective,
                        space=model_config['space'],
                        algo=tpe.suggest,
                        max_evals=50,
                        trials=trials,
                        show_progressbar=True
                    )

                    # Map hyperparameters
                    space = model_config['space']
                    best_params = {}
                    for param_name, param_value in best.items():
                        if hasattr(space[param_name], 'choices'):
                            best_params[param_name] = space[param_name].choices[param_value]
                        else:
                            best_params[param_name] = param_value

                    # Log optimization results
                    trials_data = pd.DataFrame({
                        'Trial': range(1, len(trials.trials) + 1),
                        'Accuracy': [-t['result']['loss'] for t in trials.trials if 'loss' in t['result']]
                    })
                    
                    current.card.append(
                        VegaChart({
                            "data": {"values": trials_data.to_dict(orient="records")},
                            "mark": "line",
                            "encoding": {
                                "x": {"field": "Trial", "type": "quantitative"},
                                "y": {"field": "Accuracy", "type": "quantitative"}
                            },
                            "title": f"Hyperparameter Optimization History for {model_config['name']}"
                        })
                    )
                else:
                    # Use default parameters
                    best_params = {
                        'dt': {'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1},
                        'knn': {'n_neighbors': 5, 'weights': 'uniform', 'p': 2},
                        'svc': {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf', 'probability': True}
                    }[model_config['name']]

                # Ensure probability=True for SVC
                if model_config['name'] == 'svc':
                    best_params['probability'] = True

                # Train final model
                final_model = model_config['model'](**best_params)
                mlflow.log_params(best_params)
                
                # Evaluate model
                cv_scores = cross_val_score(
                    final_model, self.X_train, self.y_train, 
                    cv=cv, scoring='accuracy'
                )
                
                final_model.fit(self.X_train, self.y_train)
                y_pred = final_model.predict(self.X_val)
                val_acc = accuracy_score(self.y_val, y_pred)
                
                # Log metrics
                mlflow.log_metrics({
                    "cv_mean_accuracy": cv_scores.mean(),
                    "cv_std_accuracy": cv_scores.std(),
                    "validation_accuracy": val_acc
                })
                
                # Add summary to card
                current.card.append(Markdown(
                    f"## Model: {model_config['name']}\n" +
                    f"* Hyperparameter Optimization: {'Enabled' if self.USE_HYPEROPT else 'Disabled'}\n" +
                    f"* Parameters: {best_params}\n" +
                    f"* Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n" +
                    f"* Validation Accuracy: {val_acc:.4f}\n"
                ))
                
                self.trained_models.append((model_config['name'], final_model, val_acc))
                
            except Exception as e:
                print(f"Error in model training: {e}")
                raise
            
        self.next(self.join)

    @card    
    @resources(cpu=8, memory=4000)  
    @step
    def join(self, inputs):
        '''
        Join the results of the parallel training steps.
        '''
        self.trained_models = [input.trained_models for input in inputs]
        self.trained_models = [item for sublist in self.trained_models for item in sublist]  # Flatten the list
        # Ensure variables are passed to the next step
        self.X_train = inputs[0].X_train  # Assuming all steps have the same data
        self.X_val = inputs[0].X_val
        self.y_train = inputs[0].y_train
        self.y_val = inputs[0].y_val
        self.next(self.ensemble)

    @card
    @resources(cpu=8, memory=8000)  
    @step
    def ensemble(self):
        '''
        Train an ensemble model using the trained models.
        '''
        # Start MLflow run for ensemble model
        with mlflow.start_run(run_name="ensemble_model") as run:
            voting_clf = VotingClassifier(
                estimators=[(name, model) for name, model, _ in self.trained_models],
                voting='soft'
            )
            
            # Train and evaluate ensemble
            voting_clf.fit(self.X_train, self.y_train)
            y_pred = voting_clf.predict(self.X_val)
            final_accuracy = accuracy_score(self.y_val, y_pred)
            
            # Log ensemble metrics
            mlflow.log_metric("ensemble_accuracy", final_accuracy)
            
            # Infer model signature for ensemble
            signature = infer_signature(self.X_train, y_pred)
            
            # Log ensemble model with signature and input example
            mlflow.sklearn.log_model(
                voting_clf, 
                "ensemble_model",
                signature=signature,
                input_example=self.X_train[:5]
            )
            
            # Add ensemble results to card
            current.card.append(Markdown(
                "## Ensemble Model Results\n" +
                f"* Model Type: Soft Voting Classifier\n" +
                f"* Base Models: {[name for name, _, _ in self.trained_models]}\n" +
                f"* Validation Accuracy: {final_accuracy:.4f}\n"
            ))
            
            # Add comparison visualization
            model_results = pd.DataFrame({
                'Model': [name for name, _, acc in self.trained_models] + ['Ensemble'],
                'Accuracy': [acc for _, _, acc in self.trained_models] + [final_accuracy]
            })
            
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
                        }
                    },
                    "title": "Model Performance Comparison"
                })
            )
            
        self.next(self.end)

    @card    
    @step
    def end(self):
        '''
        Final step to conclude the pipeline.
        '''
        print("Pipeline completed.")

if __name__ == '__main__':
    MLFlowPipeline()
