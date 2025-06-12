from sklearn.model_selection import cross_val_score
import numpy as np

def create_objective(model_class, X, y, hyperparameters):
    def objective(trial):
        params = {}
        
        # Generate parameters based on hyperparameter definitions
        for param_name, param_config in hyperparameters.items():
            if isinstance(param_config, tuple):
                if len(param_config) == 3 and param_config[2] == 'log':
                    # Log-scale parameter
                    params[param_name] = trial.suggest_float(
                        param_name, param_config[0], param_config[1], log=True
                    )
                elif isinstance(param_config[0], float):
                    # Float parameter
                    params[param_name] = trial.suggest_float(
                        param_name, param_config[0], param_config[1]
                    )
                else:
                    # Integer parameter
                    params[param_name] = trial.suggest_int(
                        param_name, param_config[0], param_config[1]
                    )
            elif isinstance(param_config, list):
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config
                )
        
        # Create and evaluate model
        model = model_class(**params)
        scores = cross_val_score(
            model, X, y, cv=5, scoring='accuracy', n_jobs=-1
        )
        return scores.mean()
    
    return objective