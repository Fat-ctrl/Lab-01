import optuna
import plotly.graph_objects as go
from datetime import datetime

def log_optimization_results(study, model_name):
    """Log optimization results and create visualizations."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"train/optimization_logs/{model_name}_{timestamp}.log"
    
    # Create optimization history plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[t.number for t in study.trials],
        y=[t.value for t in study.trials],
        mode='markers+lines',
        name='Trial values'
    ))
    fig.update_layout(
        title=f'{model_name} Optimization History',
        xaxis_title='Trial number',
        yaxis_title='Accuracy'
    )
    fig.write_html(f"train/optimization_logs/{model_name}_history_{timestamp}.html")
    
    # Log best trial information
    with open(log_file, 'w') as f:
        f.write(f"Best trial for {model_name}:\n")
        f.write(f"  Value: {study.best_trial.value}\n")
        f.write(f"  Params: {study.best_trial.params}\n")
        f.write("\nParameter importance:\n")
        importance = optuna.importance.get_param_importances(study)
        for param, score in importance.items():
            f.write(f"  {param}: {score:.3f}\n")