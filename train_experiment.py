# train_experiments.py

import os
import itertools
import pandas as pd
import torch
import torch.optim as optim
from datetime import datetime

# Import your custom modules.
from data_preprocessing import load_data, get_dataloaders
from models import NeuralNet, HybridNeuralNet
from training import train_model, evaluate_model
# from utils.save_artifacts import save_model_and_history,save_scalers
from utils.save_artifacts import save_model_and_history, save_scalers, generate_experiment_prefix, check_if_experiment_exists

def get_model(model_type, input_size, **kwargs):
    """
    Returns an instance of the model based on model_type.
    """
    if model_type == "NN":
        return NeuralNet(input_size=input_size,
                         hidden_size=kwargs.get("hidden_size", 8),
                         num_hidden_layers=kwargs.get("num_hidden_layers", 3))
    elif model_type == "HNN":
        return HybridNeuralNet(input_size=input_size,
                               hidden_size=kwargs.get("hidden_size", 8),
                               num_hidden_layers=kwargs.get("num_hidden_layers", 3))
    else:
        raise ValueError("Unknown model type: {}".format(model_type))

def run_experiment(model_type, file_path, input_columns, epochs, batch_size, lr, weight_decay, additional_model_params=None, device='cpu'):
    """
    Runs one complete training experiment:
      - Loads and preprocesses data.
      - Instantiates the specified model.
      - Trains and evaluates it.
      - Saves the model and training history.
    
    Returns:
      - final loss
      - history dict (detailed training loss for each epoch)
    """
    # Load data for the given input columns.
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_data(file_path, input_columns=input_columns)
    train_loader, test_loader = get_dataloaders(X_train, y_train, X_test, y_test, batch_size=batch_size)
    
    input_size = X_train.shape[1]
    # Instantiate the model.
    model = get_model(model_type, input_size, **(additional_model_params or {})).to(device)
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training {model_type} with lr={lr}, batch_size={batch_size}, epochs={epochs}")
    
    # Train model and capture training history
    history = train_model(model, train_loader, test_loader, optimizer, criterion, epochs, device)
    final_loss = evaluate_model(model, test_loader, criterion, device)
    
    # Create a dictionary of model parameters to be included in filenames.
    model_params = {
        "model_type": model_type,
        "learning_rate": lr,
        "batch_size": batch_size,
        "num_epochs": epochs,
        "weight_decay": weight_decay
    }
    if additional_model_params:
        model_params.update(additional_model_params)
        
    # Define directories for saving artifacts (modify as needed).
    model_dir = os.path.join(os.getcwd(), "src", "saved_models")
    history_dir = os.path.join(os.getcwd(), "src", "saved_training_history")
    scaler_dir = os.path.join(os.getcwd(), "src", "saved_scalers")
    
    # Save model and training history.
    save_model_and_history(model, history, model_params, model_dir, history_dir)
    save_scalers(scaler_X, scaler_y, model_params, scaler_dir)

    return final_loss, history

if __name__ == "__main__":
    # Global settings.
    FILE_PATH = r'C:\nano_optics_ml_data\processed\article_main_data.csv'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    MODEL_DIR = os.path.join(os.getcwd(), "src", "saved_models")
    RESULTS_FILE = os.path.join(os.getcwd(), "experiment_results.csv")

    # Hyperparameter grid.
    # model_types = ["NN", "HNN"]
    model_types = ["NN","HNN"]
    learning_rates = [1e-3, 1e-4]
    batch_sizes = [8, 32, 256, 512]
    epochs_list = [500, 5000, 7000]
    weight_decays = [1e-5, 1e-4, 1e-3]
    # Additional architecture parameters.
    hidden_sizes = [8, 16]
    num_hidden_layers_list = [3]
    
    print(f"--- Starting Grid Search ---")
    print(f"Results will be saved to: {RESULTS_FILE}")
    print(f"Models will be saved to: {MODEL_DIR}")

    # Loop over all combinations.
    for model_type in model_types:
        # For each model type, choose the right input columns.
        if model_type == "NN":
            input_columns = ['beta1', 'beta2']
        else:  # HNN
            input_columns = ['beta1', 'beta2', 'Sn_fdm_real', 'Sn_fdm_imag']
            
        for lr in learning_rates:
            for batch_size in batch_sizes:
                if batch_size<=1200:
                    DEVICE='cpu'
                    print(f"Using device: {DEVICE}")
                for epochs in epochs_list:
                    for weight_decay in weight_decays:
                        for hidden_size in hidden_sizes:
                            for num_hidden_layers in num_hidden_layers_list:

                                # 1. Collate all parameters for this run
                                current_params = {
                                    "model_type": model_type,
                                    "learning_rate": lr,
                                    "batch_size": batch_size,
                                    "num_epochs": epochs,
                                    "weight_decay": weight_decay,
                                    "hidden_size": hidden_size,
                                    "num_hidden_layers": num_hidden_layers
                                }
                                
                                # 2. Generate the unique prefix for these parameters
                                experiment_prefix = generate_experiment_prefix(**current_params)
                                
                                # 3. Check if a model with this prefix already exists
                                if check_if_experiment_exists(experiment_prefix, MODEL_DIR):
                                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SKIPPING existing experiment: {experiment_prefix}")
                                    continue # Skip to the next iteration
                                
                                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] RUNNING: {experiment_prefix}")
                                print(f"Using device: {DEVICE}")

                                additional_model_params = {
                                    "hidden_size": hidden_size,
                                    "num_hidden_layers": num_hidden_layers
                                }
                                # Execute a single experiment.
                                final_loss, history = run_experiment(
                                    model_type=model_type,
                                    file_path=FILE_PATH,
                                    input_columns=input_columns,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    lr=lr,
                                    weight_decay=weight_decay,
                                    additional_model_params=additional_model_params,
                                    device=DEVICE
                                )
                                experiment_info = {
                                    "model_type": model_type,
                                    "learning_rate": lr,
                                    "batch_size": batch_size,
                                    "epochs": epochs,
                                    "weight_decay": weight_decay,
                                    "hidden_size": hidden_size,
                                    "num_hidden_layers": num_hidden_layers,
                                    "final_loss": final_loss
                                }
                                df_to_append = pd.DataFrame([experiment_info])
                                
                                # Check if file exists to determine if we need to write headers
                                file_exists = os.path.exists(RESULTS_FILE)
                                
                                df_to_append.to_csv(
                                    RESULTS_FILE, 
                                    mode='a', #append
                                    header=not file_exists, # Write header only if file does *not* exist
                                    index=False
                                )
                                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Result saved for: {experiment_prefix}")
                                
        
    print(f"\n--- Grid Search Finished ---")
    print(f"All results saved to: {RESULTS_FILE}")
