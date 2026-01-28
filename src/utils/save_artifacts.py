# save_artifacts.py
import os
from datetime import datetime
import json
import torch
import pickle

# def generate_filename(model_type, learning_rate, batch_size, num_epochs, weight_decay, additional_params=None, extension='pth'):
#     """
#     Generate a unique filename based on model parameters and timestamp.
    
#     Args:
#         model_type (str): Type of model ('NN' or 'HNN', for example).
#         learning_rate (float): Learning rate used.
#         batch_size (int): Batch size used.
#         num_epochs (int): Number of training epochs.
#         additional_params (dict, optional): Additional parameters (e.g., hidden_size, num_layers).
#         extension (str): File extension (default is 'pth').
    
#     Returns:
#         str: Generated filename.
#     """
#     # Create base string
#     now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"{model_type}_lr{learning_rate}_bs{batch_size}_epochs{num_epochs}_wgtdecay{weight_decay}"
    
#     if additional_params:
#         for key, value in additional_params.items():
#             filename += f"_{key}{value}"
    
#     filename += f"_{now_str}.{extension}"
#     return filename

def generate_filename(model_type, learning_rate, batch_size, num_epochs, weight_decay, extension='pth', **kwargs):
    """
    Generate a unique filename based on model parameters and timestamp.
    
    Args:
        model_type (str): Type of model.
        learning_rate (float): Learning rate used.
        batch_size (int): Batch size used.
        num_epochs (int): Number of training epochs.
        weight_decay (float): Weight decay used.
        extension (str): File extension.
        **kwargs: Any additional model parameters.
    
    Returns:
        str: Generated filename.
    """
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_type}_lr{learning_rate}_bs{batch_size}_epochs{num_epochs}_wgtdecay{weight_decay}"
    
    # Append any additional keyword arguments.
    for key, value in kwargs.items():
        filename += f"_{key}{value}"
    
    filename += f"_{now_str}.{extension}"
    return filename


def save_model_and_history(model, history, model_params, model_dir, history_dir):
    """
    Saves both the model and the training history using dynamically generated filenames.
    
    Args:
        model (torch.nn.Module): Trained model.
        history (dict): Training history.
        model_params (dict): Dictionary containing parameters: 
            - model_type (str)
            - learning_rate (float)
            - batch_size (int)
            - num_epochs (int)
            Optionally include any additional params.
        model_dir (str): Directory where the model will be saved.
        history_dir (str): Directory where the training history will be saved.
    """
    # Create directories if they do not exist.
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    
    # Generate filenames for the model and history JSON.
    model_filename = generate_filename(**model_params, extension='pth')
    history_filename = generate_filename(**model_params, extension='json')
    
    model_path = os.path.join(model_dir, model_filename)
    history_path = os.path.join(history_dir, history_filename)
    
    # Save the model's state dict.
    torch.save(model.state_dict(), model_path)
    
    # Save the training history to JSON.
    with open(history_path, 'w') as json_file:
        json.dump(history, json_file, indent=4)
        
    print(f"Model saved to: {model_path}")
    print(f"Training history saved to: {history_path}")

def save_scalers(scaler_X, scaler_y, model_params, scaler_dir):
    """
    Save both X and Y scalers together as a pickle file.
    """
    os.makedirs(scaler_dir, exist_ok=True)
    scalers_filename = generate_filename(**model_params, extension='pkl')
    scalers_path = os.path.join(scaler_dir, scalers_filename)
    
    scalers = {'scaler_X': scaler_X, 'scaler_y': scaler_y}
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
        
    print(f"Scalers saved to: {scalers_path}")

def generate_experiment_prefix(model_type, learning_rate, batch_size, num_epochs, weight_decay, **kwargs):
    """
    Generates a unique filename *prefix* based on model parameters,
    *without* the timestamp or extension.
    """
    prefix = f"{model_type}_lr{learning_rate}_bs{batch_size}_epochs{num_epochs}_wgtdecay{weight_decay}"
    
    # Append any additional keyword arguments (like hidden_size, etc.)
    for key, value in kwargs.items():
        prefix += f"_{key}{value}"
    
    return prefix

def check_if_experiment_exists(prefix, directory):
    """
    Checks if any file in the given directory starts with the specified prefix.
    """
    # Ensure directory exists to avoid FileNotFoundError
    if not os.path.exists(directory):
        # If the directory doesn't exist, no experiments have been saved yet.
        os.makedirs(directory, exist_ok=True)
        return False
        
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            return True
    return False