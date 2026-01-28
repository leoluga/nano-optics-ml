import os
import glob
import pickle
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime

# --- Import your custom modules ---
# We need the functions to load data, get dataloaders, and define models
from data_preprocessing import get_raw_data_split, get_dataloaders, CustomDataset
from models import NeuralNet, HybridNeuralNet
from training import evaluate_model

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

def parse_filename(model_filename):
    """
    Parses a model filename to extract the hyperparameters.
    
    Example: NN_lr1e-3_bs32_..._20251108_135000.pth
    Returns:
        - stem_with_timestamp (str): The filename *with* timestamp, without extension.
                                     (e.g., "NN_lr1e-3_..._20251108_135000")
        - params (dict): A dictionary of the parsed hyperparameters
    """
    basename = os.path.basename(model_filename)
    
    # --- THIS IS THE KEY ---
    # We get the full name *with* the timestamp, just without the .pth
    # This stem will match the scaler's filename perfectly.
    stem_with_timestamp = os.path.splitext(basename)[0] 
    
    parts = stem_with_timestamp.split('_')
    
    if len(parts) < 3: # Not a valid file (e.g., "model_type_timestamp.pth")
        print(f"  [Error] Invalid filename format: {basename}")
        return None, None
        
    # We parse the params *without* the timestamp
    param_parts = parts[:-2] # All parts except the date and time
    
    params = {}
    
    try:
        params['model_type'] = param_parts[0]
        
        # Loop through the key-value parts
        for part in param_parts[1:]:
            if part.startswith('lr'):
                params['learning_rate'] = float(part[2:])
            elif part.startswith('bs'):
                params['batch_size'] = int(part[2:])
            elif part.startswith('epochs'):
                params['num_epochs'] = int(part[6:])
            elif part.startswith('wgtdecay'):
                params['weight_decay'] = float(part[8:])
            elif part.startswith('hidden_size'):
                params['hidden_size'] = int(part[11:])
            elif part.startswith('num_hidden_layers'):
                params['num_hidden_layers'] = int(part[17:])
                
    except Exception as e:
        print(f"  [Error] Could not parse parameter from filename '{basename}': {e}")
        return None, None

    # --- THIS IS THE FIX ---
    # We return the stem *with* the timestamp, which we got in the first line.
    return stem_with_timestamp, params

def run_evaluation():
    """
    Main function to find, load, and re-evaluate all saved models.
    """
    
    # --- Configuration ---
    FILE_PATH = r'C:\nano_optics_ml_data\processed\article_main_data.csv'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Directories
    BASE_DIR = os.getcwd()
    MODEL_DIR = os.path.join(BASE_DIR, "src", "saved_models")
    SCALER_DIR = os.path.join(BASE_DIR, "src", "saved_scalers")
    
    # We'll save to a new file to avoid conflicts
    RESULTS_FILE = os.path.join(BASE_DIR, f"experiment_results_RE-EVALUATED_{datetime.now().strftime('%Y%m%d')}.csv")

    print(f"Using device: {DEVICE}")
    print(f"Loading models from: {MODEL_DIR}")
    print(f"Loading scalers from: {SCALER_DIR}")
    
    # --- Load Raw Test Data ---
    # We load both versions of the test data (for NN and HNN) once.
    print("Loading raw test data...")
    input_cols_NN = ['beta1', 'beta2']
    X_test_raw_NN, y_test_raw_NN = get_raw_data_split(FILE_PATH, input_columns=input_cols_NN)

    input_cols_HNN = ['beta1', 'beta2', 'Sn_fdm_real', 'Sn_fdm_imag']
    X_test_raw_HNN, y_test_raw_HNN = get_raw_data_split(FILE_PATH, input_columns=input_cols_HNN)
    print("Test data loaded.")

    # --- Find all model files ---
    model_files = glob.glob(os.path.join(MODEL_DIR, "*.pth"))
    if not model_files:
        print("No .pth models found. Exiting.")
        return

    print(f"\nFound {len(model_files)} models to evaluate...")
    
    all_results = []
    criterion = nn.MSELoss()

    # --- Loop and Evaluate ---
    for model_file in model_files:
        print(f"--- Processing: {os.path.basename(model_file)} ---")
        
        # 1. Parse filename to get params and scaler name
        stem, params = parse_filename(model_file)
        if not stem:
            continue
            
        # 2. Find and load the matching scaler
        scaler_file = os.path.join(SCALER_DIR, stem + ".pkl")
        if not os.path.exists(scaler_file):
            print(f"  [Warning] SKIPPING: No matching scaler file found at {scaler_file}")
            continue
        
        try:
            with open(scaler_file, 'rb') as f:
                scalers = pickle.load(f)
            scaler_X = scalers['scaler_X']
            scaler_y = scalers['scaler_y']
        except Exception as e:
            print(f"  [Error] SKIPPING: Could not load scaler file: {e}")
            continue

        # 3. Get the correct raw test data based on model type
        if params['model_type'] == 'NN':
            X_test_raw = X_test_raw_NN
            y_test_raw = y_test_raw_NN
            input_size = len(input_cols_NN)
        elif params['model_type'] == 'HNN':
            X_test_raw = X_test_raw_HNN
            y_test_raw = y_test_raw_HNN
            input_size = len(input_cols_HNN)
        else:
            print(f"  [Warning] SKIPPING: Unknown model type '{params['model_type']}'")
            continue

        # 4. Apply the *loaded* scaler to the raw data
        X_test_scaled = scaler_X.transform(X_test_raw)
        y_test_scaled = scaler_y.transform(y_test_raw)
        
        # 5. Create Tensors and DataLoader
        X_test_tensor = torch.from_numpy(X_test_scaled).float()
        y_test_tensor = torch.from_numpy(y_test_scaled).float()
        
        # Use get_dataloaders, but we only need the test_loader
        # We can create a simple one manually to avoid complexity
        test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.get('batch_size', 32))

        # 6. Instantiate model and load state
        try:
            model = get_model(params['model_type'], input_size, **params).to(DEVICE)
            model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        except Exception as e:
            print(f"  [Error] SKIPPING: Could not load model state: {e}")
            continue
        
        # 7. Evaluate the model
        final_loss = evaluate_model(model, test_loader, criterion, DEVICE)
        
        # 8. Store the result
        result_entry = params.copy()
        result_entry['final_loss'] = final_loss
        result_entry['evaluated_model_file'] = os.path.basename(model_file)
        all_results.append(result_entry)
        
        print(f"  [Success] Final Loss: {final_loss:.6f}")

    # --- Save All Results ---
    if not all_results:
        print("\nNo models were successfully evaluated.")
        return

    print(f"\nEvaluation complete. Saving {len(all_results)} results to {RESULTS_FILE}")
    results_df = pd.DataFrame(all_results)
    
    # Re-order columns for clarity
    cols_order = [
        'model_type', 'learning_rate', 'batch_size', 'num_epochs', 
        'weight_decay', 'hidden_size', 'num_hidden_layers', 
        'final_loss', 'evaluated_model_file'
    ]
    # Filter to only include columns that exist
    final_cols = [col for col in cols_order if col in results_df.columns]
    results_df = results_df[final_cols]
    
    results_df.to_csv(RESULTS_FILE, index=False)
    print("Done.")


if __name__ == "__main__":
    run_evaluation()