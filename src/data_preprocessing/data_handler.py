import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
def load_data(file_path, input_columns = ['beta1', 'beta2'], testsize = 0.2, remove_LSAT=True):
    """Load data for the NN model with features [beta1, beta2]."""
    df = pd.read_csv(file_path)
    if remove_LSAT:
        df = df.loc[(df['material'] != 'LSAT')].reset_index(drop=True)
    
    df = process_df_eps_to_beta(df)

    X = df[input_columns].values.astype(np.float32)
    y = df[['Sn_exp_real', 'Sn_exp_imag']].values.astype(np.float32)
    materials = df['material'].values

    X_train, X_test, y_train, y_test, _, _ = train_test_split(
        X, y, materials, test_size=testsize, random_state=42)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    return (
        torch.from_numpy(X_train_scaled).float(),
        torch.from_numpy(X_test_scaled).float(),
        torch.from_numpy(y_train_scaled).float(),
        torch.from_numpy(y_test_scaled).float(),
        scaler_X,   
        scaler_y    
    )


def load_xgboost_data(file_path, input_columns=['beta1', 'beta2'], testsize=0.2,remove_LSAT=True):
    """Load and preprocess data for XGBoost"""
    df = pd.read_csv(file_path)
    
    if remove_LSAT:
        df = df.loc[(df['material'] != 'LSAT')].reset_index(drop=True)
    
    df = process_df_eps_to_beta(df)

    X = df[input_columns].values.astype(np.float32)
    y = df[['Sn_exp_real', 'Sn_exp_imag']].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=42)

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_train, X_test = scaler_X.fit_transform(X_train), scaler_X.transform(X_test)
    y_train, y_test = scaler_y.fit_transform(y_train), scaler_y.transform(y_test)

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


def get_dataloaders(X_train, y_train, X_test, y_test, batch_size):
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def process_df_eps_to_beta(df_in: pd.DataFrame) -> pd.DataFrame:
    """Function that takes df with 'eps1' and 'eps2' columsn and returns the same df with columns for 'beta1' and 'beta2'"""
    assert all(col in df_in.columns for col in ['eps1', 'eps2']), 'Make sure the df has columns eps1 and eps2'

    df = df_in.copy()
    epsilon = df['eps1'].values + 1j * df['eps2'].values
    beta = (epsilon - 1) / (epsilon + 1)
    df['beta1'] = beta.real
    df['beta2'] = beta.imag

    return df