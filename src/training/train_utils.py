import torch
import torch.optim as optim
import copy

def train_model(
        model, 
        train_loader, 
        test_loader, 
        optimizer, 
        criterion, 
        num_epochs, 
        device, 
        patience=50000
    ):
    model.train()
    history = {}

    best_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        running_loss = 0.0

        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
        test_loss /= len(test_loader.dataset)
        history[epoch+1] = {'loss': epoch_loss, 'test_loss': test_loss}
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        if patience is not None:
            if test_loss < best_loss:
                # Found new best model; save it
                best_loss = test_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve == patience:
                print(f"Early stopping on epoch {epoch+1} with best test loss {best_loss:.4f}.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(test_loader.dataset)
    print(f"Final Test Loss: {avg_loss:.4f}")
    return avg_loss

def save_model(model, path):
    """Save the trained model."""
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    """Load a saved model."""
    model.load_state_dict(torch.load(path, map_location=device))
    return model
