import matplotlib.pyplot as plt

def plot_training_history(history, title_prefix='Model'):
    """Plots train vs test loss given a history dictionary from train_model."""
    # Extract epochs (keys) and their corresponding losses
    epochs = list(history.keys())
    train_loss = [history[e]['loss'] for e in epochs]
    test_loss = [history[e]['test_loss'] for e in epochs]

    plt.figure()  # (1) each chart on its own distinct plot
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss') 
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'{title_prefix} Training History')
    plt.legend()
    plt.show()