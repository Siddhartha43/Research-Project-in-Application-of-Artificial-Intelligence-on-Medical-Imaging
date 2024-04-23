import argparse
import datetime
import matplotlib.pyplot as plt
import torch
import time
import torch_optimizer as optim

from models import EEGNet, DeepConvNet
from dataloader import read_bci_data

from rich.progress import Progress, TaskID, track
from tqdm import tqdm
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR



def load_data(batch_size=64):
    train_data, train_label, test_data, test_label = read_bci_data()
    train_dataset = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_label).long())
    test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_label).long())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def create_model(model_name, activation_func, dropout, lr, gamma, device='cpu'):
    if model_name == 'EEGNet':
        model = EEGNet(activation_func=activation_func, dropout=dropout).to(device)
    elif model_name == 'DeepConvNet':
        model = DeepConvNet(activation_func=activation_func, dropout=dropout).to(device)
    else:
        raise ValueError(f"Invalid model_name: {model_name}")

    #optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    return model, optimizer, scheduler

def train_model(model, optimizer, scheduler, train_loader, test_loader, epochs, device, model_name, activation):
    criterion = nn.CrossEntropyLoss()

    # Initialize dictionaries to hold accuracy values
    train_accuracies = {f"{model_name}_{activation}": []}
    test_accuracies = {f"{model_name}_{activation}": []}

    progress = Progress()
    task = progress.add_task("[cyan]Training...", total=epochs)

    with progress:
        for epoch in range(epochs):
            model.train()
            train_correct = 0
            train_total = 0
            total_loss = 0  # Initialize total loss
            
            for i, (data, labels) in enumerate(train_loader):
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()  # Accumulate loss

                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            average_loss = total_loss / len(train_loader)  # Calculate average loss
            train_accuracy = round(train_correct / train_total, 2)
            train_accuracies[f"{model_name}_{activation}"].append(train_accuracy)

            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

            test_accuracy = round(test_correct / test_total, 2)
            test_accuracies[f"{model_name}_{activation}"].append(test_accuracy)
            
            progress.update(task, advance=1, description=f"Epoch: {epoch+1}, Training Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}, Loss: {average_loss:.4f}")
            
            scheduler.step()

    # Calculate the highest accuracies
    highest_train_accuracy = max(train_accuracies[f"{model_name}_{activation}"])
    highest_test_accuracy = max(test_accuracies[f"{model_name}_{activation}"])

    return train_accuracies, test_accuracies, highest_train_accuracy, highest_test_accuracy


def plot_results(model_name, activations, accuracies, epochs):
    plt.figure(figsize=(10, 6))
    for activation in activations:
        # Convert accuracies to percentage
        train_accuracies = [acc * 100 for acc in accuracies[f"{model_name}_{activation}_Train"]]
        test_accuracies = [acc * 100 for acc in accuracies[f"{model_name}_{activation}_Test"]]

        line1, = plt.plot(range(epochs), train_accuracies, label=f'{activation}_Train')  # Don't set linestyle here
        line1.set_dashes([2, 2, 10, 2])  # Use set_dashes() to create dashed line
        plt.plot(range(epochs), test_accuracies, label=f'{activation}_Test')
    plt.title(f'Activation Function Comparison {model_name} ')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')  # Change y-axis label to 'Accuracy (%)'
    plt.legend(loc='lower right')
    plt.grid(True)

    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    
    plt.savefig(f"{model_name}_comparison_{timestamp}.png",dpi=300)  # Save the figure to a file with a timestamp
    plt.show()  # Move plt.show() after plt.savefig()

def train_and_collect_accuracies(model_name, activations, args, device, train_loader, test_loader):
    highest_train_accuracies = {}
    highest_test_accuracies = {}
    all_accuracies = {}

    for activation in activations:
        print(f"Training {model_name} with {activation} activation...")
        model, optimizer, scheduler = create_model(model_name, activation, args.dropout, args.lr, args.gamma, device)
        train_accuracies, test_accuracies, highest_train_accuracy, highest_test_accuracy = train_model(model, optimizer, scheduler, train_loader, test_loader, args.epochs, device, model_name, activation)
        
        # Store the highest train and test accuracies for this model
        highest_train_accuracies[f"{model_name}_{activation}"] = highest_train_accuracy
        highest_test_accuracies[f"{model_name}_{activation}"] = highest_test_accuracy
        
        # Store all accuracies for plotting
        all_accuracies[f"{model_name}_{activation}_Train"] = train_accuracies[f"{model_name}_{activation}"]
        all_accuracies[f"{model_name}_{activation}_Test"] = test_accuracies[f"{model_name}_{activation}"]
    
    return highest_train_accuracies, highest_test_accuracies, all_accuracies

def main():
    parser = argparse.ArgumentParser(description='PyTorch BCI Example')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dropout', type=float, default=0.25, metavar='D',
                        help='Dropout rate (default: 0.25)')
    parser.add_argument('--activation', type=str, choices=['relu', 'leaky_relu', 'elu'], default=None, 
                    help='activation function for the model (default: None, which means all activation functions)')
    parser.add_argument('--model', type=str, default='all', metavar='MODEL',
                    help='model to use (EEGNet or DeepConvNet or all)')
    args = parser.parse_args()

    start_time = time.time()  # Record the start time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(args.batch_size)

    # Define the models and activation functions
    if args.model != 'all':
        models = [args.model]  # 只訓練指定的模型
    else:
        models = ["EEGNet", "DeepConvNet"]  # 沒有指定模型時，訓練所有模型
    activations = [args.activation] if args.activation else ["relu", "leaky_relu", "elu"]
    
    # Store the highest accuracy of each model with each activation function
    highest_train_accuracies = {}
    highest_test_accuracies = {}

    for model_name in models:
        highest_train_accuracy, highest_test_accuracy, all_accuracies = train_and_collect_accuracies(model_name, activations, args, device, train_loader, test_loader)
        plot_results(model_name, activations, all_accuracies, args.epochs)

        # Store the highest accuracy
        highest_train_accuracies.update(highest_train_accuracy)
        highest_test_accuracies.update(highest_test_accuracy)

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time

    # Print the highest accuracy of each model with each activation function
    print("\nHighest Training Accuracies:")
    for (model_activation, highest_accuracy) in highest_train_accuracies.items():
        print(f"{model_activation}: {highest_accuracy:.2f}")

    print("\nHighest Test Accuracies:")
    for (model_activation, highest_accuracy) in highest_test_accuracies.items():
        print(f"{model_activation}: {highest_accuracy:.2f}")

    print(f"\nTotal elapsed time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()
