import os
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

import seaborn as sns
from matplotlib.ticker import MaxNLocator

import torch
import torchvision

def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if (int(outputs[j]) == 1 and int(labels[j]) == 1):
            tp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 0):
            tn += 1
        if (int(outputs[j]) == 1 and int(labels[j]) == 0):
            fp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 1):
            fn += 1
    return tp, tn, fp, fn

def plot_accuracy(train_acc_list, test_acc_list, val_acc_list, model_type,  title="Training vs. Testing vs. Validation Accuracy"):
  """
  This function plots the training and testing accuracy curves.

  Args:
      train_acc_list: A list of training accuracy values for each epoch.
      test_acc_list: A list of testing accuracy values for each epoch.
      title: The title of the plot (default: "Training vs. Testing Accuracy").
  """
  plt.figure(figsize=(10, 6))
  plt.plot(train_acc_list, label='Training Accuracy')
  plt.plot(test_acc_list, label='Testing Accuracy')
  plt.plot(val_acc_list, label='Validation Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.title(title)
  plt.legend()
  plt.grid(True)
  plt.show()
  filename = f"{model_type}_accuracy.png"
  plt.savefig(filename)

def plot_f1_score(f1_score_list, val_f1_score_list, model_type, title="Traning and Validation F1 Score"):
  """
  This function plots the testing F1 score curve.

  Args:
      f1_score_list: A list of testing F1 score values for each epoch.
      title: The title of the plot (default: "F1 Score").
  """
  plt.figure(figsize=(8, 5))
  plt.plot(f1_score_list, label='Training F1 score')
  plt.plot(val_f1_score_list, label='Validation F1 score')
  plt.xlabel('Epoch')
  plt.ylabel('F1 Score')
  plt.title(title)
  plt.grid(True)
  plt.show()
  filename = f"{model_type}_f1_score"
  plt.savefig(filename)    

def plot_confusion_matrix(confusion_matrix, model_type, labels=["Normal", "Pneumonia"], title="Confusion Matrix"):
  """
  This function plots the confusion matrix.

  Args:
      confusion_matrix: A 2D numpy array representing the confusion matrix.
      labels: A list of labels for the rows and columns (default: ["Negative", "Positive"]).
      title: The title of the plot (default: "Confusion Matrix").
  """
  plt.figure(figsize=(8, 8))
  plt.imshow(confusion_matrix, cmap='Blues')
  plt.xticks(range(len(labels)), labels, rotation=45)
  plt.yticks(range(len(labels)), labels)
  plt.colorbar()
  plt.title(title)
  plt.grid(False) 
#  Add text labels to each cell
  for i in range(len(confusion_matrix)):
    for j in range(len(confusion_matrix[0])):
      plt.text(j, i, int(confusion_matrix[i][j]), ha='center', va='center', fontsize=12)
  plt.show()
  filename = f"{model_type}_confusion_matrix"
  plt.savefig(filename)

def train(device, train_loader, model, criterion, optimizer, model_type):
    best_acc = 0.0
    best_model_wts = None
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    f1_score_list = []
    val_f1_score_list = []
    best_matrix = []

    for epoch in range(1, args.num_epochs+1):

        with torch.set_grad_enabled(True):
            avg_loss = 0.0
            train_acc = 0.0
            tp, tn, fp, fn = 0, 0, 0, 0     
            for _, data in enumerate(tqdm(train_loader)):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                outputs = torch.max(outputs, 1).indices
                sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
                tp += sub_tp
                tn += sub_tn
                fp += sub_fp
                fn += sub_fn          
                
            c_matrix = [[int(tp), int(fn)],
                        [int(fp), int(tn)]]

            avg_loss /= len(train_loader.dataset)
            train_acc = (tp+tn) / (tp+tn+fp+fn) * 100
            f1_score = (2*tp) / (2*tp+fp+fn)
            print(f'Epoch: {epoch}')
            print(f'↳ Loss: {avg_loss}')
            print(f'↳ Training Acc.(%): {train_acc:.2f}%')

        # write validation if you needed
        # val_acc, f1_score, c_matrix = test(val_loader, model)
        
        train_acc_list.append(train_acc)
        f1_score_list.append(f1_score)
       
        # Calculate test accuracy # Calculate val accuracy
        test_acc, _, test_matrix = test(test_loader, model)
        val_acc, val_f1_score, _ = val(val_loader, model)
        
        # Update the best test acc and matrix
        if test_acc > best_acc:
            best_acc = test_acc
            best_matrix = test_matrix
    
        val_acc_list.append(val_acc)  # Append test_acc to the list
        test_acc_list.append(test_acc)  # Append test_acc to the list
        val_f1_score_list.append(val_f1_score) # Append test_f1_score to the list

    # print('-----------------------')
    # print('In training phase:')
    # print("train_acc_list: ", train_acc_list)
    # print("test_acc_list: ", test_acc_list)
    # print("val_acc_list: ", val_acc_list)
    # print('-----------------------')

    # print("In testing phase: ")
    # print("test_f1_score: ", f1_score_list)
    # print('-----------------------')
    # print("best_test_matrix: ", test_matrix)
    # print('-----------------------')

    # plot
    plot_accuracy(train_acc_list, test_acc_list, val_acc_list, model_type)  # Plot the accuracy curves after training
    plot_f1_score(f1_score_list, val_f1_score_list, model_type)
    plot_confusion_matrix(test_matrix, model_type)

    torch.save(model.state_dict(), f"{model_type}_model_weights.pt")

    return train_acc_list, f1_score_list, best_matrix

def test(test_loader, model):
    val_acc = 0.0
    f1_score_list = []
    test_acc_list = []

    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        model.eval()
        for images, labels in test_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.max(outputs, 1).indices

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

        c_matrix = [[int(tp), int(fn)],
                    [int(fp), int(tn)]]
        
        test_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        val_acc = test_acc
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1_score = (2*tp) / (2*tp+fp+fn)

        # # Append F1 score to the list after calculation
        # f1_score_list.append(f1_score)
        # print("-------------------")
        # print("In testing phase: ")
        # print("test_f1_score: ", f1_score_list)
        # print("-------------------")

        # Calculate test accuracy
        # test_acc, _, _ = test(test_loader, model)
        test_acc_list.append(test_acc)  # Append test_acc to the list

        print (f'↳ Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}')
        print (f'↳ Test Acc.(%): {test_acc:.2f}%')
        
    return test_acc, f1_score, c_matrix

def val(test_loader, model):
    val_acc = 0.0
    f1_score_list = []
    test_acc_list = []

    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        model.eval()
        for images, labels in test_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.max(outputs, 1).indices

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

        c_matrix = [[int(tp), int(fn)],
                    [int(fp), int(tn)]]
        
        test_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1_score = (2*tp) / (2*tp+fp+fn)

        # # Append F1 score to the list after calculation
        # f1_score_list.append(f1_score)
        # print("-------------------")
        # print("In testing phase: ")
        # print("test_f1_score: ", f1_score_list)
        # print("-------------------")

        # Calculate test accuracy
        # test_acc, _, _ = test(test_loader, model)
        test_acc_list.append(test_acc)  # Append test_acc to the list

        print (f'↳ Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}')
        print (f'↳ Val Acc.(%): {test_acc:.2f}%')
        
    return test_acc, f1_score, c_matrix

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    # for model
    parser.add_argument('--num_classes', type=int, required=False, default=2)
    parser.add_argument('--model_type', type=str, choices=['resnet18', 'resnet50'], required=True,
                        help='Specify the model type (resnet18 or resnet50)')

    # for training
    parser.add_argument('--num_epochs', type=int, required=False, default=30)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--lr', type=float, default=0.000005)
    parser.add_argument('--wd', type=float, default=0.9)

    # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='chest_xray')

    # for data augmentation
    parser.add_argument('--degree', type=int, default=90)
    parser.add_argument('--resize', type=int, default=224)

    args = parser.parse_args()

    # set gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')

    # set dataloader (Train and Test dataset, write your own validation dataloader if needed.)
    train_dataset = ImageFolder(root=os.path.join(args.dataset, 'train'),
                                transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                                transforms.RandomRotation(args.degree),
                                                                transforms.ToTensor()]))
    # remove {, resample=False} in transforms.Compose
    test_dataset = ImageFolder(root=os.path.join(args.dataset, 'test'),
                               transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                               transforms.ToTensor()]))
    val_dataset = ImageFolder(root=os.path.join(args.dataset, 'val'),
                               transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                               transforms.ToTensor()]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # define model based on argument
    if args.model_type == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif args.model_type == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    
    # Freeze pre-trained layers (except the last layer)
    # for param in model.parameters():
    #     param.requires_grad = False  # Freeze all parameters

    # Modify the last layer for classification with specified num_classes
    # num_features_in = model.fc.in_features
    # model.fc = nn.Linear(num_features_in, args.num_classes)  # Replace the last layer

    # Unfreeze specific layers if needed (for fine-tuning)
    # For example, to unfreeze the last two convolutional layers:
    # for name, param in model.named_parameters():
    #     if 'layer4' in name:  # Unfreeze layers starting from layer4
    #         param.requires_grad = True

    num_neurons = model.fc.in_features
    model.fc = nn.Linear(num_neurons, args.num_classes)
    model = model.to(device)

    # define loss function, optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346]))
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # training
    train_acc_list, f1_score_list, best_train_matrix = train(device, train_loader, model, criterion, optimizer, args.model_type)

    print("finish")
