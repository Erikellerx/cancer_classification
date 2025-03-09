import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from torchsummary import summary

from utils import get_model
from dataset import CancerDataset

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set tf32 to true 
torch.backends.cuda.matmul.allow_tf32 = True


def train(model, train_loader, criterion, optimizer, epoch, accumulation_steps=16):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Zero the gradients at the start of the epoch
    optimizer.zero_grad()
    
    for idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch", leave=False)):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss and scale it by the accumulation steps
        loss = criterion(outputs, labels) / accumulation_steps
        
        # Backward pass: accumulate gradients
        loss.backward()
        
        # Update running loss (multiply back by accumulation_steps to get the original loss scale)
        running_loss += loss.item() * accumulation_steps * images.size(0)
        
        # Calculate predictions for accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Perform optimizer step every accumulation_steps batches
        if (idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    # In case the number of batches isn't exactly divisible by accumulation_steps,
    # perform an optimizer step on the remaining gradients
    if (idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def evaluation(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Lists to store true labels and predictions for metrics computation
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            # Get predictions and update accuracy metrics
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels (move to CPU and convert to numpy arrays)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')  # Change average if needed
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, (epoch_acc, f1, precision, recall)



if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a model on the Cancer dataset")
    parser.add_argument('--model', type=str, default='resnext', help='Root directory of the dataset')
    args = parser.parse_args()
    
    
    root_dir = 'data'
    model_name = args.model
    accumulation_steps = 8
    training_batch_size = 32
    testing_batch_size = 64
    
    load_checkpoint = None
    
    #compute the effective batch size
    effective_batch_size = training_batch_size * accumulation_steps
    print(f"Effective Batch Size: {effective_batch_size}")
    
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.SVHN),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.7230, 0.5556, 0.5390], std=[0.1209, 0.1340, 0.1448])
    ])
    
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.7230, 0.5556, 0.5390], std=[0.1209, 0.1340, 0.1448])
    ])
    
    
    # Create a dataset instance for the training split
    train_dataset = CancerDataset(root_dir=root_dir, split='train', transform=train_transforms)
    test_dataset = CancerDataset(root_dir=root_dir, split='test', transform=test_transforms)
    
    # Create a DataLoader for the training split
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=testing_batch_size, shuffle=False)
    
    # Define the model, loss function, and optimizer
    model = get_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1, verbose=True)

    # Print model summary
    summary(model, input_size=(3, 224, 224))
    
    if load_checkpoint:
        model.load_state_dict(torch.load(load_checkpoint))
        print(f"Loaded checkpoint from {load_checkpoint}")
        test_loss, (test_acc, f1, precision, recall) = evaluation(model, test_loader, criterion)
    
    num_epochs = 160
    best_f1 = f1 if load_checkpoint else 0
    
    
    for epoch in tqdm(range(num_epochs)):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch, accumulation_steps)
        test_loss, (test_acc, f1, precision, recall) = evaluation(model, test_loader, criterion)
        scheduler.step()
        
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Save the model if it has the best F1 score so far
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f'checkpoints/{model_name}.pth')
            tqdm.write(f"**Model saved with F1 Score: {best_f1:.4f}**")
        
        tqdm.write("-" * 64)
        
        
    
    
    
    
    
    
    