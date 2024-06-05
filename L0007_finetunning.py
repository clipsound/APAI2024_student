import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import Caltech101
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm
import numpy as np


# Custom transform to ensure all images are in RGB format
class ConvertToRGB:
    def __call__(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image


# Data augmentation and normalization
transform = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    #transforms.RandomCrop(128, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def load_data(root='./data', batch_size=16, worker=4):
    dataset = Caltech101(root=root, download=True, transform=transform)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=worker)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=worker)

    return trainloader, testloader, dataset.categories


def create_pretrained_model(num_classes):
    model = models.resnet101(weights=True)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def create_pretrained_model_mobilenet(num_classes):
    model = models.mobilenet_v3_large(weights=True)  # Puoi usare anche mobilenet_v3_small se preferisci
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    return model

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


def train_model(model,
                criterion,
                optimizer,
                scheduler,
                train_dataloader,
                test_dataloader,
                device,
                epochs=50,
                early_stop_patience=5,
                progress_interval=10):
    model.train()
    best_accuracy = 0.0
    patience_counter = 0
    writer = SummaryWriter()
    writer.add_text('info', f'Epochs {epochs}')
    writer.add_text('info', 'model: mobilenet_v3_large')

    for epoch in range(epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        total = 0

        progress_bar = tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader),
                            desc=f"Epoch {epoch + 1}/{epochs}")

        for i, data in progress_bar:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            accuracy = calculate_accuracy(outputs, labels)
            running_accuracy += accuracy
            total += 1

            if i % progress_interval == 0:
                train_loss = running_loss / total
                train_accuracy = running_accuracy / total
                progress_bar.set_postfix(loss=f'{train_loss:.4f}', accuracy=f'{train_accuracy:.4f}')

        train_loss = running_loss / total
        train_accuracy = running_accuracy / total

        # Calcolo dell'accuracy sulla validation
        val_accuracy, avg_val_loss = validate_model(model, test_dataloader, device, criterion)

        # Registrazione delle metriche su TensorBoard
        writer.add_scalar('Accuracy/Training', train_accuracy, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)


        # Step del scheduler
        scheduler.step()

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f'Early stopping at epoch {epoch}')
                break

    writer.close()


def validate_model(model, testloader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(testloader)
    return accuracy, avg_loss


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    trainloader, testloader, categories = load_data(batch_size=16, worker=8)

    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Using device: {device}")

    use_mobilenet = True
    if use_mobilenet:
        net = create_pretrained_model_mobilenet(num_classes=102).to(device)
    else:
        net = create_pretrained_model(num_classes=102).to(device)

    print("dimensione del modello preaddestrato ", sum(p.numel() for p in net.parameters() if p.requires_grad))

    criterion = nn.CrossEntropyLoss()
    if use_mobilenet:
        optimizer = optim.SGD(net.classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(net.fc.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_model(net, criterion, optimizer, scheduler, trainloader, testloader, device, epochs=100, early_stop_patience=10)

    PATH = './caltech101_pretrained_net.pth'
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH)


if __name__ == "__main__":
    main()
