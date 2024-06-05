import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import Caltech101
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
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
    transforms.RandomCrop(128, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_data(root='./data', batch_size=32, worker=8):
    dataset = Caltech101(root=root, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=worker)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=worker)

    return trainloader, testloader, dataset.categories


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 102)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MediumNet256(nn.Module): #256x256
    def __init__(self):
        super(MediumNet256, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 102)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MediumNet(nn.Module):
    def __init__(self):
        super(MediumNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Aggiornato per 128x128 input
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 102)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 128x128 -> 64x64
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 64x64 -> 32x32
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 16x16 -> 8x8
        x = x.view(-1, 128 * 8 * 8)  # Aggiornato per 128x128 input
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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

    for epoch in range(epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        total = 0

        progress_bar = tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{epochs}")

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

    trainloader, testloader, categories = load_data(batch_size=32, worker=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    net = SimpleNet().to(device)

    print("dimensione SimpleNet ", sum(p.numel() for p in net.parameters() if p.requires_grad))

    net = MediumNet().to(device)

    print("dimensione MediumNet ", sum(p.numel() for p in net.parameters() if p.requires_grad))

    criterion = nn.CrossEntropyLoss()

    if False:
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_model(net, criterion, optimizer, scheduler, trainloader, testloader, device, epochs=100, early_stop_patience=10)

    PATH = './caltech101_net.pth'
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH)

if __name__ == "__main__":
    main()
