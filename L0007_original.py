import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import Caltech101
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

class ConvertToRGB:
    def __call__(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

transform = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_data(root='./data', batch_size=16, worker=4):
    dataset = Caltech101(root=root, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=worker)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=worker)

    return trainloader, testloader, dataset.categories

class SimpleNet(nn.Module):
    #bigger images
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 102)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




class MidNet(nn.Module):
    def __init__(self):
        super(MidNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 102)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


from torch.utils.tensorboard import SummaryWriter


def train_model(model,
                criterion,
                optimizer,
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
        val_accuracy, avg_val_loss = validate_model(model, test_dataloader, device, criterion=None)

        # Registrazione delle metriche su TensorBoard
        writer.add_scalar('Accuracy/Training', train_accuracy, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        #writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f'Early stopping at epoch {epoch}')
                break

    writer.close()


def validate_model(model, testloader, device, criterion=None):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(testloader)
    return accuracy, avg_loss

def main():
    trainloader, testloader, categories = load_data(batch_size=16, worker=8)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    net = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_model(net, criterion, optimizer, trainloader, testloader, device, epochs=50)

    PATH = './caltech101_net.pth'
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH)

if __name__ == "__main__":
    main()
