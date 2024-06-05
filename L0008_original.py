import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Custom transform to ensure all images are in RGB format
class ConvertToRGB:
    def __call__(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

# Custom transform to handle target transformation
class TargetTransform:
    def __call__(self, target):
        target = np.array(target)
        target[target == 255] = 0  # Map the 255 label to 0 (or any other valid label)
        return torch.squeeze(torch.tensor(target, dtype=torch.long))

# Data augmentation and normalization
transform = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

target_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    TargetTransform()
])

def load_data(root='./data', batch_size=16, worker=4):
    dataset = VOCSegmentation(root=root, year='2012', image_set='train', download=True, transform=transform, target_transform=target_transform)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=worker)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=worker)

    return trainloader, testloader

def create_fcn_resnet50_model(num_classes):
    model = fcn_resnet50(weights="FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1")
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    return model

def calculate_iou(outputs, labels, num_classes):
    smooth = 1e-6
    outputs = torch.argmax(outputs, dim=1)
    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def train_model(model, criterion, optimizer, scheduler, train_dataloader, test_dataloader, device, epochs=50, early_stop_patience=5, progress_interval=10):
    model.train()
    best_iou = 0.0
    patience_counter = 0
    writer = SummaryWriter()

    for epoch in range(epochs):
        running_loss = 0.0
        running_iou = 0.0
        total = 0

        progress_bar = tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{epochs}")

        for i, data in progress_bar:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)['out']
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            iou = calculate_iou(outputs, labels, num_classes=model.classifier[4].out_channels)
            running_iou += iou
            total += 1

            if i % progress_interval == 0:
                train_loss = running_loss / total
                train_iou = running_iou / total
                progress_bar.set_postfix(loss=f'{train_loss:.4f}', iou=f'{train_iou:.4f}')

        train_loss = running_loss / total
        train_iou = running_iou / total

        val_iou, avg_val_loss = validate_model(model, test_dataloader, device, criterion)

        writer.add_scalar('IoU/Training', train_iou, epoch)
        writer.add_scalar('IoU/Validation', val_iou, epoch)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        scheduler.step()

        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f'Early stopping at epoch {epoch}')
                break

    writer.close()

def validate_model(model, testloader, device, criterion):
    model.eval()
    total_iou = 0.0
    total_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            iou = calculate_iou(outputs, labels, num_classes=model.classifier[4].out_channels)
            total_iou += iou

    avg_iou = total_iou / len(testloader)
    avg_loss = total_loss / len(testloader)
    return avg_iou, avg_loss

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    trainloader, testloader = load_data(batch_size=16, worker=8)

    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Using device: {device}")

    net = create_fcn_resnet50_model(num_classes=21).to(device)

    print("dimensione del modello preaddestrato ", sum(p.numel() for p in net.parameters() if p.requires_grad))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_model(net, criterion, optimizer, scheduler, trainloader, testloader, device, epochs=100, early_stop_patience=10)

    PATH = './voc_fcn_resnet50_model.pth'
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH)

if __name__ == "__main__":
    main()
