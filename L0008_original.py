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
# Data augmentation and normalization transformations for input images
transform = transforms.Compose([
    # Ensure all images are in RGB format
    ConvertToRGB(),
    # Resize images to a fixed size of (128, 128)
    transforms.Resize((128, 128)),
    # Apply random horizontal flipping with a probability of 0.5
    transforms.RandomHorizontalFlip(),
    # Apply random rotation to images with a maximum angle of 10 degrees
    transforms.RandomRotation(10),
    # Adjust brightness, contrast, saturation, and hue of images randomly within specified ranges
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    # Convert images to PyTorch tensors
    transforms.ToTensor(),
    # Normalize pixel values of images to have a mean of 0.5 and standard deviation of 0.5 for each channel
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Transformation for target segmentation masks
target_transform = transforms.Compose([
    # Resize segmentation masks to the same size as input images (128x128)
    transforms.Resize((128, 128)),
    # Apply custom target transformation to handle label mapping and conversion to tensors
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
    """
    Calculate the Intersection over Union (IoU) metric for semantic segmentation evaluation.

    This function computes the Intersection over Union (IoU) metric, a measure of the accuracy of semantic
    segmentation models. It compares the predicted segmentation masks with the ground truth masks to determine
    the degree of overlap between them.

    Parameters:
        outputs (torch.Tensor): Predicted segmentation masks from the model. Shape: (batch_size, num_classes, height, width)
        labels (torch.Tensor): Ground truth segmentation masks. Shape: (batch_size, height, width)
        num_classes (int): Number of classes (excluding background) in the dataset.

    Returns:
        float: Mean IoU value across the batch.
    """
    smooth = 1e-6

    # Convert predicted masks to binary masks by selecting the class index with highest probability
    outputs = torch.argmax(outputs, dim=1)

    # Calculate intersection and union of predicted and ground truth masks
    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))

    # Calculate IoU for each sample in the batch and add a small smooth term to prevent division by zero
    iou = (intersection + smooth) / (union + smooth)

    # Return the mean IoU value across the batch as a Python float
    return iou.mean().item()


def train_model(model, criterion, optimizer, scheduler, train_dataloader, test_dataloader, device, epochs=50, early_stop_patience=5, progress_interval=10):
    """
    Train the semantic segmentation model.

    This function trains the model using the specified criterion, optimizer, and learning rate scheduler.
    It iterates over the specified number of epochs, monitoring the training progress and evaluating the
    model on the validation set after each epoch. Early stopping is employed if the validation IoU does not
    improve for a certain number of epochs.

    Parameters:
        model (torch.nn.Module): The semantic segmentation model to be trained.
        criterion (torch.nn.Module): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device (CPU or GPU) on which to perform training.
        epochs (int): Number of training epochs. Default is 50.
        early_stop_patience (int): Number of epochs with no improvement on validation IoU before early stopping. Default is 5.
        progress_interval (int): Interval for displaying training progress in batches. Default is 10.

    Returns:
        None
    """
    model.train()  # Set the model to training mode
    best_iou = 0.0
    patience_counter = 0
    writer = SummaryWriter()  # TensorBoard writer for logging training metrics

    # Iterate over epochs
    for epoch in range(epochs):
        running_loss = 0.0
        running_iou = 0.0
        total = 0

        # Iterate over batches in the training DataLoader
        progress_bar = tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{epochs}")
        for i, data in progress_bar:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients

            outputs = model(inputs)['out']  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

            running_loss += loss.item()
            iou = calculate_iou(outputs, labels, num_classes=model.classifier[4].out_channels)
            running_iou += iou
            total += 1

            if i % progress_interval == 0:
                train_loss = running_loss / total
                train_iou = running_iou / total
                progress_bar.set_postfix(loss=f'{train_loss:.4f}', iou=f'{train_iou:.4f}')  # Update progress bar

        # Compute average training loss and IoU
        train_loss = running_loss / total
        train_iou = running_iou / total

        # Validate the model on the validation dataset
        val_iou, avg_val_loss = validate_model(model, test_dataloader, device, criterion)

        # Log training and validation metrics
        writer.add_scalar('IoU/Training', train_iou, epoch)
        writer.add_scalar('IoU/Validation', val_iou, epoch)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        scheduler.step()  # Adjust learning rate

        # Check if validation IoU improved, otherwise update patience counter
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
        else:
            patience_counter += 1
            # Early stopping if validation IoU does not improve for a certain number of epochs
            if patience_counter >= early_stop_patience:
                print(f'Early stopping at epoch {epoch}')
                break

    writer.close()  # Close TensorBoard writer


def validate_model(model, testloader, device, criterion):
    """
    Validate the semantic segmentation model on the validation dataset.

    This function evaluates the model's performance on the validation dataset by computing the average IoU and loss.

    Parameters:
        model (torch.nn.Module): The semantic segmentation model to be evaluated.
        testloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device (CPU or GPU) on which to perform evaluation.
        criterion (torch.nn.Module): The loss function used for evaluation.

    Returns:
        float: Mean IoU value across the validation dataset.
        float: Average loss value across the validation dataset.
    """
    model.eval()  # Set the model to evaluation mode
    total_iou = 0.0
    total_loss = 0.0

    # Iterate over batches in the validation DataLoader
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)['out']  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            total_loss += loss.item()

            # Calculate IoU for the batch and add to the total IoU
            iou = calculate_iou(outputs, labels, num_classes=model.classifier[4].out_channels)
            total_iou += iou

    # Compute average IoU and loss across the validation dataset
    avg_iou = total_iou / len(testloader)
    avg_loss = total_loss / len(testloader)

    return avg_iou, avg_loss


def train_session():
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

    PATH = 'fcn_resnet50_model_v00.pth'
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH)

if __name__ == "__main__":
    train_session()


