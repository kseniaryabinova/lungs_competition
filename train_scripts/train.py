import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import roc_auc_score

from dataloader import ImageDataset
from resnet18 import ResNet18

torch.manual_seed(25)

train_df = pd.read_csv('../dataset/train.csv')

image_transforms = transforms.Compose([
    transforms.ToTensor(),
])

train_set = ImageDataset(train_df, image_transforms, '../dataset/train')

train_loader = DataLoader(
        train_set,
        batch_size=128,
        num_workers=12,
        pin_memory=True
    )

model = ResNet18(11, 1, pretrained_backbone=False, mixed_precision=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.to(device)


def eval_model(model: torch.nn.Module, val_loader: DataLoader,
               device: torch.device, criterion):
    model.eval()

    predictions = []
    ground_truth = []
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            prediction_batch = model(images.to(device))

            predictions.extend(prediction_batch.cpu().numpy())
            ground_truth.extend(labels.numpy())

            batch_loss = criterion(prediction_batch, labels.to(device))
            total_loss += batch_loss.item()
            break

    total_loss /= len(val_loader)
    avg_auc = get_metric(np.array(predictions), np.array(ground_truth))

    return total_loss, avg_auc


def get_metric(predictions, ground_truth):
    aucs = roc_auc_score(ground_truth, predictions, average=None)
    return np.mean(aucs)


data = next(iter(train_loader))

for epoch in range(1):

    running_loss = 0.0
    model.train()

    for i in range(1):
    # for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1 == 0:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

    torch.save(model.state_dict(), 'model.pth')


print(labels.cpu())
print(outputs.cpu())
print(eval_model(model, train_loader, device, criterion))
