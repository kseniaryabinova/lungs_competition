import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import ImageDataset
from resnet18 import ResNet18

torch.manual_seed(25)

train_df = pd.read_csv('../dataset/train.csv')

image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
])

train_set = ImageDataset(train_df, image_transforms, '../dataset/train')

train_loader = DataLoader(
        train_set,
        batch_size=16,
        num_workers=12,
        pin_memory=True
    )

model = ResNet18(11, 1, pretrained_backbone=False, mixed_precision=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.to(device)

data = next(iter(train_loader))

for epoch in range(1):

    running_loss = 0.0
    for i in range(1000):
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

print(labels.cpu())
print(outputs.cpu())
