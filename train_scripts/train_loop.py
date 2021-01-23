import pandas as pd
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import ImageDataset
from resnet18 import ResNet18
from train_functions import one_batch_train

torch.manual_seed(25)

train_df = pd.read_csv('../dataset/train.csv')
image_transforms = transforms.Compose([
    transforms.ToTensor(),
])
train_set = ImageDataset(train_df, image_transforms, '../dataset/train')
train_loader = DataLoader(
        train_set,
        batch_size=16,
        num_workers=12,
        pin_memory=True
    )

scaler = GradScaler()
# scaler = None
if scaler is None:
    model = ResNet18(11, 1, pretrained_backbone=False, mixed_precision=False)
else:
    model = ResNet18(11, 1, pretrained_backbone=False, mixed_precision=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.to(device)


for epoch in range(1):

    running_loss = 0.0
    model.train()

    for i, batch in enumerate(train_loader, 0):
        current_loss = one_batch_train(batch, model, optimizer, criterion, device, scaler)
        running_loss += current_loss

        if i % 1 == 0:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

        if i == 100:
            break

    torch.save(model.state_dict(), 'model.pth')


# print(eval_model(model, train_loader, device, criterion))
