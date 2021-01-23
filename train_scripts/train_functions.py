import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader


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


def one_batch_train(batch, model, optimizer, criterion, device, scaler):
    current_loss = 0
    inputs, labels = batch
    inputs = inputs.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    if scaler is not None:
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    current_loss += loss.item()

    return current_loss
