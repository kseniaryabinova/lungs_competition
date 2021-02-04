import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import time


def eval_model(model, val_loader: DataLoader,
               device: torch.device, criterion, scaler):
    predictions = []
    ground_truth = []
    total_loss = 0
    sigmoid = torch.nn.Sigmoid()
    iter_counter = 0
    start_time = time.time()

    model.eval()
    model = model.float()

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch

            if scaler is not None:
                with autocast():
                    outputs = model(inputs.to(device, non_blocking=True))
                    batch_loss = criterion(outputs, labels.to(device, non_blocking=True))
            else:
                outputs = model(inputs.to(device, non_blocking=True))
                batch_loss = criterion(outputs, labels.to(device, non_blocking=True))

            predictions.extend(sigmoid(outputs).cpu().detach().numpy())
            ground_truth.extend(labels.numpy())

            total_loss += batch_loss.item()
            iter_counter += 1

    total_loss /= iter_counter
    avg_auc, aucs = get_metric(np.array(predictions, dtype=np.float), np.array(ground_truth, dtype=np.float))

    return total_loss, avg_auc, aucs, time.time() - start_time


def get_metric(predictions, ground_truth):
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=0.0)
    ground_truth = np.nan_to_num(ground_truth, nan=0.0, posinf=1.0, neginf=0.0)
    aucs = roc_auc_score(ground_truth, predictions, average=None)
    return np.mean(aucs), aucs


def one_epoch_train(model, train_loader, optimizer, criterion, device, scaler):
    total_loss = 0
    iter_counter = 0
    predictions = []
    ground_truth = []
    sigmoid = torch.nn.Sigmoid()
    start_time = time.time()

    model.train()

    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = model(inputs.to(device, non_blocking=True))
                loss = criterion(outputs, labels.to(device, non_blocking=True))
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs.to(device, non_blocking=True))
            loss = criterion(outputs, labels.to(device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
            optimizer.step()

        predictions.extend(sigmoid(outputs).cpu().detach().numpy())
        ground_truth.extend(labels.numpy())
        iter_counter += 1
        total_loss += loss.item()

    total_loss /= iter_counter
    avg_auc, aucs = get_metric(np.array(predictions, dtype=np.float), np.array(ground_truth, dtype=np.float))

    return total_loss, avg_auc, aucs, time.time() - start_time


