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

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch

            if scaler is not None:
                with autocast():
                    outputs = model(inputs.to(device))
                    batch_loss = criterion(outputs, labels.to(device))
            else:
                outputs = model(inputs.to(device))
                batch_loss = criterion(outputs, labels.to(device))

            predictions.extend(sigmoid(outputs).cpu().detach().numpy())
            ground_truth.extend(labels.numpy())

            total_loss += batch_loss.item()
            iter_counter += 1

    total_loss /= iter_counter
    avg_auc, aucs = get_metric(np.array(predictions), np.array(ground_truth))

    return total_loss, avg_auc, aucs, time.time() - start_time


def get_metric(predictions, ground_truth):
    predictions[predictions == -np.inf] = 0.
    predictions[predictions == np.inf] = 1.
    predictions[predictions == np.nan] = 0
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
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

        predictions.extend(sigmoid(outputs).cpu().detach().numpy())
        ground_truth.extend(labels.numpy())
        iter_counter += 1
        total_loss += loss.item()

    total_loss /= iter_counter
    avg_auc, aucs = get_metric(np.array(predictions), np.array(ground_truth))

    return total_loss, avg_auc, aucs, time.time() - start_time
