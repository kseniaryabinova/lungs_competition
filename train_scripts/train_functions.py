import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn, Tensor
from torch.autograd.grad_mode import F
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
    predictions = np.array(predictions, dtype=np.float)
    ground_truth = np.array(ground_truth, dtype=np.float)
    avg_auc, aucs = get_metric(predictions, ground_truth)

    return total_loss, avg_auc, aucs, (predictions, ground_truth), time.time() - start_time


def get_metric(predictions, ground_truth):
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=0.0)
    ground_truth = np.nan_to_num(ground_truth, nan=0.0, posinf=1.0, neginf=0.0)
    aucs = roc_auc_score(ground_truth, predictions, average=None)

    rocs_parameters = []
    for i in range(predictions.shape[1]):
        fpr, tpr, _ = roc_curve(ground_truth[:, i], predictions[:, i])
        rocs_parameters.append((fpr, tpr))

    return np.mean(aucs), aucs


def group_weight(module, weight_decay):
    decay = []
    no_decay = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in ['bn', 'bias']:
            no_decay.append(param)
        else:
            decay.append(param)

    print(len(no_decay), len(decay))
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def smooth_labels(targets: Tensor, smoothing=0.0):
    assert 0 <= smoothing < 1
    with torch.no_grad():
        result = targets.float() * (1 - smoothing) + 0.5 * smoothing
    return result


def one_epoch_train(model, train_loader, optimizer, criterion, device, scaler, iters_to_accumulate=2, clip_grads=False):
    total_loss = 0
    iter_counter = 0
    predictions = []
    ground_truth = []
    sigmoid = torch.nn.Sigmoid()
    start_time = time.time()

    model.train()
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        inputs, labels = batch

        if scaler is not None:
            with autocast():
                outputs = model(inputs.to(device))
                loss = criterion(outputs, smooth_labels(labels.to(device), smoothing=0.2))
                loss = loss / iters_to_accumulate
            scaler.scale(loss).backward()

            if (i + 1) % iters_to_accumulate == 0:
                if clip_grads:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, smooth_labels(labels.to(device), smoothing=0.2))
            loss = loss / iters_to_accumulate
            loss.backward()

            if (i + 1) % iters_to_accumulate == 0:
                if clip_grads:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
                optimizer.step()
                optimizer.zero_grad()

        predictions.extend(sigmoid(outputs).cpu().detach().numpy())
        ground_truth.extend(labels.numpy())
        iter_counter += 1
        total_loss += loss.item()

    total_loss /= iter_counter / iters_to_accumulate
    predictions = np.array(predictions, dtype=np.float)
    ground_truth = np.array(ground_truth, dtype=np.float)
    avg_auc, aucs = get_metric(predictions, ground_truth)

    return total_loss, avg_auc, aucs, (predictions, ground_truth), time.time() - start_time
