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
    predictions = np.array(predictions, dtype=np.float)
    ground_truth = np.array(ground_truth, dtype=np.float)
    avg_auc, aucs, rocs_parameters = get_metric(predictions, ground_truth)

    return total_loss, avg_auc, aucs, rocs_parameters, (predictions, ground_truth), \
           time.time() - start_time


def get_metric(predictions, ground_truth):
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=0.0)
    ground_truth = np.nan_to_num(ground_truth, nan=0.0, posinf=1.0, neginf=0.0)
    aucs = roc_auc_score(ground_truth, predictions, average=None)

    rocs_parameters = []
    for i in range(predictions.shape[1]):
        fpr, tpr, _ = roc_curve(ground_truth[:, i], predictions[:, i])
        rocs_parameters.append((fpr, tpr))

    return np.mean(aucs), aucs, rocs_parameters


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
                loss = criterion(outputs, smooth_labels(labels.to(device), smoothing=0.0))
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
            loss = criterion(outputs, smooth_labels(labels.to(device), smoothing=0.0))
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
    avg_auc, aucs, rocs_parameters = get_metric(predictions, ground_truth)

    return total_loss, avg_auc, aucs, rocs_parameters, (predictions, ground_truth), \
           time.time() - start_time


class CustomLoss(nn.Module):
    def __init__(self, weights=(1, 1), class_weights=None):
        super(CustomLoss, self).__init__()
        self.weights = weights
        self.class_weights = class_weights

    def forward(self, teacher_features, features, y_pred, labels):
        consistency_loss = nn.MSELoss()(teacher_features.view(-1), features.view(-1))
        if self.class_weights is not None:
            cls_loss = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)(y_pred, labels)
        else:
            cls_loss = nn.BCEWithLogitsLoss()(y_pred, labels)
        loss = self.weights[0] * consistency_loss + self.weights[1] * cls_loss
        return loss


def one_epoch_train_2_stage(teacher, student, train_loader, optimizer, criterion, device, scaler,
                            iters_to_accumulate=2, clip_grads=False):
    total_loss = 0
    iter_counter = 0
    predictions = []
    ground_truth = []
    sigmoid = torch.nn.Sigmoid()
    start_time = time.time()

    student.train()
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        inputs_annot, inputs, labels = batch

        with torch.no_grad():
            teacher_features, _, _ = teacher(inputs_annot.to(device))

        if scaler is not None:
            with autocast():
                student_features, _, outputs = student(inputs.to(device))
                loss = criterion(teacher_features, student_features, outputs, labels.to(device))
                loss = loss / iters_to_accumulate
            scaler.scale(loss).backward()

            if (i + 1) % iters_to_accumulate == 0:
                if clip_grads:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1000)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        predictions.extend(sigmoid(outputs).cpu().detach().numpy())
        ground_truth.extend(labels.numpy())
        iter_counter += 1
        total_loss += loss.item()

    total_loss /= iter_counter / iters_to_accumulate
    predictions = np.array(predictions, dtype=np.float)
    ground_truth = np.array(ground_truth, dtype=np.float)
    avg_auc, aucs, _ = get_metric(predictions, ground_truth)

    return total_loss, avg_auc, aucs, (predictions, ground_truth), time.time() - start_time


def eval_model_2_stage(model, val_loader: DataLoader,
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
                    _, _, outputs = model(inputs.to(device))
                    batch_loss = criterion(outputs, labels.to(device))
            else:
                _, _, outputs = model(inputs.to(device))
                batch_loss = criterion(outputs, labels.to(device))

            predictions.extend(sigmoid(outputs).cpu().detach().numpy())
            ground_truth.extend(labels.numpy())

            total_loss += batch_loss.item()
            iter_counter += 1

    total_loss /= iter_counter
    predictions = np.array(predictions, dtype=np.float)
    ground_truth = np.array(ground_truth, dtype=np.float)
    avg_auc, aucs, _ = get_metric(predictions, ground_truth)

    return total_loss, avg_auc, aucs, (predictions, ground_truth), time.time() - start_time


class UnlabeledLoss(nn.Module):
    def __init__(self, t1, t2, alpha, class_weights=None):
        super(UnlabeledLoss, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.alpha = alpha
        self.class_weights = class_weights
        self.unlabeled_loss = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)

    def get_weight(self, epoch):
        if epoch < self.t1:
            return 0.
        elif self.t1 <= epoch < self.t2:
            return (epoch - self.t1) / (self.t2 - self.t1) * self.alpha
        elif self.t2 <= epoch:
            return self.alpha

    def get_labels_out_of_predictions(self, predictions):
        return (predictions > 0.5).float()

    def forward(self, epoch, predictions):
        weight = self.get_weight(epoch)
        labels = self.get_labels_out_of_predictions(predictions)
        return weight * self.unlabeled_loss(predictions, labels)


def train_one_epoch_pseudolabel(epoch: int, model: nn.Module,
                                train_loader_with_labels: DataLoader,
                                train_loader_without_labels: DataLoader,
                                optimizer: torch.optim.Optimizer,
                                labeled_loss: nn.Module, unlabeled_loss: nn.Module,
                                device, scaler, iters_to_accumulate, clip_grads):
    start_time = time.time()
    model.train()
    optimizer.zero_grad()

    if unlabeled_loss.get_weight(epoch) == 0:
        train_loss, train_avg_auc, train_auc, train_rocs, train_data_pr, _ = one_epoch_train(
            model, train_loader_with_labels, optimizer, labeled_loss,
            device, scaler, iters_to_accumulate, clip_grads)
    else:
        for i, unlabeled_inputs in enumerate(train_loader_without_labels):
            if scaler is not None:
                with autocast():
                    outputs = model(unlabeled_inputs.to(device))
                    loss = unlabeled_loss(epoch, outputs)
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
                outputs = model(unlabeled_inputs.to(device))
                loss = unlabeled_loss(epoch, outputs)
                loss = loss / iters_to_accumulate
                loss.backward()

                if (i + 1) % iters_to_accumulate == 0:
                    if clip_grads:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
                    optimizer.step()
                    optimizer.zero_grad()

        train_loss, train_avg_auc, train_auc, train_rocs, train_data_pr, _ = one_epoch_train(
            model, train_loader_with_labels, optimizer, labeled_loss,
            device, scaler, iters_to_accumulate, clip_grads)

    return train_loss, train_avg_auc, train_auc, train_rocs, train_data_pr, time.time() - start_time
