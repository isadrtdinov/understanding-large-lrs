import math
import torch
import torch.nn.functional as F
from torch import nn
from model import (
    get_weights, set_weights,
    get_grads, num_params
)


def eval_model(model, train_loader, test_loader):
    X_train, y_train = train_loader.dataset.tensors
    X_test, y_test = test_loader.dataset.tensors

    with torch.no_grad():
        train_preds = model(X_train)[:, 0]
        test_preds = model(X_test)[:, 0]
        train_loss = F.binary_cross_entropy_with_logits(train_preds, y_train.to(torch.float)).item()
        test_loss = F.binary_cross_entropy_with_logits(test_preds, y_test.to(torch.float)).item()
        train_acc = ((train_preds > 0).to(torch.long) == y_train).to(torch.float).mean().item()
        test_acc = ((test_preds > 0).to(torch.long) == y_test).to(torch.float).mean().item()

    return train_loss, test_loss, train_acc, test_acc


def train_model(model, train_loader, test_loader, lr=3e-4, wd=0,
                num_iters=1000, ckpt_iters=200, log_iters=1, riemann_opt=False):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCEWithLogitsLoss()

    batch_losses = []
    trace = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': [],
        'weight': [], 'stoch_grad': [],
        'stoch_grad_norm': [],
    }
    it = 0

    train_loss, test_loss, train_acc, test_acc = \
        eval_model(model, train_loader, test_loader)
    trace['train_loss'].append(train_loss)
    trace['test_loss'].append(test_loss)
    trace['train_acc'].append(train_acc)
    trace['test_acc'].append(test_acc)
    trace['weight'].append(get_weights(model))
    trace['stoch_grad'].append(torch.zeros(num_params(model)))
    trace['stoch_grad_norm'].append(0.0)

    X_train_copy = torch.clone(train_loader.dataset.tensors[0])
    X_test_copy = torch.clone(test_loader.dataset.tensors[0])
    X_train_copy.requires_grad = True
    X_test_copy.requires_grad = True

    while True:
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            predictions = model(X)[:, 0]
            loss = criterion(predictions, y.to(torch.float))
            loss.backward()
            grad = get_grads(model)

            if riemann_opt:
                weights = get_weights(model)
                weight_norm = torch.norm(weights).item()
                grad_norm = torch.norm(grad).item()
                # sometimes grad norm is 0, put threshold 1e-7 for numerical stability
                normalized_grad = grad / max(grad_norm, 1e-7) * weight_norm 
                t = -lr * grad_norm / weight_norm
                weights = math.cos(t) * weights + math.sin(t) * normalized_grad
            else:
                optimizer.step()
                weights = get_weights(model)
                weights = weights / torch.norm(weights)

            set_weights(model, weights)
            batch_losses.append(loss.item())
            it += 1

            if it % log_iters == 0:
                train_loss, test_loss, train_acc, test_acc = \
                    eval_model(model, train_loader, test_loader)
                trace['train_loss'].append(train_loss)
                trace['test_loss'].append(test_loss)
                trace['train_acc'].append(train_acc)
                trace['test_acc'].append(test_acc)
                trace['stoch_grad_norm'].append(torch.norm(grad).item())

            if it % ckpt_iters == 0:
                trace['weight'].append(get_weights(model).cpu())
                trace['stoch_grad'].append(grad.cpu())

            if it >= num_iters:
                break

        if it >= num_iters:
            break

    return batch_losses, trace
