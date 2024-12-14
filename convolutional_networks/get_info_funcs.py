import torch
import numpy as np
from torch import nn
# from gpytorch.utils.lanczos import lanczos_tridiag
from collections import defaultdict


def Rop(ys, xs, vs):
    if isinstance(ys, tuple):
        ws = [torch.zeros_like(y, requires_grad=True) for y in ys]
    else:
        ws = torch.zeros_like(ys, requires_grad=True)

    gs = torch.autograd.grad(ys, xs, grad_outputs=ws, create_graph=True)
    re = torch.autograd.grad(gs, ws, grad_outputs=vs)
    return tuple([j.detach() for j in re])

def Lop(ys, xs, ws):
    vJ = torch.autograd.grad(ys, xs, grad_outputs=ws)
    return tuple([j.detach() for j in vJ])

def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        n = tensor.numel()
        outList.append(vector[i : i + n].view(tensor.shape))
        i += n
    return outList

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def Hvp(vec, params, outputs, data_size, targets, criterion=None, **kwargs):
    "Returns Hessian vec. prod."
    criterion = criterion or nn.CrossEntropyLoss(reduction='sum')
    loss = criterion(outputs, targets) / data_size
    grad = torch.autograd.grad(loss, params, create_graph=True)

    # Compute inner product of gradient with the direction vector
    prod = 0.
    for (g, v) in zip(grad, vec):
        prod += (g * v).sum()

    # Compute the Hessian-vector product, H*v
    Hv = torch.autograd.grad(prod, params)
    return Hv

def Fvp(vec, params, outputs, data_size, **kwargs):
    "Returns Fisher vec. prod."
    Jv, = Rop(outputs, params, vec)
    batch, dims = outputs.size(0), outputs.size(1)
    probs = torch.softmax(outputs, dim=1)
    
    M = torch.zeros(batch, dims, dims, device=probs.device)
    M.view(batch, -1)[:, ::dims + 1] = probs
    H = M - torch.einsum('bi,bj->bij', (probs, probs))
    
    HJv = torch.squeeze(H @ torch.unsqueeze(Jv, -1), -1) / data_size
    JHJv = Lop(outputs, params, HJv)
    return JHJv

def eval_mvp(Mvp, vec, params, net, dataloader, **kwargs):
    M_v = [torch.zeros_like(p) for p in params]
    data_size = len(dataloader.dataset)

    for inputs, targets in dataloader:
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = net(inputs)

        kwargs['targets'] = targets
        for i, v in enumerate(Mvp(vec, params, outputs, data_size, **kwargs)):
            M_v[i] += v
                
    return M_v


def lanczos_tridiag_to_diag(t_mat):
    orig_device = t_mat.device
    
    if t_mat.size(-1) < 32:
        retr = torch.symeig(t_mat.cpu(), eigenvectors=True)
    else:
        retr = torch.symeig(t_mat, eigenvectors=True)

    evals, evecs = retr
    return evals.to(orig_device), evecs.to(orig_device)

def eval_eigs(model, dataloader, fisher=True, train_mode=False, nsteps=10, return_evecs=False, pnames=None):
    """
        Get "nsteps" approximate eigenvalues of the Fisher or Hessian marix via Lanczos.
        Args:
            model: the trained model.
            dataloader: dataloader for the dataset, may use a subset of it.
            fisher: whether to use Fisher MVP or Hessian MVP
            train_mode: whether to run the model in train mode.
            nsteps: number of lanczos steps to perform
            return_evecs: whether to return evecs as well
            pnames: names of parameters to consider (if None all params are used)
        Returns:
            eigvals: calculated approximate eigenvalues
            eigvecs: calculated approximate eigenvectors, optional
    """
    if train_mode:
        model.train()
    else:
        model.eval()
        
    kwargs = {}
    if fisher:
        Mvp = Fvp
    else:
        Mvp = Hvp
        criterion = nn.CrossEntropyLoss(reduction='sum')
        kwargs['criterion'] = criterion

    params = list(model.parameters()) if pnames is None else [p for n, p in model.named_parameters() if n in pnames]
    N = sum(p.numel() for p in params)
    
    def lanczos_mvp(vec):
        vec = unflatten_like(vec.view(-1), params)
        fvp = eval_mvp(Mvp, vec, params, model, dataloader, **kwargs)
        return flatten(fvp).unsqueeze(1)
    
    # use lanczos to get the t and q matrices out
    q_mat, t_mat = lanczos_tridiag(
        lanczos_mvp,
        nsteps,
        device=params[0].device,
        dtype=params[0].dtype,
        matrix_shape=(N, N),
    )
    
    # convert the tridiagonal t matrix to the eigenvalues
    eigvals, eigvecs = lanczos_tridiag_to_diag(t_mat)
    eigvecs = q_mat @ eigvecs if return_evecs else None
    return eigvals, eigvecs

def eval_trace(model, dataloader, fisher=True, train_mode=False, n_vecs=5, pnames=None):
    "Returns Fisher or Hessian traces divided by number of parameters."
    if train_mode:
        model.train()
    else:
        model.eval()
        
    kwargs = {}
    if fisher:
        Mvp = Fvp
    else:
        Mvp = Hvp
        criterion = nn.CrossEntropyLoss(reduction='sum')
        kwargs['criterion'] = criterion

    trace = 0.0
    #params = list(model.parameters()) if pnames is None else [p for n, p in model.named_parameters() if n in pnames]
    if pnames is None:
        params = list([param for param in model.parameters() if param.requires_grad == True])
    else:
        params = [p for n, p in model.named_parameters() if (n in pnames and p.requires_grad == True)]
        
    N = sum(p.numel() for p in params)
    
    for _ in range(n_vecs):
        vec = torch.randn(N, device=params[0].device)
        vec /= torch.norm(vec)
        vec = unflatten_like(vec, params)
        M_v = eval_mvp(Mvp, vec, params, model, dataloader, **kwargs)
        for m, v in zip(M_v, vec):
            trace += (m * v).sum().item()
    
    return trace / n_vecs

def eval_model(model, dataloader, train_mode=False, return_mean=True):
    "Evaluate a model on test or train set: compute CEL and accuracy."
    if train_mode:
        model.train()
    else:
        model.eval()
    
    loss = 0.0
    correct = 0
    
    criterion = nn.CrossEntropyLoss(reduction="sum")
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            labels = labels.cuda()
            
            outputs = model(images)
            predicted = outputs.argmax(-1)
            
            loss += criterion(outputs, labels).item()
            correct += (predicted == labels).sum().item()

    if return_mean:
        loss /= len(dataloader.dataset)
        correct /= len(dataloader.dataset)
    
    return loss, correct 

def calc_grads(model, dataloader, train_mode=False, return_numpy=False, pnames=None):
    "Calculate batch gradients of the model."
    if train_mode:
        model.train()
    else:
        model.eval()

    if pnames is None:
        params = list([param for param in model.parameters() if param.requires_grad == True])
    else:
        params = [p for n, p in model.named_parameters() if (n in pnames and p.requires_grad == True)]
        
    grads_list = []
    criterion = nn.CrossEntropyLoss()
    
    for images, labels in dataloader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = criterion(outputs, labels)
        grads = torch.autograd.grad(loss, params)
        if return_numpy:
            grads = np.concatenate([g.cpu().numpy().ravel() for g in grads])
        grads_list.append(grads)

    return grads_list

def calc_grads_norms_small_memory(model, dataloader, train_mode=False, return_numpy=False, pnames=None):
    "Calculate batch gradients of the model."
    if train_mode:
        model.train()
    else:
        model.eval()

    if pnames is None:
        params = list([param for param in model.parameters() if param.requires_grad == True])
    else:
        params = [p for n, p in model.named_parameters() if (n in pnames and p.requires_grad == True)]
        
    grads_list = []
    criterion = nn.CrossEntropyLoss()
    
    gm = []
    for images, labels in dataloader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = criterion(outputs, labels)
        grads = torch.autograd.grad(loss, params)
        gm.append(np.sqrt(sum((t**2).sum().item() for t in grads)))
        
    return np.array(gm).mean()


