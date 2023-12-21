import torch


def get_bandwidth(f, device):
    """
    Select a bandwidth for the kernel based on maximizing the leave-one-out likelihood (LOO MLE).

    :param f: The vector containing the probability scores, shape [num_samples, num_classes]
    :return: The bandwidth of the kernel
    """
    bandwidths = torch.cat((torch.logspace(start=-3, end=-1, steps=5), torch.linspace(0.2, 1, steps=5)))
    max_b = -1
    max_l = torch.finfo(torch.float).min
    n = len(f)
    for b in bandwidths:
        log_kern = get_kernel(f, b, device)
        log_fhat = torch.logsumexp(log_kern, 1) - torch.log((n-1)*b)
        l = torch.sum(log_fhat)
        if l > max_l:
            max_l = l
            max_b = b

    return max_b


def get_ece_kde(f, y, bandwidth, p=1, cal_type='canonical', device='cuda'):
    """
    Calculate an estimate of Lp calibration error.

    :param f: The vector containing the probability scores, shape [num_samples, num_classes]
    :param y: The vector containing the labels or IoU scores,
              shape [num_samples] for binary and [num_samples, num_classes] for canonical
    :param bandwidth: The bandwidth of the kernel
    :param p: The p-norm. Typically, p=1 or p=2
    :param cal_type: The type of calibration: canonical or binary
    :param device: The device type: 'cpu' or 'cuda'

    :return: An estimate of Lp calibration error
    """
    check_input(f, bandwidth)
    if cal_type == 'canonical':
        return get_ratio_canonical(f, y, bandwidth, p, device)
    else:
        return get_ratio_binary(f, y, bandwidth, p, device)


def get_ratio_canonical(f, y_onehot, bandwidth, p, device):
    assert f.shape == y_onehot.shape
    if f.shape[1] > 60:
        # Slower but more numerically stable implementation for larger number of classes
        return get_ratio_canonical_log(f, y_onehot, bandwidth, p, device)

    log_kern = get_kernel(f, bandwidth, device)
    # matrix multiplication in log space using broadcasting
    log_kern_y = torch.logsumexp(log_kern.unsqueeze(2) + torch.log(y_onehot).unsqueeze(0), dim=1)
    log_den = torch.logsumexp(log_kern, dim=1)

    log_ratio = log_kern_y - log_den.unsqueeze(-1)
    ratio = torch.exp(log_ratio)

    assert ratio.shape == f.shape
    ratio = torch.sum(torch.abs(ratio - f)**p, dim=1)

    return torch.mean(ratio)


# Note for training: Make sure there are at least two examples for every class present in the batch, otherwise
# LogsumexpBackward returns nans.
def get_ratio_canonical_log(f, y_onehot, bandwidth, p, device='cpu'):
    log_kern = get_kernel(f, bandwidth, device)
    log_y = torch.log(y_onehot)
    log_den = torch.logsumexp(log_kern, dim=1)
    final_ratio = 0
    for k in range(f.shape[1]):
        log_kern_y = log_kern + (torch.ones([f.shape[0], 1]) * log_y[:, k].unsqueeze(0))
        log_inner_ratio = torch.logsumexp(log_kern_y, dim=1) - log_den
        inner_ratio = torch.exp(log_inner_ratio)
        assert inner_ratio.shape == f[:, k].shape
        inner_diff = torch.abs(inner_ratio - f[:, k])**p
        final_ratio += inner_diff

    return torch.mean(final_ratio)


def get_ratio_binary(f, y, bandwidth, p, device):
    assert f.shape[1] == 1, "Incorrect shape of probability scores."
    log_kern = get_kernel(f, bandwidth, device)

    log_kern_y = log_kern + torch.log(y)
    log_num = torch.logsumexp(log_kern_y, dim=1)
    log_den = torch.logsumexp(log_kern, dim=1)

    log_ratio = log_num - log_den
    ratio = torch.exp(log_ratio)
    assert ratio.shape == f.squeeze(-1).shape
    ratio = torch.abs(ratio - f.squeeze(-1)) ** p

    return torch.mean(ratio)


def get_kernel(f, bandwidth, device):
    # if num_classes == 1
    if f.shape[1] == 1:
        log_kern = beta_kernel(f, f, bandwidth).squeeze()
    else:
        log_kern = dirichlet_kernel(f, bandwidth).squeeze()
    # Trick: -inf on the diagonal
    return log_kern + torch.diag(torch.finfo(torch.float).min * torch.ones(len(f))).to(device)


def beta_kernel(z, zi, bandwidth=0.1):
    p = zi / bandwidth + 1
    q = (1-zi) / bandwidth + 1
    z = z.unsqueeze(-2)

    log_beta = torch.lgamma(p) + torch.lgamma(q) - torch.lgamma(p + q)
    log_num = (p-1) * torch.log(z) + (q-1) * torch.log(1-z)
    log_beta_pdf = log_num - log_beta

    return log_beta_pdf


def dirichlet_kernel(z, bandwidth=0.1):
    alphas = z / bandwidth + 1

    log_beta = (torch.sum((torch.lgamma(alphas)), dim=1) - torch.lgamma(torch.sum(alphas, dim=1)))
    log_num = torch.matmul(torch.log(z), (alphas-1).T)
    log_dir_pdf = log_num - log_beta

    return log_dir_pdf


def check_input(f, bandwidth):
    assert not torch.any(torch.isnan(f))
    assert len(f.shape) == 2
    assert bandwidth > 0
    assert torch.min(f) >= 0
    assert torch.max(f) <= 1
