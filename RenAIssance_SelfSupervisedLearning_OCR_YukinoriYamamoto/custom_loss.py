import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize


def cosine_similarity(x, y):
    x = normalize(x, dim=2)
    y = normalize(y, dim=2)
    return torch.bmm(x, y.transpose(1, 2))


def noise_contrastive_estimation(x, y):
    shuffle_indices = torch.randperm(x.shape[1])
    x = x[:, shuffle_indices, :]
    target = torch.where(shuffle_indices == torch.arange(shuffle_indices.shape[0]).unsqueeze(1))[1]
    target = target.unsqueeze(0).expand(x.shape[0], -1).to(x.device)
    sim_output = cosine_similarity(x, y).to(x.device)
    cross_entropy = CrossEntropyLoss()
    # target = torch.arange(sim_output.shape[1]).unsqueeze(0).expand(sim_output.shape[0], -1).to(x.device)
    loss = cross_entropy(sim_output / 0.2, target)
    return loss


def contrastive_loss(x, y):
    return noise_contrastive_estimation(x, y) + noise_contrastive_estimation(y, x)
