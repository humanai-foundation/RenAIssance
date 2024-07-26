import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize


def cosine_similarity(x, y):
    x = normalize(x, dim=1)
    y = normalize(y, dim=1)
    return torch.mm(x, y.t())


def noise_contrastive_estimation(x, y):
    sim_output = cosine_similarity(x, y).to(x.device)
    cross_entropy = CrossEntropyLoss()
    loss = cross_entropy(sim_output / torch.sqrt(torch.tensor(x.shape[1]).to(x.device)), torch.arange(sim_output.shape[0]).to(x.device))
    return loss


def contrastive_loss(x, y):
    return noise_contrastive_estimation(x, y) + noise_contrastive_estimation(y, x)
