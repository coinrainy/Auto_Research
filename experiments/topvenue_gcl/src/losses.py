import torch
import torch.nn.functional as F

from .utils import off_diagonal


def info_nce_loss(z1, z2, tau):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    refl_sim_1 = torch.exp(z1 @ z1.t() / tau)
    refl_sim_2 = torch.exp(z2 @ z2.t() / tau)
    between_sim = torch.exp(z1 @ z2.t() / tau)
    positive = between_sim.diag()
    loss_1 = -torch.log(
        positive
        / (refl_sim_1.sum(1) + between_sim.sum(1) - refl_sim_1.diag())
    )
    loss_2 = -torch.log(
        positive
        / (refl_sim_2.sum(1) + between_sim.sum(0) - refl_sim_2.diag())
    )
    return 0.5 * (loss_1 + loss_2).mean()


def sampled_info_nce(anchor, positive, negatives, tau):
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negatives = F.normalize(negatives, dim=2)
    pos_logit = (anchor * positive).sum(dim=1, keepdim=True) / tau
    neg_logits = torch.einsum("nd,nkd->nk", anchor, negatives) / tau
    logits = torch.cat([pos_logit, neg_logits], dim=1)
    labels = torch.zeros(anchor.size(0), device=anchor.device, dtype=torch.long)
    return F.cross_entropy(logits, labels)


def negative_cosine(pred, target):
    pred = F.normalize(pred, dim=1)
    target = F.normalize(target.detach(), dim=1)
    return -(pred * target).sum(dim=1).mean()


def weighted_negative_cosine(pred, target, weight):
    pred = F.normalize(pred, dim=1)
    target = F.normalize(target.detach(), dim=1)
    loss = -(pred * target).sum(dim=1)
    weight = weight.detach().to(loss.device, dtype=loss.dtype)
    weight = weight / weight.mean().clamp_min(1e-12)
    return (loss * weight).mean()


def variance_loss(z, gamma=1.0, eps=1e-4):
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z):
    z = z - z.mean(dim=0)
    denom = max(1, z.size(0) - 1)
    cov = (z.t() @ z) / denom
    return off_diagonal(cov).pow(2).sum() / z.size(1)


def vicreg_regularizer(z):
    return variance_loss(z), covariance_loss(z)
