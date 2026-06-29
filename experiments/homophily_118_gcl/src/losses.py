import torch
import torch.nn.functional as F


def sampled_contrastive(anchor, target, pos_idx, neg_idx, tau, neg_weight=None):
    anchor = F.normalize(anchor, dim=1)
    target = F.normalize(target, dim=1)
    positives = target[pos_idx]
    if positives.dim() == 2:
        positives = positives.unsqueeze(1)
    negatives = target[neg_idx.reshape(-1)].view(neg_idx.size(0), neg_idx.size(1), -1)
    pos_logits = (anchor.unsqueeze(1) * positives).sum(dim=2) / float(tau)
    neg_logits = (anchor.unsqueeze(1) * negatives).sum(dim=2) / float(tau)
    pos_term = torch.logsumexp(pos_logits, dim=1)
    if neg_weight is None:
        neg_term = torch.logsumexp(neg_logits, dim=1)
    else:
        weights = neg_weight.clamp_min(1e-8).to(neg_logits.dtype)
        neg_term = torch.logsumexp(neg_logits + weights.log(), dim=1)
    denom = torch.logaddexp(pos_term, neg_term)
    return -(pos_term - denom).mean()


def sample_negative_indices(num_nodes, num_negatives, device):
    num_negatives = min(int(num_negatives), max(1, num_nodes - 1))
    row = torch.arange(num_nodes, device=device).view(-1, 1)
    neg = torch.randint(0, max(1, num_nodes - 1), (num_nodes, num_negatives), device=device)
    return neg + (neg >= row).long()
