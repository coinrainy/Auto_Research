import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree


def row_normalized_propagate(x, edge_index):
    edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    src, dst = edge_index
    deg = degree(dst, x.size(0), dtype=x.dtype).clamp_min(1.0)
    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src] / deg[dst].view(-1, 1))
    return out


@torch.no_grad()
def propagation_signature(x, edge_index, hops=2):
    pieces = [x.float()]
    current = x.float()
    for _ in range(int(hops)):
        current = row_normalized_propagate(current, edge_index)
        pieces.append(current)
    return F.normalize(torch.cat(pieces, dim=1), dim=1)


@torch.no_grad()
def topk_similar(keys, topk, chunk_size=512, exclude_self=True):
    keys = F.normalize(keys.float(), dim=1)
    n = keys.size(0)
    topk = min(int(topk), max(1, n - 1 if exclude_self else n))
    rows = []
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        sim = keys[start:end] @ keys.t()
        if exclude_self:
            diag = torch.arange(start, end, device=keys.device)
            sim[torch.arange(end - start, device=keys.device), diag] = -2.0
        rows.append(torch.topk(sim, k=topk, dim=1).indices)
    return torch.cat(rows, dim=0)
