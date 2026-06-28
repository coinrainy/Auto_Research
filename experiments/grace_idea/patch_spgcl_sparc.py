import argparse
from pathlib import Path


PATCH_MARKER = 'SPARC_RESIDUAL_PATCH'


def replace_once(text, old, new, path):
    if old not in text:
        raise SystemExit(f'Patch pattern not found in {path}: {old[:80]!r}')
    return text.replace(old, new, 1)


def patch_model(path):
    text = path.read_text()
    if PATCH_MARKER in text:
        return False

    text = replace_once(
        text,
        "class SPGCL(nn.Module):\n",
        f"# {PATCH_MARKER}: residual auxiliary objective and embedding mode.\n"
        "class SPGCL(nn.Module):\n",
        path,
    )
    text = replace_once(
        text,
        "\n\n    def projection(self, z: torch.Tensor):\n"
        "        return self.fc_pipe(z)\n",
        "\n"
        "        self.sparc_aux_pipe = None\n"
        "        if getattr(self.args, 'sparc_residual_weight', 0.0) > 0.0:\n"
        "            if self.args.num_proj_layer == 1:\n"
        "                self.sparc_aux_pipe = torch.nn.Sequential(\n"
        "                    torch.nn.Linear(args.hidden, args.hidden)\n"
        "                )\n"
        "            elif self.args.num_proj_layer == 2:\n"
        "                self.sparc_aux_pipe = torch.nn.Sequential(\n"
        "                    torch.nn.Linear(args.hidden, args.hidden),\n"
        "                    get_activation(args.proj_activation),\n"
        "                    torch.nn.Linear(args.hidden, args.hidden)\n"
        "                )\n"
        "            elif self.args.num_proj_layer == 3:\n"
        "                self.sparc_aux_pipe = torch.nn.Sequential(\n"
        "                    torch.nn.Linear(args.hidden, args.hidden),\n"
        "                    get_activation(args.proj_activation),\n"
        "                    torch.nn.Linear(args.hidden, args.hidden),\n"
        "                    get_activation(args.proj_activation),\n"
        "                    torch.nn.Linear(args.hidden, args.hidden),\n"
        "                )\n"
        "\n"
        "\n"
        "    def projection(self, z: torch.Tensor):\n"
        "        return self.fc_pipe(z)\n"
        "\n"
        "\n"
        "    def auxiliary_projection(self, z: torch.Tensor):\n"
        "        if self.sparc_aux_pipe is None:\n"
        "            return self.projection(z)\n"
        "        return self.sparc_aux_pipe(z)\n"
        "\n"
        "\n"
        "    def propagate_embeddings(self, z: torch.Tensor, edge_index: torch.Tensor):\n"
        "        source, target = edge_index\n"
        "        aggregate = torch.zeros_like(z)\n"
        "        degree = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)\n"
        "        aggregate.index_add_(0, target, z[source])\n"
        "        degree.index_add_(0, target, torch.ones_like(target, dtype=z.dtype))\n"
        "        aggregate = aggregate + z\n"
        "        degree = degree + 1.0\n"
        "        return aggregate / degree.clamp_min(1.0).view(-1, 1)\n"
        "\n"
        "\n"
        "    def residual_embedding(self, z: torch.Tensor, edge_index: torch.Tensor):\n"
        "        return z - self.propagate_embeddings(z, edge_index)\n"
        "\n"
        "\n"
        "    def sparc_representation(self, z: torch.Tensor, edge_index: torch.Tensor, mode: str):\n"
        "        residual = self.residual_embedding(z, edge_index)\n"
        "        if mode == 'hidden':\n"
        "            return z\n"
        "        if mode == 'resid':\n"
        "            return residual\n"
        "        return torch.cat([\n"
        "            F.normalize(z, p=2, dim=-1),\n"
        "            F.normalize(residual, p=2, dim=-1),\n"
        "        ], dim=-1)\n",
        path,
    )
    text = replace_once(
        text,
        "    def embed(self, data):\n"
        "        x, edge_index = data.x, data.edge_index\n"
        "        embed = self.encoder(x, edge_index)\n"
        "        return embed.detach()\n",
        "    def embed(self, data):\n"
        "        x, edge_index = data.x, data.edge_index\n"
        "        embed = self.encoder(x, edge_index)\n"
        "        mode = getattr(self.args, 'sparc_embed_mode', 'hidden')\n"
        "        if mode != 'hidden':\n"
        "            embed = self.sparc_representation(embed, edge_index, mode)\n"
        "        return embed.detach()\n",
        path,
    )
    text = replace_once(
        text,
        "        loss = pos_part + neg_part\n"
        "\n"
        "        loss.backward()\n",
        "        loss = pos_part + neg_part\n"
        "\n"
        "        residual_weight = getattr(self.args, 'sparc_residual_weight', 0.0)\n"
        "        if residual_weight > 0.0:\n"
        "            residual_full = self.residual_embedding(orig_node_embed_full, data.edge_index)\n"
        "            if self.args.square_subg:\n"
        "                residual_sample = self.auxiliary_projection(F.relu(residual_full[sample_idx]))\n"
        "                norm_residual = F.normalize(residual_sample, p=2, dim=-1)\n"
        "                residual_sim_matrix = torch.mm(norm_residual, norm_residual.t())\n"
        "            else:\n"
        "                residual_projected = self.auxiliary_projection(F.relu(residual_full))\n"
        "                norm_residual_full = F.normalize(residual_projected, p=2, dim=-1)\n"
        "                norm_residual = norm_residual_full[sample_idx]\n"
        "                residual_sim_matrix = torch.mm(norm_residual, norm_residual_full.t())\n"
        "\n"
        "            residual_pos_score_per_node = torch.zeros(sample_idx.shape[0]).to(residual_sim_matrix.device)\n"
        "            residual_pos_score_per_node = residual_pos_score_per_node.scatter_add_(0, filter_index[:, 0], residual_sim_matrix[filter_index[:, 0], filter_index[:, 1]])\n"
        "            residual_pos_part = (-2 * residual_pos_score_per_node / per_node_count).mean()\n"
        "            residual_neg_score_per_node = torch.zeros(sample_idx.shape[0]).to(residual_sim_matrix.device)\n"
        "            residual_neg_score_per_node = residual_neg_score_per_node.scatter_add_(0, filter_index_neg[:, 0], residual_sim_matrix[filter_index_neg[:, 0], filter_index_neg[:, 1]]**2)\n"
        "            residual_neg_part = residual_neg_score_per_node.mean()\n"
        "            loss = loss + residual_weight * (residual_pos_part + residual_neg_part)\n"
        "\n"
        "        loss.backward()\n",
        path,
    )
    path.write_text(text)
    return True


def patch_main(path):
    text = path.read_text()
    changed = False
    if "--sparc_residual_weight" not in text:
        text = replace_once(
            text,
            "    parser.add_argument('--reset_hidden', type=int, default=None)\n",
            "    parser.add_argument('--reset_hidden', type=int, default=None)\n"
            "    parser.add_argument('--sparc_residual_weight', type=float, default=0.0)\n"
            "    parser.add_argument('--sparc_embed_mode', type=str, default='hidden',\n"
            "                        choices=['hidden', 'resid', 'hidden_resid'])\n",
            path,
        )
        changed = True
    if 'args.sparc_residual_weight' not in text.split("title = '{}_{}_")[0]:
        # Keep the run title unchanged so old log parsing remains stable.
        pass
    path.write_text(text)
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spgcl-root', required=True)
    args = parser.parse_args()
    root = Path(args.spgcl_root).resolve()
    model_path = root / 'src' / 'Non_Homophily_GCL.py'
    main_path = root / 'src' / 'main.py'
    changed_model = patch_model(model_path)
    changed_main = patch_main(main_path)
    print(
        f'(I) patched SP-GCL SPARC: model_changed={changed_model}, '
        f'main_changed={changed_main}, root={root}'
    )


if __name__ == '__main__':
    main()
