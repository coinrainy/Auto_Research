#!/usr/bin/env bash
set -euo pipefail

DATASETS=${DATASETS:-"Chameleon Squirrel"}
SPGCL_ROOT=${SPGCL_ROOT:-"../../third_party_baselines/SPGCL"}
PYG_ROOT=${PYG_ROOT:-"../../data"}
OUT_DIR=${OUT_DIR:-"runs/spgcl_official_embeddings"}
RESET_EPOCHS=${RESET_EPOCHS:-100}
LINEAR_EPOCHS=${LINEAR_EPOCHS:-10}
RESET_HIDDEN=${RESET_HIDDEN:-256}
RESET_SEED_NUM=${RESET_SEED_NUM:-32}
RESET_MAX_SIZE=${RESET_MAX_SIZE:-512}
RESET_SUBG_NUM_HOPS=${RESET_SUBG_NUM_HOPS:-2}
NEG_SELECTION=${NEG_SELECTION:-random}
SEED=${SEED:-42}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPGCL_ROOT_ABS="$(cd "${GRACE_DIR}" && cd "${SPGCL_ROOT}" && pwd)"
OUT_DIR_ABS="$(cd "${GRACE_DIR}" && mkdir -p "${OUT_DIR}" && cd "${OUT_DIR}" && pwd)"

cd "${GRACE_DIR}"
python export_spgcl_geom_data.py \
  --datasets ${DATASETS} \
  --pyg-root "${PYG_ROOT}" \
  --out-root "${SPGCL_ROOT_ABS}"

python - "${SPGCL_ROOT_ABS}/src/main.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
if "SPGCL_EMBED_OUT" in text:
    raise SystemExit(0)
needle = "    with torch.no_grad():\n        embeds = model.embed(data)\n"
insert = (
    "    with torch.no_grad():\n"
    "        embeds = model.embed(data)\n"
    "    embed_out = os.environ.get('SPGCL_EMBED_OUT')\n"
    "    if embed_out:\n"
    "        torch.save({\n"
    "            'embeddings': embeds.detach().cpu(),\n"
    "            'labels': data.y.detach().cpu(),\n"
    "            'dataset': args.dataset,\n"
    "            'args': vars(args),\n"
    "        }, embed_out)\n"
    "        print(f'[save] embeddings={embed_out}')\n"
)
if needle not in text:
    raise SystemExit(f"Could not patch SP-GCL main.py; pattern not found: {path}")
path.write_text(text.replace(needle, insert, 1))
PY

mkdir -p "${OUT_DIR_ABS}/raw" "${OUT_DIR_ABS}/artifacts"

for dataset in ${DATASETS}; do
  lower_dataset="$(printf '%s' "${dataset}" | tr '[:upper:]' '[:lower:]')"
  raw_out="${OUT_DIR_ABS}/raw/${lower_dataset}_seed${SEED}.pt"
  echo "[spgcl] dataset=${lower_dataset} raw_out=${raw_out}"
  (
    cd "${SPGCL_ROOT_ABS}"
    SPGCL_EMBED_OUT="${raw_out}" PYTHONPATH="${SPGCL_ROOT_ABS}" python src/main.py \
      --dataset "${lower_dataset}" \
      --seed "${SEED}" \
      --neg_selection "${NEG_SELECTION}" \
      --load_params 1 \
      --save_folder logs \
      --reset_epochs "${RESET_EPOCHS}" \
      --linear_epochs "${LINEAR_EPOCHS}" \
      --reset_hidden "${RESET_HIDDEN}" \
      --reset_seed_num "${RESET_SEED_NUM}" \
      --reset_max_size "${RESET_MAX_SIZE}" \
      --reset_subg_num_hops "${RESET_SUBG_NUM_HOPS}"
  )

  python - "${raw_out}" "${OUT_DIR_ABS}/artifacts" "${dataset}" "${SEED}" <<'PY'
from pathlib import Path
import json
import sys
import torch

raw_path = Path(sys.argv[1])
out_root = Path(sys.argv[2])
dataset = sys.argv[3]
seed = int(sys.argv[4])
obj = torch.load(raw_path, map_location='cpu')
run_dir = out_root / f'{dataset}_spgcl_official_seed{seed}_split0'
run_dir.mkdir(parents=True, exist_ok=True)
args = {
    'dataset': dataset,
    'method': 'spgcl_official',
    'seed': seed,
    'resolved_seed': seed,
    'split_index': 0,
    'source_raw_path': str(raw_path),
}
torch.save({'embeddings': obj['embeddings'], 'args': args}, run_dir / 'artifacts.pt')
with (run_dir / 'metadata.json').open('w') as handle:
    json.dump(args, handle, indent=2)
print(f'[artifact] {run_dir / "artifacts.pt"}')
PY
done
