#!/usr/bin/env bash
set -euo pipefail

SPARC_RESIDUAL_WEIGHT=${SPARC_RESIDUAL_WEIGHT:-0.1}
SPARC_EMBED_MODE=${SPARC_EMBED_MODE:-hidden_resid}
SPGCL_EXTRA_ARGS=${SPGCL_EXTRA_ARGS:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPGCL_ROOT=${SPGCL_ROOT:-"../../third_party_baselines/SPGCL"}
SPGCL_ROOT_ABS="$(cd "${GRACE_DIR}" && cd "${SPGCL_ROOT}" && pwd)"

cd "${GRACE_DIR}"
python patch_spgcl_sparc.py --spgcl-root "${SPGCL_ROOT_ABS}"

SPGCL_ARTIFACT_METHOD=${SPGCL_ARTIFACT_METHOD:-spgcl_sparc_residual}

SPGCL_ARTIFACT_METHOD="${SPGCL_ARTIFACT_METHOD}" \
SPGCL_EXTRA_ARGS="--sparc_residual_weight ${SPARC_RESIDUAL_WEIGHT} --sparc_embed_mode ${SPARC_EMBED_MODE} ${SPGCL_EXTRA_ARGS}" \
  bash scripts/run_spgcl_embedding_export.sh
