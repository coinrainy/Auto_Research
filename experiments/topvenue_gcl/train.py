import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from torch_geometric.utils import dropout_edge
except ImportError:  # pragma: no cover
    from torch_geometric.utils import dropout_adj as dropout_edge

from src.data import graph_stats, load_dataset, should_use_mask_eval, split_masks
from src.eval import linear_probe_random, linear_probe_with_masks
from src.losses import (
    info_nce_loss,
    covariance_loss,
    multi_positive_info_nce,
    multi_positive_info_nce_per_node,
    negative_cosine,
    negative_cosine_per_node,
    sampled_info_nce,
    variance_loss,
    vicreg_regularizer,
    weighted_negative_cosine,
)
from src.models import EnergyRoutedCacheGCL, GraceModel
from src.utils import (
    append_csv,
    ensure_dir,
    feature_drop,
    load_yaml,
    propagation_signature,
    row_normalized_propagate,
    set_seed,
    topk_cache_indices,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", default="energy_spgcl",
                        choices=[
                            "grace",
                            "danv_gcl",
                            "danv_degree_gcl",
                            "fdnv_gcl",
                            "sspnv_gcl",
                            "afpnv_gcl",
                            "bspnv_gcl",
                            "mpnv_gcl",
                            "aompnv_gcl",
                            "srgnv_gcl",
                            "pcnv_gcl",
                            "lcos_gcl",
                            "lcm_gcl",
                            "dsp_gcl",
                            "rrnv_gcl",
                            "darrnv_gcl",
                            "dsrrnv_gcl",
                            "dirrnv_gcl",
                            "dprrnv_gcl",
                            "nprrnv_gcl",
                            "rwirrnv_gcl",
                            "eairrnv_gcl",
                            "bprrnv_gcl",
                            "tns_gcl",
                            "ragc_gcl",
                            "energy_spgcl",
                            "gcn_mlp_gcl",
                            "raw_features",
                            "er_residual_gcl",
                            "er_cache_gcl",
                        ])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--eval-mode", default=None,
                        choices=[None, "auto", "mask", "random"])
    parser.add_argument("--final-repr", default=None,
                        choices=[None, "ego", "graph", "high", "ego_high", "ego_graph"])
    parser.add_argument("--cache-topk", type=int, default=None)
    parser.add_argument("--cache-key-mode", default=None,
                        choices=[None, "raw_low", "raw_signature", "learned_low"])
    parser.add_argument("--cache-update-interval", type=int, default=None)
    parser.add_argument("--danv-alignment-weight", type=float, default=None)
    parser.add_argument("--danv-disagreement-weight", type=float, default=None)
    parser.add_argument("--danv-gate-temperature", type=float, default=None)
    parser.add_argument("--danv-min-align-weight", type=float, default=None)
    parser.add_argument("--danv-degree-threshold", type=float, default=None)
    parser.add_argument("--danv-degree-temperature", type=float, default=None)
    parser.add_argument("--fdnv-route-weight", type=float, default=None)
    parser.add_argument("--fdnv-bootstrap-weight", type=float, default=None)
    parser.add_argument("--fdnv-filter-temperature", type=float, default=None)
    parser.add_argument("--fdnv-min-filter-weight", type=float, default=None)
    parser.add_argument("--sspnv-semantic-weight", type=float, default=None)
    parser.add_argument("--sspnv-spatial-weight", type=float, default=None)
    parser.add_argument("--sspnv-bootstrap-weight", type=float, default=None)
    parser.add_argument("--sspnv-semantic-topk", type=int, default=None)
    parser.add_argument("--sspnv-random-semantic", action="store_true")
    parser.add_argument("--sspnv-random-spatial", action="store_true")
    parser.add_argument("--afpnv-semantic-conf-threshold", type=float, default=None)
    parser.add_argument("--afpnv-spatial-conf-threshold", type=float, default=None)
    parser.add_argument("--afpnv-conf-temperature", type=float, default=None)
    parser.add_argument("--afpnv-min-branch-weight", type=float, default=None)
    parser.add_argument("--bspnv-branch-temperature", type=float, default=None)
    parser.add_argument("--bspnv-bootstrap-bias", type=float, default=None)
    parser.add_argument("--mpnv-semantic-weight", type=float, default=None)
    parser.add_argument("--mpnv-spatial-weight", type=float, default=None)
    parser.add_argument("--mpnv-bootstrap-weight", type=float, default=None)
    parser.add_argument("--mpnv-shuffle-positives", action="store_true")
    parser.add_argument("--aompnv-router-temperature", type=float, default=None)
    parser.add_argument("--aompnv-min-branch-prob", type=float, default=None)
    parser.add_argument("--aompnv-confidence-weight", type=float, default=None)
    parser.add_argument("--aompnv-semantic-weight", type=float, default=None)
    parser.add_argument("--aompnv-spatial-weight", type=float, default=None)
    parser.add_argument("--aompnv-bootstrap-weight", type=float, default=None)
    parser.add_argument("--aompnv-shuffle-positives", action="store_true")
    parser.add_argument("--srgnv-base-weight", type=float, default=None)
    parser.add_argument("--srgnv-residual-weight", type=float, default=None)
    parser.add_argument("--srgnv-residual-threshold", type=float, default=None)
    parser.add_argument("--srgnv-residual-temperature", type=float, default=None)
    parser.add_argument("--srgnv-min-residual-weight", type=float, default=None)
    parser.add_argument("--srgnv-shuffle-residual", action="store_true")
    parser.add_argument("--pcnv-num-prototypes", type=int, default=None)
    parser.add_argument("--pcnv-base-weight", type=float, default=None)
    parser.add_argument("--pcnv-prototype-weight", type=float, default=None)
    parser.add_argument("--pcnv-balance-weight", type=float, default=None)
    parser.add_argument("--pcnv-assignment-temperature", type=float, default=None)
    parser.add_argument("--pcnv-target-temperature", type=float, default=None)
    parser.add_argument("--pcnv-min-target-confidence", type=float, default=None)
    parser.add_argument("--pcnv-confidence-power", type=float, default=None)
    parser.add_argument("--pcnv-min-view-agreement", type=float, default=None)
    parser.add_argument("--pcnv-view-agreement-power", type=float, default=None)
    parser.add_argument("--pcnv-entropy-guard", action="store_true")
    parser.add_argument("--pcnv-min-usage-entropy-frac", type=float, default=None)
    parser.add_argument("--pcnv-entropy-guard-temperature", type=float, default=None)
    parser.add_argument("--pcnv-shuffle-assignments", action="store_true")
    parser.add_argument("--lcos-route-temperature", type=float, default=None)
    parser.add_argument("--lcos-route-threshold", type=float, default=None)
    parser.add_argument("--lcos-min-branch-weight", type=float, default=None)
    parser.add_argument("--lcos-degree-weight", type=float, default=None)
    parser.add_argument("--lcos-shuffle-gate", action="store_true")
    parser.add_argument("--dsp-margin-topk", type=int, default=None)
    parser.add_argument("--dsp-margin-temperature", type=float, default=None)
    parser.add_argument("--dsp-min-weight", type=float, default=None)
    parser.add_argument("--dsp-view-weight", type=float, default=None)
    parser.add_argument("--dsp-shuffle-weight", action="store_true")
    parser.add_argument("--rrnv-invariance-weight", type=float, default=None)
    parser.add_argument("--rrnv-variance-weight", type=float, default=None)
    parser.add_argument("--rrnv-covariance-weight", type=float, default=None)
    parser.add_argument("--rrnv-shuffle-pairs", action="store_true")
    parser.add_argument("--darrnv-rr-weight", type=float, default=None)
    parser.add_argument("--darrnv-degree-threshold", type=float, default=None)
    parser.add_argument("--darrnv-degree-temperature", type=float, default=None)
    parser.add_argument("--dsrrnv-degree-threshold", type=float, default=None)
    parser.add_argument("--dsrrnv-degree-temperature", type=float, default=None)
    parser.add_argument("--dsrrnv-min-high-gate", type=float, default=None)
    parser.add_argument("--dsrrnv-max-high-gate", type=float, default=None)
    parser.add_argument("--dirrnv-invariance-power", type=float, default=None)
    parser.add_argument("--dirrnv-min-invariance-scale", type=float, default=None)
    parser.add_argument("--dprrnv-shuffle-power", type=float, default=None)
    parser.add_argument("--dprrnv-min-shuffle-prob", type=float, default=None)
    parser.add_argument("--dprrnv-max-shuffle-prob", type=float, default=None)
    parser.add_argument("--nprrnv-route-temperature", type=float, default=None)
    parser.add_argument("--nprrnv-route-threshold", type=float, default=None)
    parser.add_argument("--nprrnv-min-local-scale", type=float, default=None)
    parser.add_argument("--nprrnv-min-shuffle-prob", type=float, default=None)
    parser.add_argument("--nprrnv-max-shuffle-prob", type=float, default=None)
    parser.add_argument("--nprrnv-degree-weight", type=float, default=None)
    parser.add_argument("--nprrnv-residual-weight", type=float, default=None)
    parser.add_argument("--nprrnv-agreement-weight", type=float, default=None)
    parser.add_argument("--nprrnv-view-weight", type=float, default=None)
    parser.add_argument("--nprrnv-shuffle-gate", action="store_true")
    parser.add_argument("--rwirrnv-min-reliability", type=float, default=None)
    parser.add_argument("--rwirrnv-weight-power", type=float, default=None)
    parser.add_argument("--rwirrnv-shuffle-weight", action="store_true")
    parser.add_argument("--rwirrnv-constant-weight", action="store_true")
    parser.add_argument("--eairrnv-energy-threshold", type=float, default=None)
    parser.add_argument("--eairrnv-strength", type=float, default=None)
    parser.add_argument("--eairrnv-power", type=float, default=None)
    parser.add_argument("--eairrnv-min-invariance-scale", type=float, default=None)
    parser.add_argument("--bprrnv-rr-weight", type=float, default=None)
    parser.add_argument("--bprrnv-invariance-weight", type=float, default=None)
    parser.add_argument("--bprrnv-variance-weight", type=float, default=None)
    parser.add_argument("--bprrnv-covariance-weight", type=float, default=None)
    parser.add_argument("--bprrnv-degree-threshold", type=float, default=None)
    parser.add_argument("--bprrnv-degree-temperature", type=float, default=None)
    parser.add_argument("--bprrnv-energy-threshold", type=float, default=None)
    parser.add_argument("--bprrnv-energy-strength", type=float, default=None)
    parser.add_argument("--bprrnv-energy-power", type=float, default=None)
    parser.add_argument("--bprrnv-min-energy-factor", type=float, default=None)
    parser.add_argument("--bprrnv-uniform-gate", action="store_true")
    parser.add_argument("--bprrnv-no-density-gate", action="store_true")
    parser.add_argument("--bprrnv-no-energy-gate", action="store_true")
    parser.add_argument("--tns-weight", type=float, default=None)
    parser.add_argument("--tns-num-negatives", type=int, default=None)
    parser.add_argument("--tns-margin", type=float, default=None)
    parser.add_argument("--tns-temperature", type=float, default=None)
    parser.add_argument("--tns-key-threshold", type=float, default=None)
    parser.add_argument("--tns-key-temperature", type=float, default=None)
    parser.add_argument("--tns-min-weight", type=float, default=None)
    parser.add_argument("--tns-shuffle-weight", action="store_true")
    parser.add_argument("--tns-uniform-weight", action="store_true")
    parser.add_argument("--ragc-raw-weight", type=float, default=None)
    parser.add_argument("--ragc-learned-weight", type=float, default=None)
    parser.add_argument("--ragc-control", default=None,
                        choices=[None, "normal", "shuffle", "random"])
    parser.add_argument("--shuffle-cache", action="store_true")
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def get_device(gpu_id):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def override_config(config, args):
    merged = dict(config)
    if args.seed is not None:
        merged["seed"] = args.seed
    if args.epochs is not None:
        merged["epochs"] = args.epochs
    if args.eval_mode is not None:
        merged["eval_mode"] = args.eval_mode
    if args.final_repr is not None:
        merged["final_repr"] = args.final_repr
    if args.cache_topk is not None:
        merged["cache_topk"] = args.cache_topk
    if args.cache_key_mode is not None:
        merged["cache_key_mode"] = args.cache_key_mode
    if args.cache_update_interval is not None:
        merged["cache_update_interval"] = args.cache_update_interval
    if args.danv_alignment_weight is not None:
        merged["danv_alignment_weight"] = args.danv_alignment_weight
    if args.danv_disagreement_weight is not None:
        merged["danv_disagreement_weight"] = args.danv_disagreement_weight
    if args.danv_gate_temperature is not None:
        merged["danv_gate_temperature"] = args.danv_gate_temperature
    if args.danv_min_align_weight is not None:
        merged["danv_min_align_weight"] = args.danv_min_align_weight
    if args.danv_degree_threshold is not None:
        merged["danv_degree_threshold"] = args.danv_degree_threshold
    if args.danv_degree_temperature is not None:
        merged["danv_degree_temperature"] = args.danv_degree_temperature
    if args.fdnv_route_weight is not None:
        merged["fdnv_route_weight"] = args.fdnv_route_weight
    if args.fdnv_bootstrap_weight is not None:
        merged["fdnv_bootstrap_weight"] = args.fdnv_bootstrap_weight
    if args.fdnv_filter_temperature is not None:
        merged["fdnv_filter_temperature"] = args.fdnv_filter_temperature
    if args.fdnv_min_filter_weight is not None:
        merged["fdnv_min_filter_weight"] = args.fdnv_min_filter_weight
    if args.sspnv_semantic_weight is not None:
        merged["sspnv_semantic_weight"] = args.sspnv_semantic_weight
    if args.sspnv_spatial_weight is not None:
        merged["sspnv_spatial_weight"] = args.sspnv_spatial_weight
    if args.sspnv_bootstrap_weight is not None:
        merged["sspnv_bootstrap_weight"] = args.sspnv_bootstrap_weight
    if args.sspnv_semantic_topk is not None:
        merged["sspnv_semantic_topk"] = args.sspnv_semantic_topk
    if args.sspnv_random_semantic:
        merged["sspnv_random_semantic"] = True
    if args.sspnv_random_spatial:
        merged["sspnv_random_spatial"] = True
    if args.afpnv_semantic_conf_threshold is not None:
        merged["afpnv_semantic_conf_threshold"] = args.afpnv_semantic_conf_threshold
    if args.afpnv_spatial_conf_threshold is not None:
        merged["afpnv_spatial_conf_threshold"] = args.afpnv_spatial_conf_threshold
    if args.afpnv_conf_temperature is not None:
        merged["afpnv_conf_temperature"] = args.afpnv_conf_temperature
    if args.afpnv_min_branch_weight is not None:
        merged["afpnv_min_branch_weight"] = args.afpnv_min_branch_weight
    if args.bspnv_branch_temperature is not None:
        merged["bspnv_branch_temperature"] = args.bspnv_branch_temperature
    if args.bspnv_bootstrap_bias is not None:
        merged["bspnv_bootstrap_bias"] = args.bspnv_bootstrap_bias
    if args.mpnv_semantic_weight is not None:
        merged["mpnv_semantic_weight"] = args.mpnv_semantic_weight
    if args.mpnv_spatial_weight is not None:
        merged["mpnv_spatial_weight"] = args.mpnv_spatial_weight
    if args.mpnv_bootstrap_weight is not None:
        merged["mpnv_bootstrap_weight"] = args.mpnv_bootstrap_weight
    if args.mpnv_shuffle_positives:
        merged["mpnv_shuffle_positives"] = True
    if args.aompnv_router_temperature is not None:
        merged["aompnv_router_temperature"] = args.aompnv_router_temperature
    if args.aompnv_min_branch_prob is not None:
        merged["aompnv_min_branch_prob"] = args.aompnv_min_branch_prob
    if args.aompnv_confidence_weight is not None:
        merged["aompnv_confidence_weight"] = args.aompnv_confidence_weight
    if args.aompnv_semantic_weight is not None:
        merged["aompnv_semantic_weight"] = args.aompnv_semantic_weight
    if args.aompnv_spatial_weight is not None:
        merged["aompnv_spatial_weight"] = args.aompnv_spatial_weight
    if args.aompnv_bootstrap_weight is not None:
        merged["aompnv_bootstrap_weight"] = args.aompnv_bootstrap_weight
    if args.aompnv_shuffle_positives:
        merged["aompnv_shuffle_positives"] = True
    if args.srgnv_base_weight is not None:
        merged["srgnv_base_weight"] = args.srgnv_base_weight
    if args.srgnv_residual_weight is not None:
        merged["srgnv_residual_weight"] = args.srgnv_residual_weight
    if args.srgnv_residual_threshold is not None:
        merged["srgnv_residual_threshold"] = args.srgnv_residual_threshold
    if args.srgnv_residual_temperature is not None:
        merged["srgnv_residual_temperature"] = args.srgnv_residual_temperature
    if args.srgnv_min_residual_weight is not None:
        merged["srgnv_min_residual_weight"] = args.srgnv_min_residual_weight
    if args.srgnv_shuffle_residual:
        merged["srgnv_shuffle_residual"] = True
    if args.pcnv_num_prototypes is not None:
        merged["pcnv_num_prototypes"] = args.pcnv_num_prototypes
    if args.pcnv_base_weight is not None:
        merged["pcnv_base_weight"] = args.pcnv_base_weight
    if args.pcnv_prototype_weight is not None:
        merged["pcnv_prototype_weight"] = args.pcnv_prototype_weight
    if args.pcnv_balance_weight is not None:
        merged["pcnv_balance_weight"] = args.pcnv_balance_weight
    if args.pcnv_assignment_temperature is not None:
        merged["pcnv_assignment_temperature"] = args.pcnv_assignment_temperature
    if args.pcnv_target_temperature is not None:
        merged["pcnv_target_temperature"] = args.pcnv_target_temperature
    if args.pcnv_min_target_confidence is not None:
        merged["pcnv_min_target_confidence"] = args.pcnv_min_target_confidence
    if args.pcnv_confidence_power is not None:
        merged["pcnv_confidence_power"] = args.pcnv_confidence_power
    if args.pcnv_min_view_agreement is not None:
        merged["pcnv_min_view_agreement"] = args.pcnv_min_view_agreement
    if args.pcnv_view_agreement_power is not None:
        merged["pcnv_view_agreement_power"] = args.pcnv_view_agreement_power
    if args.pcnv_entropy_guard:
        merged["pcnv_entropy_guard"] = True
    if args.pcnv_min_usage_entropy_frac is not None:
        merged["pcnv_min_usage_entropy_frac"] = args.pcnv_min_usage_entropy_frac
    if args.pcnv_entropy_guard_temperature is not None:
        merged["pcnv_entropy_guard_temperature"] = args.pcnv_entropy_guard_temperature
    if args.pcnv_shuffle_assignments:
        merged["pcnv_shuffle_assignments"] = True
    if args.lcos_route_temperature is not None:
        merged["lcos_route_temperature"] = args.lcos_route_temperature
    if args.lcos_route_threshold is not None:
        merged["lcos_route_threshold"] = args.lcos_route_threshold
    if args.lcos_min_branch_weight is not None:
        merged["lcos_min_branch_weight"] = args.lcos_min_branch_weight
    if args.lcos_degree_weight is not None:
        merged["lcos_degree_weight"] = args.lcos_degree_weight
    if args.lcos_shuffle_gate:
        merged["lcos_shuffle_gate"] = True
    if args.dsp_margin_topk is not None:
        merged["dsp_margin_topk"] = args.dsp_margin_topk
    if args.dsp_margin_temperature is not None:
        merged["dsp_margin_temperature"] = args.dsp_margin_temperature
    if args.dsp_min_weight is not None:
        merged["dsp_min_weight"] = args.dsp_min_weight
    if args.dsp_view_weight is not None:
        merged["dsp_view_weight"] = args.dsp_view_weight
    if args.dsp_shuffle_weight:
        merged["dsp_shuffle_weight"] = True
    if args.rrnv_invariance_weight is not None:
        merged["rrnv_invariance_weight"] = args.rrnv_invariance_weight
    if args.rrnv_variance_weight is not None:
        merged["rrnv_variance_weight"] = args.rrnv_variance_weight
    if args.rrnv_covariance_weight is not None:
        merged["rrnv_covariance_weight"] = args.rrnv_covariance_weight
    if args.rrnv_shuffle_pairs:
        merged["rrnv_shuffle_pairs"] = True
    if args.darrnv_rr_weight is not None:
        merged["darrnv_rr_weight"] = args.darrnv_rr_weight
    if args.darrnv_degree_threshold is not None:
        merged["darrnv_degree_threshold"] = args.darrnv_degree_threshold
    if args.darrnv_degree_temperature is not None:
        merged["darrnv_degree_temperature"] = args.darrnv_degree_temperature
    if args.dsrrnv_degree_threshold is not None:
        merged["dsrrnv_degree_threshold"] = args.dsrrnv_degree_threshold
    if args.dsrrnv_degree_temperature is not None:
        merged["dsrrnv_degree_temperature"] = args.dsrrnv_degree_temperature
    if args.dsrrnv_min_high_gate is not None:
        merged["dsrrnv_min_high_gate"] = args.dsrrnv_min_high_gate
    if args.dsrrnv_max_high_gate is not None:
        merged["dsrrnv_max_high_gate"] = args.dsrrnv_max_high_gate
    if args.dirrnv_invariance_power is not None:
        merged["dirrnv_invariance_power"] = args.dirrnv_invariance_power
    if args.dirrnv_min_invariance_scale is not None:
        merged["dirrnv_min_invariance_scale"] = args.dirrnv_min_invariance_scale
    if args.dprrnv_shuffle_power is not None:
        merged["dprrnv_shuffle_power"] = args.dprrnv_shuffle_power
    if args.dprrnv_min_shuffle_prob is not None:
        merged["dprrnv_min_shuffle_prob"] = args.dprrnv_min_shuffle_prob
    if args.dprrnv_max_shuffle_prob is not None:
        merged["dprrnv_max_shuffle_prob"] = args.dprrnv_max_shuffle_prob
    if args.nprrnv_route_temperature is not None:
        merged["nprrnv_route_temperature"] = args.nprrnv_route_temperature
    if args.nprrnv_route_threshold is not None:
        merged["nprrnv_route_threshold"] = args.nprrnv_route_threshold
    if args.nprrnv_min_local_scale is not None:
        merged["nprrnv_min_local_scale"] = args.nprrnv_min_local_scale
    if args.nprrnv_min_shuffle_prob is not None:
        merged["nprrnv_min_shuffle_prob"] = args.nprrnv_min_shuffle_prob
    if args.nprrnv_max_shuffle_prob is not None:
        merged["nprrnv_max_shuffle_prob"] = args.nprrnv_max_shuffle_prob
    if args.nprrnv_degree_weight is not None:
        merged["nprrnv_degree_weight"] = args.nprrnv_degree_weight
    if args.nprrnv_residual_weight is not None:
        merged["nprrnv_residual_weight"] = args.nprrnv_residual_weight
    if args.nprrnv_agreement_weight is not None:
        merged["nprrnv_agreement_weight"] = args.nprrnv_agreement_weight
    if args.nprrnv_view_weight is not None:
        merged["nprrnv_view_weight"] = args.nprrnv_view_weight
    if args.nprrnv_shuffle_gate:
        merged["nprrnv_shuffle_gate"] = True
    if args.rwirrnv_min_reliability is not None:
        merged["rwirrnv_min_reliability"] = args.rwirrnv_min_reliability
    if args.rwirrnv_weight_power is not None:
        merged["rwirrnv_weight_power"] = args.rwirrnv_weight_power
    if args.rwirrnv_shuffle_weight:
        merged["rwirrnv_shuffle_weight"] = True
    if args.rwirrnv_constant_weight:
        merged["rwirrnv_constant_weight"] = True
    if args.eairrnv_energy_threshold is not None:
        merged["eairrnv_energy_threshold"] = args.eairrnv_energy_threshold
    if args.eairrnv_strength is not None:
        merged["eairrnv_strength"] = args.eairrnv_strength
    if args.eairrnv_power is not None:
        merged["eairrnv_power"] = args.eairrnv_power
    if args.eairrnv_min_invariance_scale is not None:
        merged["eairrnv_min_invariance_scale"] = args.eairrnv_min_invariance_scale
    if args.bprrnv_rr_weight is not None:
        merged["bprrnv_rr_weight"] = args.bprrnv_rr_weight
    if args.bprrnv_invariance_weight is not None:
        merged["bprrnv_invariance_weight"] = args.bprrnv_invariance_weight
    if args.bprrnv_variance_weight is not None:
        merged["bprrnv_variance_weight"] = args.bprrnv_variance_weight
    if args.bprrnv_covariance_weight is not None:
        merged["bprrnv_covariance_weight"] = args.bprrnv_covariance_weight
    if args.bprrnv_degree_threshold is not None:
        merged["bprrnv_degree_threshold"] = args.bprrnv_degree_threshold
    if args.bprrnv_degree_temperature is not None:
        merged["bprrnv_degree_temperature"] = args.bprrnv_degree_temperature
    if args.bprrnv_energy_threshold is not None:
        merged["bprrnv_energy_threshold"] = args.bprrnv_energy_threshold
    if args.bprrnv_energy_strength is not None:
        merged["bprrnv_energy_strength"] = args.bprrnv_energy_strength
    if args.bprrnv_energy_power is not None:
        merged["bprrnv_energy_power"] = args.bprrnv_energy_power
    if args.bprrnv_min_energy_factor is not None:
        merged["bprrnv_min_energy_factor"] = args.bprrnv_min_energy_factor
    if args.bprrnv_uniform_gate:
        merged["bprrnv_uniform_gate"] = True
    if args.bprrnv_no_density_gate:
        merged["bprrnv_no_density_gate"] = True
    if args.bprrnv_no_energy_gate:
        merged["bprrnv_no_energy_gate"] = True
    if args.tns_weight is not None:
        merged["tns_weight"] = args.tns_weight
    if args.tns_num_negatives is not None:
        merged["tns_num_negatives"] = args.tns_num_negatives
    if args.tns_margin is not None:
        merged["tns_margin"] = args.tns_margin
    if args.tns_temperature is not None:
        merged["tns_temperature"] = args.tns_temperature
    if args.tns_key_threshold is not None:
        merged["tns_key_threshold"] = args.tns_key_threshold
    if args.tns_key_temperature is not None:
        merged["tns_key_temperature"] = args.tns_key_temperature
    if args.tns_min_weight is not None:
        merged["tns_min_weight"] = args.tns_min_weight
    if args.tns_shuffle_weight:
        merged["tns_shuffle_weight"] = True
    if args.tns_uniform_weight:
        merged["tns_uniform_weight"] = True
    if args.ragc_raw_weight is not None:
        merged["ragc_raw_weight"] = args.ragc_raw_weight
    if args.ragc_learned_weight is not None:
        merged["ragc_learned_weight"] = args.ragc_learned_weight
    if args.ragc_control is not None:
        merged["ragc_control"] = args.ragc_control
    return merged


def make_run_dir(args, config):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = args.run_name or (
        f"{timestamp}_{args.method}_{args.dataset}_"
        f"seed{config['seed']}_split{args.split_index}"
    )
    run_dir = Path(args.runs_dir) / stem
    if run_dir.exists() and any(run_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Run directory already exists and is non-empty: {run_dir}. "
                "Use --overwrite or choose a new --run-name."
            )
        shutil.rmtree(run_dir)
    ensure_dir(run_dir)
    return run_dir


def train_grace(model, data, config, args):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        edge_1 = dropout_edge(
            data.edge_index,
            p=float(config["drop_edge_rate_1"]),
            force_undirected=False,
            training=True,
        )[0]
        edge_2 = dropout_edge(
            data.edge_index,
            p=float(config["drop_edge_rate_2"]),
            force_undirected=False,
            training=True,
        )[0]
        x_1 = feature_drop(data.x, float(config["drop_feature_rate_1"]))
        x_2 = feature_drop(data.x, float(config["drop_feature_rate_2"]))
        z_1 = model.project(model(x_1, edge_1))
        z_2 = model.project(model(x_2, edge_2))
        loss = info_nce_loss(z_1, z_2, float(config["tau"]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        row = {"epoch": epoch, "loss": float(loss.item())}
        history.append(row)
        if epoch == 1 or epoch % args.log_every == 0:
            print(f"epoch={epoch:03d} loss={row['loss']:.6f}")
    model.eval()
    with torch.no_grad():
        final = model(data.x, data.edge_index)
    return final.detach(), history, {}


def _cache_positive_mean(z, indices):
    if indices.dim() == 1:
        indices = indices.view(-1, 1)
    positives = z[indices.reshape(-1)].view(indices.size(0), indices.size(1), -1)
    return positives.mean(dim=1)


@torch.no_grad()
def _cache_confidence(cache_keys, cache_idx, config):
    if cache_keys is None:
        return None
    keys = F.normalize(cache_keys, dim=1)
    positive = keys[cache_idx[:, 0]]
    sim = (keys * positive).sum(dim=1)
    threshold = float(config["cache_confidence_threshold"])
    temperature = max(float(config["cache_confidence_temperature"]), 1e-12)
    min_weight = float(config["cache_confidence_min_weight"])
    weight = torch.sigmoid((sim - threshold) / temperature)
    weight = weight * (1.0 - min_weight) + min_weight
    return sim, weight


def _sample_positive_indices(cache_idx):
    if cache_idx.size(1) == 1:
        return cache_idx[:, 0]
    choice = torch.randint(
        low=0,
        high=cache_idx.size(1),
        size=(cache_idx.size(0),),
        device=cache_idx.device,
    )
    return cache_idx[torch.arange(cache_idx.size(0), device=cache_idx.device), choice]


def _sample_negative_indices(num_nodes, num_negatives, device):
    num_negatives = min(max(1, int(num_negatives)), max(1, num_nodes - 1))
    negatives = torch.randint(
        low=0,
        high=num_nodes,
        size=(num_nodes, num_negatives),
        device=device,
    )
    row = torch.arange(num_nodes, device=device).view(-1, 1)
    negatives = torch.where(negatives == row, (negatives + 1) % num_nodes, negatives)
    return negatives


def _standardize(vector):
    return (vector - vector.mean()) / vector.std(unbiased=False).clamp_min(1e-12)


@torch.no_grad()
def _danv_alignment_gate(data, parts, config):
    raw_low_raw = row_normalized_propagate(data.x.detach(), data.edge_index, add_self=True)
    raw = F.normalize(data.x.detach().float(), dim=1)
    raw_low = F.normalize(raw_low_raw.float(), dim=1)
    raw_agreement = (raw * raw_low).sum(dim=1)
    raw_residual = (data.x.detach().float() - raw_low_raw.float()).norm(dim=1)
    raw_scale = data.x.detach().float().norm(dim=1).clamp_min(1e-12)
    raw_residual_energy = raw_residual / raw_scale
    view_cosine = (
        F.normalize(parts["ego"].detach(), dim=1)
        * F.normalize(parts["graph"].detach(), dim=1)
    ).sum(dim=1)
    score = (
        _standardize(raw_agreement)
        + _standardize(view_cosine)
        - _standardize(raw_residual_energy)
    )
    gate = torch.sigmoid(score / max(float(config["danv_gate_temperature"]), 1e-12))
    min_weight = float(config["danv_min_align_weight"])
    return gate * (1.0 - min_weight) + min_weight


def _weighted_cosine_abs(z1, z2, weight):
    cosine = (
        F.normalize(z1, dim=1)
        * F.normalize(z2, dim=1)
    ).sum(dim=1).abs()
    weight = weight.detach().to(cosine.device, dtype=cosine.dtype)
    weight = weight / weight.mean().clamp_min(1e-12)
    return (cosine * weight).mean()


@torch.no_grad()
def _srgnv_residual_target(data, parts, config):
    ego = parts["ego"].detach()
    graph = parts["graph"].detach()
    ego_unit = F.normalize(ego, dim=1)
    parallel = (graph * ego_unit).sum(dim=1, keepdim=True) * ego_unit
    residual = graph - parallel
    if bool(config.get("srgnv_shuffle_residual", False)):
        residual = residual[torch.randperm(residual.size(0), device=residual.device)]
    raw = data.x.detach().float()
    raw_low = row_normalized_propagate(raw, data.edge_index, add_self=True).float()
    residual_score = 1.0 - (
        F.normalize(raw, dim=1)
        * F.normalize(raw_low, dim=1)
    ).sum(dim=1)
    temperature = max(float(config["srgnv_residual_temperature"]), 1e-12)
    gate = torch.sigmoid(
        (residual_score - float(config["srgnv_residual_threshold"])) / temperature
    )
    min_weight = float(config["srgnv_min_residual_weight"])
    gate = gate * (1.0 - min_weight) + min_weight
    return residual, residual_score, gate


def _pcnv_assignment_terms(anchor, target, prototypes, config, view_weight=None):
    assignment_temperature = max(float(config["pcnv_assignment_temperature"]), 1e-12)
    target_temperature = max(float(config["pcnv_target_temperature"]), 1e-12)
    proto = F.normalize(prototypes, dim=1)
    anchor_logits = F.normalize(anchor, dim=1) @ proto.t() / assignment_temperature
    with torch.no_grad():
        target_logits = F.normalize(target.detach(), dim=1) @ proto.t() / target_temperature
        target_prob = torch.softmax(target_logits, dim=1)
        if bool(config.get("pcnv_shuffle_assignments", False)):
            target_prob = target_prob[torch.randperm(target_prob.size(0), device=target_prob.device)]
        target_confidence = target_prob.max(dim=1).values
        min_confidence = float(config.get("pcnv_min_target_confidence", 0.0))
        confidence_power = float(config.get("pcnv_confidence_power", 0.0))
        if confidence_power > 0.0:
            denom = max(1.0 - min_confidence, 1e-12)
            sample_weight = ((target_confidence - min_confidence) / denom).clamp(0.0, 1.0)
            sample_weight = sample_weight.pow(confidence_power)
        else:
            sample_weight = torch.ones_like(target_confidence)
        if view_weight is None:
            view_weight = torch.ones_like(sample_weight)
        else:
            view_weight = view_weight.to(device=sample_weight.device, dtype=sample_weight.dtype)
        sample_weight = sample_weight * view_weight
    per_node_consistency = -(target_prob * F.log_softmax(anchor_logits, dim=1)).sum(dim=1)
    consistency = (per_node_consistency * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)
    anchor_prob = torch.softmax(anchor_logits, dim=1)
    mean_prob = anchor_prob.mean(dim=0)
    balance = (mean_prob * (mean_prob.clamp_min(1e-12).log() + math.log(mean_prob.numel()))).sum()
    entropy = -(anchor_prob * anchor_prob.clamp_min(1e-12).log()).sum(dim=1)
    usage_entropy = -(mean_prob * mean_prob.clamp_min(1e-12).log()).sum()
    max_usage_entropy = math.log(mean_prob.numel())
    min_usage_entropy = float(config.get("pcnv_min_usage_entropy_frac", 0.0)) * max_usage_entropy
    guard_temperature = max(float(config.get("pcnv_entropy_guard_temperature", 0.1)), 1e-12)
    if bool(config.get("pcnv_entropy_guard", False)):
        entropy_guard = torch.sigmoid((usage_entropy.detach() - min_usage_entropy) / guard_temperature)
    else:
        entropy_guard = usage_entropy.new_tensor(1.0)
    return consistency, balance, {
        "assignment_entropy": entropy.mean(),
        "assignment_max_prob": anchor_prob.max(dim=1).values.mean(),
        "target_confidence": target_confidence.mean(),
        "view_weight": view_weight.mean(),
        "target_weight": sample_weight.mean(),
        "entropy_guard": entropy_guard,
        "prototype_usage_entropy": usage_entropy,
    }


def _pcnv_loss(ego, graph, prototypes, config):
    with torch.no_grad():
        proto = F.normalize(prototypes, dim=1)
        target_temperature = max(float(config["pcnv_target_temperature"]), 1e-12)
        ego_prob = torch.softmax(F.normalize(ego.detach(), dim=1) @ proto.t() / target_temperature, dim=1)
        graph_prob = torch.softmax(F.normalize(graph.detach(), dim=1) @ proto.t() / target_temperature, dim=1)
        view_agreement = (ego_prob * graph_prob).sum(dim=1)
        view_power = float(config.get("pcnv_view_agreement_power", 0.0))
        if view_power > 0.0:
            min_agreement = float(config.get("pcnv_min_view_agreement", 0.0))
            denom = max(1.0 - min_agreement, 1e-12)
            view_weight = ((view_agreement - min_agreement) / denom).clamp(0.0, 1.0)
            view_weight = view_weight.pow(view_power)
        else:
            view_weight = torch.ones_like(view_agreement)
    ego_to_graph, ego_balance, ego_stats = _pcnv_assignment_terms(
        ego,
        graph,
        prototypes,
        config,
        view_weight,
    )
    graph_to_ego, graph_balance, graph_stats = _pcnv_assignment_terms(
        graph,
        ego,
        prototypes,
        config,
        view_weight,
    )
    consistency = 0.5 * (ego_to_graph + graph_to_ego)
    entropy_guard = 0.5 * (ego_stats["entropy_guard"] + graph_stats["entropy_guard"])
    guarded_consistency = consistency * entropy_guard
    balance = 0.5 * (ego_balance + graph_balance)
    stats = {
        "pcnv_consistency_loss": consistency.detach(),
        "pcnv_guarded_consistency_loss": guarded_consistency.detach(),
        "pcnv_balance_loss": balance.detach(),
        "pcnv_assignment_entropy_mean": 0.5 * (
            ego_stats["assignment_entropy"].detach()
            + graph_stats["assignment_entropy"].detach()
        ),
        "pcnv_assignment_max_prob_mean": 0.5 * (
            ego_stats["assignment_max_prob"].detach()
            + graph_stats["assignment_max_prob"].detach()
        ),
        "pcnv_usage_entropy_mean": 0.5 * (
            ego_stats["prototype_usage_entropy"].detach()
            + graph_stats["prototype_usage_entropy"].detach()
        ),
        "pcnv_target_confidence_mean": 0.5 * (
            ego_stats["target_confidence"].detach()
            + graph_stats["target_confidence"].detach()
        ),
        "pcnv_view_agreement_mean": view_agreement.detach().mean(),
        "pcnv_view_weight_mean": 0.5 * (
            ego_stats["view_weight"].detach()
            + graph_stats["view_weight"].detach()
        ),
        "pcnv_target_weight_mean": 0.5 * (
            ego_stats["target_weight"].detach()
            + graph_stats["target_weight"].detach()
        ),
        "pcnv_entropy_guard_mean": entropy_guard.detach(),
    }
    return guarded_consistency, balance, stats


def _pcnv_control_name(config):
    guarded = (
        bool(config.get("pcnv_entropy_guard", False))
        or float(config.get("pcnv_confidence_power", 0.0)) > 0.0
        or float(config.get("pcnv_min_target_confidence", 0.0)) > 0.0
        or float(config.get("pcnv_view_agreement_power", 0.0)) > 0.0
        or float(config.get("pcnv_min_view_agreement", 0.0)) > 0.0
    )
    name = "pcnv_guarded" if guarded else "pcnv"
    if bool(config.get("pcnv_shuffle_assignments", False)):
        name += "_shuffled"
    return name


@torch.no_grad()
def _lcos_conflict_gate(data, config):
    raw = data.x.detach().float()
    raw_low = row_normalized_propagate(raw, data.edge_index, add_self=True).float()
    raw_agreement = (
        F.normalize(raw, dim=1)
        * F.normalize(raw_low, dim=1)
    ).sum(dim=1)
    raw_residual = (raw - raw_low).norm(dim=1) / (
        raw.norm(dim=1) + raw_low.norm(dim=1)
    ).clamp_min(1e-12)

    degree = torch.zeros(data.num_nodes, device=data.edge_index.device, dtype=torch.float32)
    ones = torch.ones(data.edge_index.size(1), device=data.edge_index.device)
    degree.scatter_add_(0, data.edge_index[0], ones)
    degree.scatter_add_(0, data.edge_index[1], ones)
    log_degree = torch.log1p(degree)

    score = (
        _standardize(raw_residual)
        - _standardize(raw_agreement)
        + float(config["lcos_degree_weight"]) * _standardize(log_degree)
    )
    temperature = max(float(config["lcos_route_temperature"]), 1e-12)
    gate = torch.sigmoid((score - float(config["lcos_route_threshold"])) / temperature)
    min_weight = float(config["lcos_min_branch_weight"])
    gate = gate * (1.0 - 2.0 * min_weight) + min_weight
    gate = gate.clamp(min_weight, 1.0 - min_weight)
    if bool(config.get("lcos_shuffle_gate", False)):
        gate = gate[torch.randperm(gate.size(0), device=gate.device)]
    return gate, score, raw_agreement, raw_residual


def _lcos_structural_mix(parts, high_gate):
    low_weight = 1.0 - high_gate
    graph = F.normalize(parts["graph"], dim=1)
    high = F.normalize(parts["high"], dim=1)
    return (
        low_weight.view(-1, 1) * graph
        + high_gate.view(-1, 1) * high
    )


def _lcos_final(model, parts, structural_mix):
    return model.final_norm(torch.cat([
        F.normalize(parts["ego"], dim=1),
        F.normalize(structural_mix, dim=1),
    ], dim=1))


@torch.no_grad()
def _dsp_separability_weight(parts, config):
    z = torch.cat([
        F.normalize(parts["ego"].detach(), dim=1),
        F.normalize(parts["graph"].detach(), dim=1),
    ], dim=1)
    z = F.normalize(z, dim=1)
    num_nodes = z.size(0)
    topk = min(max(1, int(config["dsp_margin_topk"])), max(1, (num_nodes - 1) // 2))
    sim = z @ z.t()
    sim.fill_diagonal_(-2.0)
    values = torch.topk(sim, k=min(2 * topk, max(1, num_nodes - 1)), dim=1).values
    near = values[:, :topk].mean(dim=1)
    border = values[:, topk:].mean(dim=1) if values.size(1) > topk else values[:, :topk].mean(dim=1)
    margin = near - border
    view_consistency = (
        F.normalize(parts["ego"].detach(), dim=1)
        * F.normalize(parts["graph"].detach(), dim=1)
    ).sum(dim=1)
    score = _standardize(margin) + float(config["dsp_view_weight"]) * _standardize(view_consistency)
    temperature = max(float(config["dsp_margin_temperature"]), 1e-12)
    weight = torch.sigmoid(score / temperature)
    min_weight = float(config["dsp_min_weight"])
    weight = weight * (1.0 - min_weight) + min_weight
    if bool(config.get("dsp_shuffle_weight", False)):
        weight = weight[torch.randperm(weight.size(0), device=weight.device)]
    return weight, margin, view_consistency, score


def _rrnv_loss(z_ego, z_graph, config, invariance_scale=1.0, shuffle_pairs=None):
    if shuffle_pairs is None:
        shuffle_pairs = bool(config.get("rrnv_shuffle_pairs", False))
    if shuffle_pairs:
        z_graph = z_graph[torch.randperm(z_graph.size(0), device=z_graph.device)]
    if not torch.is_tensor(invariance_scale):
        invariance_scale = torch.tensor(
            float(invariance_scale),
            device=z_ego.device,
            dtype=z_ego.dtype,
        )
    invariance = F.mse_loss(F.normalize(z_ego, dim=1), F.normalize(z_graph, dim=1))
    variance = 0.5 * (variance_loss(z_ego) + variance_loss(z_graph))
    covariance = 0.5 * (covariance_loss(z_ego) + covariance_loss(z_graph))
    loss = (
        invariance_scale * float(config["rrnv_invariance_weight"]) * invariance
        + float(config["rrnv_variance_weight"]) * variance
        + float(config["rrnv_covariance_weight"]) * covariance
    )
    return loss, {
        "rrnv_invariance_loss": invariance.detach(),
        "rrnv_variance_loss": variance.detach(),
        "rrnv_covariance_loss": covariance.detach(),
    }


def _rrnv_component_losses(z_ego, z_graph, shuffle_pairs=False):
    if shuffle_pairs:
        z_graph = z_graph[torch.randperm(z_graph.size(0), device=z_graph.device)]
    norm_ego = F.normalize(z_ego, dim=1)
    norm_graph = F.normalize(z_graph, dim=1)
    invariance = F.mse_loss(norm_ego, norm_graph)
    variance = 0.5 * (variance_loss(z_ego) + variance_loss(z_graph))
    covariance = 0.5 * (covariance_loss(z_ego) + covariance_loss(z_graph))
    cosine = (norm_ego * norm_graph).sum(dim=1)
    return {
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
        "cosine": cosine,
    }


def _rrnv_weighted_invariance_loss(z_ego, z_graph, reliability, config, shuffle_pairs=None):
    if shuffle_pairs is None:
        shuffle_pairs = bool(config.get("rrnv_shuffle_pairs", False))
    if shuffle_pairs:
        z_graph = z_graph[torch.randperm(z_graph.size(0), device=z_graph.device)]
    per_node = (
        F.normalize(z_ego, dim=1)
        - F.normalize(z_graph, dim=1)
    ).pow(2).mean(dim=1)
    reliability = reliability.detach().to(per_node.device, dtype=per_node.dtype)
    if bool(config.get("rwirrnv_shuffle_weight", False)):
        reliability = reliability[torch.randperm(reliability.size(0), device=reliability.device)]
    min_reliability = float(config["rwirrnv_min_reliability"])
    reliability = reliability.clamp(0.0, 1.0)
    reliability = min_reliability + (1.0 - min_reliability) * reliability
    reliability = reliability.pow(float(config["rwirrnv_weight_power"]))
    if bool(config.get("rwirrnv_constant_weight", False)):
        reliability = torch.full_like(reliability, reliability.mean())
    invariance = (per_node * reliability).mean()
    unweighted_invariance = per_node.mean()
    variance = 0.5 * (variance_loss(z_ego) + variance_loss(z_graph))
    covariance = 0.5 * (covariance_loss(z_ego) + covariance_loss(z_graph))
    loss = (
        float(config["rrnv_invariance_weight"]) * invariance
        + float(config["rrnv_variance_weight"]) * variance
        + float(config["rrnv_covariance_weight"]) * covariance
    )
    return loss, {
        "rrnv_invariance_loss": invariance.detach(),
        "rrnv_unweighted_invariance_loss": unweighted_invariance.detach(),
        "rrnv_variance_loss": variance.detach(),
        "rrnv_covariance_loss": covariance.detach(),
        "rwirrnv_reliability": reliability.detach(),
    }


def _darrnv_density_gate(data, config):
    avg_degree = data.edge_index.size(1) / max(1, data.num_nodes)
    threshold = max(float(config["darrnv_degree_threshold"]), 1e-12)
    temperature = max(float(config["darrnv_degree_temperature"]), 1e-12)
    gate = torch.sigmoid(
        torch.tensor(
            (math.log1p(threshold) - math.log1p(avg_degree)) / temperature,
            device=data.x.device,
            dtype=data.x.dtype,
        )
    )
    return gate, avg_degree


def _dsrrnv_density_high_gate(data, config):
    avg_degree = data.edge_index.size(1) / max(1, data.num_nodes)
    threshold = max(float(config["dsrrnv_degree_threshold"]), 1e-12)
    temperature = max(float(config["dsrrnv_degree_temperature"]), 1e-12)
    raw_gate = torch.sigmoid(
        torch.tensor(
            (math.log1p(avg_degree) - math.log1p(threshold)) / temperature,
            device=data.x.device,
            dtype=data.x.dtype,
        )
    )
    min_gate = float(config["dsrrnv_min_high_gate"])
    max_gate = float(config["dsrrnv_max_high_gate"])
    gate = min_gate + (max_gate - min_gate) * raw_gate
    return gate.clamp(0.0, 1.0), avg_degree


def _dirrnv_invariance_scale(high_gate, config):
    scale = (1.0 - high_gate).pow(float(config["dirrnv_invariance_power"]))
    min_scale = float(config["dirrnv_min_invariance_scale"])
    return (min_scale + (1.0 - min_scale) * scale).clamp(0.0, 1.0)


def _eairrnv_invariance_scale(parts, config):
    high_norm = parts["high"].detach().norm(dim=1)
    graph_norm = parts["graph"].detach().norm(dim=1).clamp_min(1e-12)
    energy_ratio = high_norm / graph_norm
    energy_mean = energy_ratio.mean()
    threshold = max(float(config["eairrnv_energy_threshold"]), 1e-12)
    conflict = energy_mean / (energy_mean + energy_mean.new_tensor(threshold))
    attenuation = float(config["eairrnv_strength"]) * conflict.pow(
        float(config["eairrnv_power"])
    )
    min_scale = float(config["eairrnv_min_invariance_scale"])
    scale = (1.0 - attenuation).clamp(min_scale, 1.0)
    return scale, {
        "energy_ratio_mean": energy_mean.detach(),
        "energy_ratio_std": energy_ratio.std(unbiased=False).detach(),
        "conflict": conflict.detach(),
    }


def _bprrnv_aux_gate(data, parts, config):
    avg_degree = data.edge_index.size(1) / max(1, data.num_nodes)
    degree_threshold = max(float(config["bprrnv_degree_threshold"]), 1e-12)
    degree_temperature = max(float(config["bprrnv_degree_temperature"]), 1e-12)
    density_factor = torch.sigmoid(
        torch.tensor(
            (math.log1p(degree_threshold) - math.log1p(avg_degree))
            / degree_temperature,
            device=data.x.device,
            dtype=data.x.dtype,
        )
    )
    if bool(config.get("bprrnv_no_density_gate", False)):
        density_factor = torch.ones_like(density_factor)

    high_norm = parts["high"].detach().norm(dim=1)
    graph_norm = parts["graph"].detach().norm(dim=1).clamp_min(1e-12)
    energy_ratio = high_norm / graph_norm
    energy_mean = energy_ratio.mean()
    energy_threshold = max(float(config["bprrnv_energy_threshold"]), 1e-12)
    energy_conflict = energy_mean / (
        energy_mean + energy_mean.new_tensor(energy_threshold)
    )
    energy_factor = 1.0 - float(config["bprrnv_energy_strength"]) * energy_conflict.pow(
        float(config["bprrnv_energy_power"])
    )
    energy_factor = energy_factor.clamp(float(config["bprrnv_min_energy_factor"]), 1.0)
    if bool(config.get("bprrnv_no_energy_gate", False)):
        energy_factor = torch.ones_like(energy_factor)

    aux_gate = (density_factor * energy_factor).clamp(0.0, 1.0)
    if bool(config.get("bprrnv_uniform_gate", False)):
        aux_gate = torch.ones_like(aux_gate)
    return {
        "aux_gate": aux_gate,
        "density_factor": density_factor,
        "energy_factor": energy_factor,
        "energy_conflict": energy_conflict.detach(),
        "energy_ratio_mean": energy_mean.detach(),
        "energy_ratio_std": energy_ratio.std(unbiased=False).detach(),
        "avg_degree": avg_degree,
    }


def _bprrnv_regularizer_loss(z_ego, z_graph, config, aux_gate):
    components = _rrnv_component_losses(
        z_ego,
        z_graph,
        shuffle_pairs=bool(config.get("rrnv_shuffle_pairs", False)),
    )
    core = (
        float(config["bprrnv_invariance_weight"]) * components["invariance"]
        + float(config["bprrnv_variance_weight"]) * components["variance"]
        + float(config["bprrnv_covariance_weight"]) * components["covariance"]
    )
    loss = float(config["bprrnv_rr_weight"]) * aux_gate * core
    return loss, {
        "rrnv_invariance_loss": components["invariance"].detach(),
        "rrnv_variance_loss": components["variance"].detach(),
        "rrnv_covariance_loss": components["covariance"].detach(),
        "rrnv_pair_cosine": components["cosine"].detach(),
        "bprrnv_core_loss": core.detach(),
        "bprrnv_regularizer_loss": loss.detach(),
    }


def _tns_negative_weight(cache_keys, neg_idx, config):
    keys = F.normalize(cache_keys.detach().float(), dim=1)
    rows = []
    chunk_size = min(max(1, int(config.get("cache_chunk_size", 256))), 256)
    for start in range(0, keys.size(0), chunk_size):
        end = min(start + chunk_size, keys.size(0))
        chunk_idx = neg_idx[start:end]
        neg_keys = keys[chunk_idx.reshape(-1)].view(
            chunk_idx.size(0),
            chunk_idx.size(1),
            -1,
        )
        key_sim_chunk = (
            keys[start:end].view(end - start, 1, -1) * neg_keys
        ).sum(dim=2)
        rows.append(key_sim_chunk)
    key_sim = torch.cat(rows, dim=0)
    temperature = max(float(config["tns_key_temperature"]), 1e-12)
    threshold = float(config["tns_key_threshold"])
    min_weight = float(config["tns_min_weight"])
    weight = torch.sigmoid((threshold - key_sim) / temperature)
    weight = min_weight + (1.0 - min_weight) * weight
    if bool(config.get("tns_shuffle_weight", False)):
        weight = weight.reshape(-1)
        weight = weight[torch.randperm(weight.numel(), device=weight.device)]
        weight = weight.view_as(key_sim)
    if bool(config.get("tns_uniform_weight", False)):
        weight = torch.ones_like(weight)
    return weight.detach(), key_sim.detach()


def _tns_loss(z, cache_keys, config):
    neg_idx = _sample_negative_indices(
        z.size(0),
        int(config["tns_num_negatives"]),
        z.device,
    )
    weight, key_sim = _tns_negative_weight(cache_keys, neg_idx, config)
    z = F.normalize(z, dim=1)
    neg = z[neg_idx.reshape(-1)].view(neg_idx.size(0), neg_idx.size(1), -1)
    pair_cosine = (z.view(z.size(0), 1, -1) * neg).sum(dim=2)
    temperature = max(float(config["tns_temperature"]), 1e-12)
    margin = float(config["tns_margin"])
    per_pair = F.softplus((pair_cosine - margin) / temperature)
    loss = (per_pair * weight).sum() / weight.sum().clamp_min(1e-12)
    return loss, {
        "tns_loss": loss.detach(),
        "tns_weight_mean": weight.mean().detach(),
        "tns_weight_std": weight.std(unbiased=False).detach(),
        "tns_key_sim_mean": key_sim.mean().detach(),
        "tns_key_sim_std": key_sim.std(unbiased=False).detach(),
        "tns_pair_cosine_mean": pair_cosine.detach().mean(),
        "tns_pair_cosine_std": pair_cosine.detach().std(unbiased=False),
        "tns_repulsion_active_fraction": (pair_cosine.detach() > margin).float().mean(),
    }


def _dprrnv_shuffle_prob(high_gate, config):
    base = high_gate.pow(float(config["dprrnv_shuffle_power"]))
    min_prob = float(config["dprrnv_min_shuffle_prob"])
    max_prob = float(config["dprrnv_max_shuffle_prob"])
    return (min_prob + (max_prob - min_prob) * base).clamp(0.0, 1.0)


def _dprrnv_target(z_graph, shuffle_prob, force_shuffle=False):
    shuffled = z_graph[torch.randperm(z_graph.size(0), device=z_graph.device)]
    if force_shuffle:
        return shuffled
    if torch.is_tensor(shuffle_prob) and shuffle_prob.dim() > 0:
        shuffle_prob = shuffle_prob.view(-1, 1)
    target = (1.0 - shuffle_prob) * z_graph + shuffle_prob * shuffled
    return F.normalize(target, dim=1)


@torch.no_grad()
def _nprrnv_pair_gate(data, parts, config):
    raw = data.x.detach().float()
    raw_low = row_normalized_propagate(raw, data.edge_index, add_self=True).float()
    raw_agreement = (
        F.normalize(raw, dim=1)
        * F.normalize(raw_low, dim=1)
    ).sum(dim=1)
    raw_residual = (raw - raw_low).norm(dim=1) / (
        raw.norm(dim=1) + raw_low.norm(dim=1)
    ).clamp_min(1e-12)

    degree = torch.zeros(data.num_nodes, device=data.edge_index.device, dtype=torch.float32)
    ones = torch.ones(data.edge_index.size(1), device=data.edge_index.device)
    degree.scatter_add_(0, data.edge_index[0], ones)
    degree.scatter_add_(0, data.edge_index[1], ones)
    log_degree = torch.log1p(degree)

    view_cosine = (
        F.normalize(parts["ego"].detach(), dim=1)
        * F.normalize(parts["graph"].detach(), dim=1)
    ).sum(dim=1)
    score = (
        float(config["nprrnv_degree_weight"]) * _standardize(log_degree)
        + float(config["nprrnv_residual_weight"]) * _standardize(raw_residual)
        - float(config["nprrnv_agreement_weight"]) * _standardize(raw_agreement)
        - float(config["nprrnv_view_weight"]) * _standardize(view_cosine)
    )
    temperature = max(float(config["nprrnv_route_temperature"]), 1e-12)
    node_gate = torch.sigmoid(
        (score - float(config["nprrnv_route_threshold"])) / temperature
    )
    min_local = float(config["nprrnv_min_local_scale"])
    local_scale = min_local + (1.0 - min_local) * node_gate
    high_gate, avg_degree = _dsrrnv_density_high_gate(data, config)
    base_prob = (high_gate * local_scale).clamp(0.0, 1.0)
    min_prob = float(config["nprrnv_min_shuffle_prob"])
    max_prob = float(config["nprrnv_max_shuffle_prob"])
    shuffle_prob = min_prob + (max_prob - min_prob) * base_prob
    shuffle_prob = shuffle_prob.clamp(0.0, 1.0)
    if bool(config.get("nprrnv_shuffle_gate", False)):
        shuffle_prob = shuffle_prob[torch.randperm(shuffle_prob.size(0), device=shuffle_prob.device)]
    return {
        "shuffle_prob": shuffle_prob,
        "node_gate": node_gate,
        "score": score,
        "raw_agreement": raw_agreement,
        "raw_residual": raw_residual,
        "view_cosine": view_cosine,
        "log_degree": log_degree,
        "graph_high_gate": high_gate,
        "avg_degree": avg_degree,
    }


@torch.no_grad()
def _rwirrnv_reliability(data, parts, config):
    gate_stats = _nprrnv_pair_gate(data, parts, config)
    reliability = 1.0 - gate_stats["shuffle_prob"]
    return reliability.clamp(0.0, 1.0), gate_stats


def _density_mixed_final(model, parts, high_gate):
    structural_mix = (
        (1.0 - high_gate) * F.normalize(parts["graph"], dim=1)
        + high_gate * F.normalize(parts["high"], dim=1)
    )
    return model.final_norm(torch.cat([
        F.normalize(parts["ego"], dim=1),
        F.normalize(structural_mix, dim=1),
    ], dim=1))


@torch.no_grad()
def _degree_disagreement_gate(data, config):
    num_nodes = data.num_nodes
    degree = torch.zeros(num_nodes, device=data.edge_index.device, dtype=torch.float32)
    ones = torch.ones(data.edge_index.size(1), device=data.edge_index.device)
    degree.scatter_add_(0, data.edge_index[0], ones)
    degree.scatter_add_(0, data.edge_index[1], ones)
    log_degree = torch.log1p(degree)
    threshold = float(config["danv_degree_threshold"])
    temperature = max(float(config["danv_degree_temperature"]), 1e-12)
    return torch.sigmoid((log_degree - threshold) / temperature)


@torch.no_grad()
def _fdnv_filter_gate(data, config):
    raw_low_raw = row_normalized_propagate(data.x.detach(), data.edge_index, add_self=True)
    raw = F.normalize(data.x.detach().float(), dim=1)
    raw_low = F.normalize(raw_low_raw.float(), dim=1)
    raw_agreement = (raw * raw_low).sum(dim=1)
    raw_residual = (data.x.detach().float() - raw_low_raw.float()).norm(dim=1)
    raw_scale = data.x.detach().float().norm(dim=1).clamp_min(1e-12)
    raw_residual_energy = raw_residual / raw_scale

    degree = torch.zeros(data.num_nodes, device=data.edge_index.device, dtype=torch.float32)
    ones = torch.ones(data.edge_index.size(1), device=data.edge_index.device)
    degree.scatter_add_(0, data.edge_index[0], ones)
    degree.scatter_add_(0, data.edge_index[1], ones)
    log_degree = torch.log1p(degree)

    score = (
        _standardize(raw_residual_energy)
        - _standardize(raw_agreement)
        + 0.5 * _standardize(log_degree)
    )
    temperature = max(float(config["fdnv_filter_temperature"]), 1e-12)
    high_gate = torch.sigmoid(score / temperature)
    min_weight = float(config["fdnv_min_filter_weight"])
    high_gate = high_gate * (1.0 - min_weight) + min_weight
    low_gate = (1.0 - high_gate) * (1.0 - min_weight) + min_weight
    return high_gate, low_gate


@torch.no_grad()
def _semantic_positive_indices(data, config):
    keys = propagation_signature(data.x.detach(), data.edge_index, hops=1)
    return topk_cache_indices(
        keys,
        topk=int(config["sspnv_semantic_topk"]),
        chunk_size=int(config["cache_chunk_size"]),
        exclude_self=True,
    )


@torch.no_grad()
def _random_positive_indices(num_nodes, topk, device):
    topk = max(1, int(topk))
    if num_nodes <= 1:
        return torch.zeros((num_nodes, topk), device=device, dtype=torch.long)
    row = torch.arange(num_nodes, device=device).view(-1, 1)
    positive = torch.randint(
        low=0,
        high=num_nodes - 1,
        size=(num_nodes, topk),
        device=device,
    )
    return positive + (positive >= row).long()


@torch.no_grad()
def _random_single_positive_indices(num_nodes, device):
    if num_nodes <= 1:
        return torch.zeros(num_nodes, device=device, dtype=torch.long)
    row = torch.arange(num_nodes, device=device)
    positive = torch.randint(
        low=0,
        high=num_nodes - 1,
        size=(num_nodes,),
        device=device,
    )
    return positive + (positive >= row).long()


@torch.no_grad()
def _spatial_positive_indices(data):
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    source = torch.cat([edge_index[0], edge_index[1]], dim=0)
    target = torch.cat([edge_index[1], edge_index[0]], dim=0)
    positive = torch.arange(num_nodes, device=edge_index.device)
    if source.numel() == 0:
        return positive
    source_sorted, order = torch.sort(source)
    target_sorted = target[order]
    unique, counts = torch.unique_consecutive(source_sorted, return_counts=True)
    starts = torch.cat([
        torch.zeros(1, device=edge_index.device, dtype=torch.long),
        counts.cumsum(dim=0)[:-1],
    ])
    positive[unique] = target_sorted[starts]
    return positive


@torch.no_grad()
def _multi_positive_masks(data, semantic_idx, config, shuffle_key="mpnv_shuffle_positives"):
    num_nodes = data.num_nodes
    device = data.x.device
    semantic_mask = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.bool)
    row = torch.arange(num_nodes, device=device).view(-1, 1).expand_as(semantic_idx)
    semantic_mask[row.reshape(-1), semantic_idx.reshape(-1)] = True

    spatial_mask = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.bool)
    source, target = data.edge_index
    spatial_mask[source, target] = True
    spatial_mask[target, source] = True
    if bool(config.get("mpnv_include_self", True)):
        diag = torch.arange(num_nodes, device=device)
        semantic_mask[diag, diag] = True
        spatial_mask[diag, diag] = True
    if bool(config.get(shuffle_key, False)):
        perm = torch.randperm(num_nodes, device=device)
        semantic_mask = semantic_mask[:, perm]
        spatial_mask = spatial_mask[:, perm]
    return semantic_mask, spatial_mask


@torch.no_grad()
def _positive_confidence(cache_keys, semantic_idx, spatial_idx):
    keys = F.normalize(cache_keys, dim=1)
    semantic_positive = keys[semantic_idx.reshape(-1)].view(
        semantic_idx.size(0),
        semantic_idx.size(1),
        -1,
    )
    semantic_sim = (keys.view(keys.size(0), 1, -1) * semantic_positive).sum(dim=2).mean(dim=1)
    spatial_sim = (keys * keys[spatial_idx]).sum(dim=1)
    return semantic_sim, spatial_sim


@torch.no_grad()
def _dense_positive_confidence(cache_keys, semantic_mask, spatial_mask):
    keys = F.normalize(cache_keys, dim=1)
    sim = keys @ keys.t()
    num_nodes = keys.size(0)
    diag = torch.arange(num_nodes, device=keys.device)

    def masked_mean(mask):
        confidence_mask = mask.clone()
        confidence_mask[diag, diag] = False
        count = confidence_mask.sum(dim=1)
        if (count == 0).any():
            confidence_mask[diag[count == 0], diag[count == 0]] = True
            count = confidence_mask.sum(dim=1)
        values = (sim * confidence_mask.to(sim.dtype)).sum(dim=1)
        return values / count.clamp_min(1).to(sim.dtype)

    return masked_mean(semantic_mask), masked_mean(spatial_mask)


@torch.no_grad()
def _aompnv_router_weights(
    loss_semantic,
    loss_spatial,
    loss_bootstrap,
    semantic_conf,
    spatial_conf,
    config,
):
    objective_scores = torch.stack([
        -_standardize(loss_semantic.detach()),
        -_standardize(loss_spatial.detach()),
        -_standardize(loss_bootstrap.detach()),
    ], dim=1)
    confidence_scores = torch.stack([
        _standardize(semantic_conf),
        _standardize(spatial_conf),
        torch.zeros_like(semantic_conf),
    ], dim=1)
    logits = (
        objective_scores
        + float(config["aompnv_confidence_weight"]) * confidence_scores
    )
    temperature = max(float(config["aompnv_router_temperature"]), 1e-12)
    probs = torch.softmax(logits / temperature, dim=1)
    min_prob = min(max(float(config["aompnv_min_branch_prob"]), 0.0), 1.0 / 3.0)
    if min_prob > 0.0:
        probs = probs * (1.0 - 3.0 * min_prob) + min_prob
    return probs


@torch.no_grad()
def _afpnv_branch_weights(cache_keys, semantic_idx, spatial_idx, config):
    semantic_sim, spatial_sim = _positive_confidence(cache_keys, semantic_idx, spatial_idx)
    temperature = max(float(config["afpnv_conf_temperature"]), 1e-12)
    min_weight = float(config["afpnv_min_branch_weight"])
    semantic_weight = torch.sigmoid(
        (semantic_sim - float(config["afpnv_semantic_conf_threshold"])) / temperature
    )
    spatial_weight = torch.sigmoid(
        (spatial_sim - float(config["afpnv_spatial_conf_threshold"])) / temperature
    )
    semantic_weight = semantic_weight * (1.0 - min_weight) + min_weight
    spatial_weight = spatial_weight * (1.0 - min_weight) + min_weight
    return semantic_weight, spatial_weight, semantic_sim, spatial_sim


@torch.no_grad()
def _bspnv_branch_weights(cache_keys, semantic_idx, spatial_idx, config):
    semantic_sim, spatial_sim = _positive_confidence(cache_keys, semantic_idx, spatial_idx)
    bootstrap_logit = torch.full_like(semantic_sim, float(config["bspnv_bootstrap_bias"]))
    logits = torch.stack([semantic_sim, spatial_sim, bootstrap_logit], dim=1)
    temperature = max(float(config["bspnv_branch_temperature"]), 1e-12)
    probs = torch.softmax(logits / temperature, dim=1)
    return probs[:, 0], probs[:, 1], probs[:, 2], semantic_sim, spatial_sim


def _sspnv_control_name(config):
    tags = []
    if bool(config.get("sspnv_random_semantic", False)):
        tags.append("random_semantic")
    if bool(config.get("sspnv_random_spatial", False)):
        tags.append("random_spatial")
    semantic_active = float(config["sspnv_semantic_weight"]) > 0.0
    spatial_active = float(config["sspnv_spatial_weight"]) > 0.0
    if semantic_active and not spatial_active:
        tags.append("semantic_only")
    elif spatial_active and not semantic_active:
        tags.append("spatial_only")
    elif not semantic_active and not spatial_active:
        tags.append("bootstrap_only")
    return "sspnv" if not tags else "sspnv_" + "_".join(tags)


@torch.no_grad()
def _static_cache_keys(data, config):
    mode = config.get("cache_key_mode", "raw_signature")
    if mode == "raw_low":
        return row_normalized_propagate(data.x.detach(), data.edge_index, add_self=True)
    if mode == "raw_signature":
        return propagation_signature(data.x.detach(), data.edge_index, hops=2)
    if mode == "learned_low":
        return None
    raise ValueError(f"Unknown cache_key_mode: {mode}")


def _cache_diagnostics(parts, cache_idx, cache_keys, cache_weight=None):
    high_norm = parts["high"].norm(dim=1)
    graph_norm = parts["graph"].norm(dim=1).clamp_min(1e-12)
    energy_ratio = high_norm / graph_norm
    if cache_idx.size(1) > 0:
        keys = F.normalize(
            parts["low"] if cache_keys is None else cache_keys,
            dim=1,
        )
        anchor = keys
        positive = keys[cache_idx[:, 0]]
        cache_sim = (anchor * positive).sum(dim=1)
    else:
        cache_sim = torch.zeros_like(energy_ratio)
    return {
        "energy_ratio_mean": float(energy_ratio.mean().item()),
        "energy_ratio_std": float(energy_ratio.std(unbiased=False).item()),
        "cache_low_sim_mean": float(cache_sim.mean().item()),
        "cache_low_sim_std": float(cache_sim.std(unbiased=False).item()),
        "cache_weight_mean": (
            float(cache_weight.mean().item()) if cache_weight is not None else 1.0
        ),
        "cache_weight_std": (
            float(cache_weight.std(unbiased=False).item()) if cache_weight is not None else 0.0
        ),
    }


def _ragc_embeddings(raw_x, learned, config):
    raw = F.normalize(raw_x.detach().float(), dim=1)
    learned = F.normalize(learned.detach().float(), dim=1)
    control = config.get("ragc_control", "normal")
    if control == "shuffle":
        learned = learned[torch.randperm(learned.size(0), device=learned.device)]
    elif control == "random":
        learned = F.normalize(torch.randn_like(learned), dim=1)
    elif control not in {"normal", None}:
        raise ValueError(f"Unknown RAGC control: {control}")
    raw = float(config["ragc_raw_weight"]) * raw
    learned = float(config["ragc_learned_weight"]) * learned
    return torch.cat([raw, learned], dim=1)


def train_er_cache_gcl(model, data, config, args):
    pcnv_gcl = args.method == "pcnv_gcl"
    extra_parameters = []
    pcnv_prototypes = None
    if pcnv_gcl:
        pcnv_prototypes = torch.empty(
            int(config["pcnv_num_prototypes"]),
            int(config["hidden_dim"]),
            device=data.x.device,
        )
        pcnv_prototypes = torch.nn.Parameter(pcnv_prototypes)
        torch.nn.init.xavier_uniform_(pcnv_prototypes)
        extra_parameters.append(pcnv_prototypes)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + extra_parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    energy_spgcl = args.method == "energy_spgcl"
    danv_gcl = args.method in {"danv_gcl", "danv_degree_gcl"}
    danv_degree_gcl = args.method == "danv_degree_gcl"
    fdnv_gcl = args.method == "fdnv_gcl"
    sspnv_gcl = args.method in {"sspnv_gcl", "afpnv_gcl", "bspnv_gcl"}
    afpnv_gcl = args.method == "afpnv_gcl"
    bspnv_gcl = args.method == "bspnv_gcl"
    mpnv_gcl = args.method == "mpnv_gcl"
    aompnv_gcl = args.method == "aompnv_gcl"
    srgnv_gcl = args.method == "srgnv_gcl"
    lcos_gcl = args.method == "lcos_gcl"
    lcm_gcl = args.method == "lcm_gcl"
    dsp_gcl = args.method == "dsp_gcl"
    rrnv_gcl = args.method == "rrnv_gcl"
    darrnv_gcl = args.method == "darrnv_gcl"
    dsrrnv_gcl = args.method == "dsrrnv_gcl"
    dirrnv_gcl = args.method == "dirrnv_gcl"
    dprrnv_gcl = args.method == "dprrnv_gcl"
    nprrnv_gcl = args.method == "nprrnv_gcl"
    rwirrnv_gcl = args.method == "rwirrnv_gcl"
    eairrnv_gcl = args.method == "eairrnv_gcl"
    bprrnv_gcl = args.method == "bprrnv_gcl"
    tns_gcl = args.method == "tns_gcl"
    ragc_gcl = args.method == "ragc_gcl"
    residual_only = args.method in {
        "er_residual_gcl",
        "gcn_mlp_gcl",
        "danv_gcl",
        "danv_degree_gcl",
        "fdnv_gcl",
        "sspnv_gcl",
        "afpnv_gcl",
        "bspnv_gcl",
        "mpnv_gcl",
        "aompnv_gcl",
        "srgnv_gcl",
        "pcnv_gcl",
        "lcos_gcl",
        "lcm_gcl",
        "dsp_gcl",
        "rrnv_gcl",
        "darrnv_gcl",
        "dsrrnv_gcl",
        "dirrnv_gcl",
        "dprrnv_gcl",
        "nprrnv_gcl",
        "rwirrnv_gcl",
        "eairrnv_gcl",
        "bprrnv_gcl",
        "tns_gcl",
        "ragc_gcl",
    }
    graph_target = args.method in {
        "gcn_mlp_gcl",
        "danv_gcl",
        "danv_degree_gcl",
        "fdnv_gcl",
        "sspnv_gcl",
        "afpnv_gcl",
        "bspnv_gcl",
        "mpnv_gcl",
        "aompnv_gcl",
        "srgnv_gcl",
        "pcnv_gcl",
        "lcos_gcl",
        "lcm_gcl",
        "dsp_gcl",
        "rrnv_gcl",
        "darrnv_gcl",
        "dsrrnv_gcl",
        "dirrnv_gcl",
        "dprrnv_gcl",
        "nprrnv_gcl",
        "rwirrnv_gcl",
        "eairrnv_gcl",
        "bprrnv_gcl",
        "tns_gcl",
        "ragc_gcl",
    }
    topk = 0 if (args.disable_cache or residual_only) else int(config["cache_topk"])
    cache_update = max(1, int(config["cache_update_interval"]))
    cache_idx = None
    cache_keys = _static_cache_keys(data, config)
    semantic_idx = _semantic_positive_indices(data, config) if (sspnv_gcl or mpnv_gcl or aompnv_gcl) else None
    spatial_idx = _spatial_positive_indices(data) if sspnv_gcl else None
    if sspnv_gcl and bool(config.get("sspnv_random_semantic", False)):
        semantic_idx = _random_positive_indices(
            data.num_nodes,
            int(config["sspnv_semantic_topk"]),
            data.x.device,
        )
    if sspnv_gcl and bool(config.get("sspnv_random_spatial", False)):
        spatial_idx = _random_single_positive_indices(data.num_nodes, data.x.device)
    afpnv_branch_stats = None
    if afpnv_gcl:
        afpnv_branch_stats = _afpnv_branch_weights(
            cache_keys,
            semantic_idx,
            spatial_idx,
            config,
        )
    bspnv_branch_stats = None
    if bspnv_gcl:
        bspnv_branch_stats = _bspnv_branch_weights(
            cache_keys,
            semantic_idx,
            spatial_idx,
            config,
        )
    mpnv_masks = None
    if mpnv_gcl:
        mpnv_masks = _multi_positive_masks(data, semantic_idx, config)
    aompnv_masks = None
    aompnv_confidence = None
    if aompnv_gcl:
        aompnv_masks = _multi_positive_masks(
            data,
            semantic_idx,
            config,
            shuffle_key="aompnv_shuffle_positives",
        )
        aompnv_confidence = _dense_positive_confidence(cache_keys, *aompnv_masks)
    history = []
    diagnostics = {}
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        parts = model(data.x, data.edge_index, final_mode=config["final_repr"])
        tns_stats = None
        target = parts["graph"] if graph_target else parts["high"]
        lcos_gate = None
        lcos_mix = None
        if lcos_gcl or lcm_gcl:
            lcos_gate, _, _, _ = _lcos_conflict_gate(data, config)
            lcos_mix = _lcos_structural_mix(parts, lcos_gate)
            parts["final"] = _lcos_final(model, parts, lcos_mix)
        if dsrrnv_gcl or dirrnv_gcl or dprrnv_gcl or nprrnv_gcl or rwirrnv_gcl or eairrnv_gcl:
            high_gate, _ = _dsrrnv_density_high_gate(data, config)
            parts["final"] = _density_mixed_final(model, parts, high_gate)
        with torch.no_grad():
            if cache_idx is None or epoch == 1 or epoch % cache_update == 0:
                if args.disable_cache or residual_only:
                    cache_idx = torch.arange(
                        data.num_nodes,
                        device=data.x.device,
                    ).view(-1, 1)
                else:
                    keys = parts["low"].detach() if cache_keys is None else cache_keys
                    cache_idx = topk_cache_indices(
                        keys,
                        topk=topk,
                        chunk_size=int(config["cache_chunk_size"]),
                        exclude_self=True,
                    )
                if args.shuffle_cache:
                    perm = torch.randperm(cache_idx.size(0), device=cache_idx.device)
                    cache_idx = cache_idx[perm]

        pred_ego = model.pred_ego(parts["ego"])
        pred_high = model.pred_high(target)
        if danv_gcl:
            align_gate = _danv_alignment_gate(data, parts, config)
            disagreement_gate = 1.0 - align_gate
            if danv_degree_gcl:
                disagreement_gate = disagreement_gate * _degree_disagreement_gate(data, config)
            loss_align = 0.5 * (
                weighted_negative_cosine(pred_ego, parts["graph"], align_gate)
                + weighted_negative_cosine(pred_high, parts["ego"], align_gate)
            )
            loss_disagreement = _weighted_cosine_abs(
                parts["ego"],
                parts["graph"],
                disagreement_gate,
            )
            loss_self = (
                float(config["danv_alignment_weight"]) * loss_align
                + float(config["danv_disagreement_weight"]) * loss_disagreement
            )
            loss_cache = parts["final"].new_tensor(0.0)
        elif fdnv_gcl:
            high_gate, low_gate = _fdnv_filter_gate(data, config)
            loss_route = 0.5 * (
                weighted_negative_cosine(pred_ego, parts["high"], high_gate)
                + weighted_negative_cosine(pred_ego, parts["low"], low_gate)
            )
            loss_bootstrap = 0.5 * (
                negative_cosine(pred_ego, parts["graph"])
                + negative_cosine(pred_high, parts["ego"])
            )
            loss_self = (
                float(config["fdnv_route_weight"]) * loss_route
                + float(config["fdnv_bootstrap_weight"]) * loss_bootstrap
            )
            loss_cache = parts["final"].new_tensor(0.0)
        elif sspnv_gcl:
            sem_pos_idx = _sample_positive_indices(semantic_idx)
            neg_idx = _sample_negative_indices(
                data.num_nodes,
                int(config["num_negative_samples"]),
                data.x.device,
            )
            loss_semantic = sampled_info_nce(
                pred_ego,
                parts["high"][sem_pos_idx],
                parts["high"][neg_idx],
                float(config["tau"]),
                (
                    afpnv_branch_stats[0] if afpnv_gcl else
                    bspnv_branch_stats[0] if bspnv_gcl else
                    None
                ),
            )
            loss_spatial = sampled_info_nce(
                pred_ego,
                parts["low"][spatial_idx],
                parts["low"][neg_idx],
                float(config["tau"]),
                (
                    afpnv_branch_stats[1] if afpnv_gcl else
                    bspnv_branch_stats[1] if bspnv_gcl else
                    None
                ),
            )
            loss_bootstrap = 0.5 * (
                negative_cosine(pred_ego, parts["graph"])
                + negative_cosine(pred_high, parts["ego"])
            )
            semantic_scale = (
                float(bspnv_branch_stats[0].mean().item()) if bspnv_gcl else 1.0
            )
            spatial_scale = (
                float(bspnv_branch_stats[1].mean().item()) if bspnv_gcl else 1.0
            )
            loss_self = (
                float(config["sspnv_bootstrap_weight"]) * loss_bootstrap
                + semantic_scale * float(config["sspnv_semantic_weight"]) * loss_semantic
                + spatial_scale * float(config["sspnv_spatial_weight"]) * loss_spatial
            )
            loss_cache = parts["final"].new_tensor(0.0)
        elif srgnv_gcl:
            residual_target, _, residual_gate = _srgnv_residual_target(data, parts, config)
            loss_base = 0.5 * (
                negative_cosine(pred_ego, parts["graph"])
                + negative_cosine(pred_high, parts["ego"])
            )
            residual_pred = model.pred_high(parts["ego"])
            loss_residual = weighted_negative_cosine(
                residual_pred,
                residual_target,
                residual_gate,
            )
            loss_self = (
                float(config["srgnv_base_weight"]) * loss_base
                + float(config["srgnv_residual_weight"]) * loss_residual
            )
            loss_cache = parts["final"].new_tensor(0.0)
        elif pcnv_gcl:
            loss_base = 0.5 * (
                negative_cosine(pred_ego, parts["graph"])
                + negative_cosine(pred_high, parts["ego"])
            )
            loss_proto, loss_balance, _ = _pcnv_loss(
                parts["ego"],
                parts["graph"],
                pcnv_prototypes,
                config,
            )
            loss_self = (
                float(config["pcnv_base_weight"]) * loss_base
                + float(config["pcnv_prototype_weight"]) * loss_proto
                + float(config["pcnv_balance_weight"]) * loss_balance
            )
            loss_cache = parts["final"].new_tensor(0.0)
        elif lcos_gcl:
            high_pred = model.pred_high(parts["high"])
            graph_nodes = 0.5 * (
                negative_cosine_per_node(pred_ego, parts["graph"])
                + negative_cosine_per_node(pred_high, parts["ego"])
            )
            high_nodes = 0.5 * (
                negative_cosine_per_node(pred_ego, parts["high"])
                + negative_cosine_per_node(high_pred, parts["ego"])
            )
            loss_nodes = (
                (1.0 - lcos_gate) * graph_nodes
                + lcos_gate * high_nodes
            )
            loss_self = loss_nodes.mean()
            loss_cache = parts["final"].new_tensor(0.0)
        elif dsp_gcl:
            dsp_weight, _, _, _ = _dsp_separability_weight(parts, config)
            loss_nodes = 0.5 * (
                negative_cosine_per_node(pred_ego, parts["graph"])
                + negative_cosine_per_node(pred_high, parts["ego"])
            )
            norm_weight = dsp_weight / dsp_weight.mean().clamp_min(1e-12)
            loss_self = (loss_nodes * norm_weight).mean()
            loss_cache = parts["final"].new_tensor(0.0)
        elif rrnv_gcl:
            loss_self, rrnv_stats = _rrnv_loss(pred_ego, pred_high, config)
            loss_cache = parts["final"].new_tensor(0.0)
        elif dsrrnv_gcl:
            loss_self, rrnv_stats = _rrnv_loss(pred_ego, pred_high, config)
            loss_cache = parts["final"].new_tensor(0.0)
        elif dirrnv_gcl:
            high_gate, _ = _dsrrnv_density_high_gate(data, config)
            invariance_scale = _dirrnv_invariance_scale(high_gate, config)
            loss_self, rrnv_stats = _rrnv_loss(
                pred_ego,
                pred_high,
                config,
                invariance_scale=invariance_scale,
            )
            loss_cache = parts["final"].new_tensor(0.0)
        elif dprrnv_gcl:
            high_gate, _ = _dsrrnv_density_high_gate(data, config)
            shuffle_prob = _dprrnv_shuffle_prob(high_gate, config)
            perturbed_high = _dprrnv_target(
                pred_high,
                shuffle_prob,
                force_shuffle=bool(config.get("rrnv_shuffle_pairs", False)),
            )
            loss_self, rrnv_stats = _rrnv_loss(
                pred_ego,
                perturbed_high,
                config,
                shuffle_pairs=False,
            )
            loss_cache = parts["final"].new_tensor(0.0)
        elif nprrnv_gcl:
            gate_stats = _nprrnv_pair_gate(data, parts, config)
            perturbed_high = _dprrnv_target(
                pred_high,
                gate_stats["shuffle_prob"],
                force_shuffle=bool(config.get("rrnv_shuffle_pairs", False)),
            )
            loss_self, rrnv_stats = _rrnv_loss(
                pred_ego,
                perturbed_high,
                config,
                shuffle_pairs=False,
            )
            loss_cache = parts["final"].new_tensor(0.0)
        elif rwirrnv_gcl:
            reliability, _ = _rwirrnv_reliability(data, parts, config)
            loss_self, rrnv_stats = _rrnv_weighted_invariance_loss(
                pred_ego,
                pred_high,
                reliability,
                config,
            )
            loss_cache = parts["final"].new_tensor(0.0)
        elif eairrnv_gcl:
            invariance_scale, _ = _eairrnv_invariance_scale(parts, config)
            loss_self, rrnv_stats = _rrnv_loss(
                pred_ego,
                pred_high,
                config,
                invariance_scale=invariance_scale,
            )
            loss_cache = parts["final"].new_tensor(0.0)
        elif bprrnv_gcl:
            loss_base = 0.5 * (
                negative_cosine(pred_ego, parts["graph"])
                + negative_cosine(pred_high, parts["ego"])
            )
            gate_stats = _bprrnv_aux_gate(data, parts, config)
            loss_rr, rrnv_stats = _bprrnv_regularizer_loss(
                pred_ego,
                pred_high,
                config,
                gate_stats["aux_gate"],
            )
            loss_self = loss_base + loss_rr
            loss_cache = parts["final"].new_tensor(0.0)
        elif tns_gcl:
            loss_base = 0.5 * (
                negative_cosine(pred_ego, parts["graph"])
                + negative_cosine(pred_high, parts["ego"])
            )
            loss_tns, tns_stats = _tns_loss(parts["final"], cache_keys, config)
            loss_self = loss_base + float(config["tns_weight"]) * loss_tns
            loss_cache = parts["final"].new_tensor(0.0)
        elif darrnv_gcl:
            loss_base = 0.5 * (
                negative_cosine(pred_ego, parts["graph"])
                + negative_cosine(pred_high, parts["ego"])
            )
            loss_rr, _ = _rrnv_loss(pred_ego, pred_high, config)
            density_gate, _ = _darrnv_density_gate(data, config)
            loss_self = loss_base + density_gate * float(config["darrnv_rr_weight"]) * loss_rr
            loss_cache = parts["final"].new_tensor(0.0)
        elif aompnv_gcl:
            semantic_mask, spatial_mask = aompnv_masks
            semantic_conf, spatial_conf = aompnv_confidence
            loss_semantic_nodes = multi_positive_info_nce_per_node(
                pred_ego,
                parts["high"],
                semantic_mask,
                float(config["tau"]),
            )
            loss_spatial_nodes = multi_positive_info_nce_per_node(
                pred_ego,
                parts["low"],
                spatial_mask,
                float(config["tau"]),
            )
            loss_bootstrap_nodes = 0.5 * (
                negative_cosine_per_node(pred_ego, parts["graph"])
                + negative_cosine_per_node(pred_high, parts["ego"])
            )
            router_probs = _aompnv_router_weights(
                loss_semantic_nodes,
                loss_spatial_nodes,
                loss_bootstrap_nodes,
                semantic_conf,
                spatial_conf,
                config,
            )
            branch_weights = torch.tensor(
                [
                    float(config["aompnv_semantic_weight"]),
                    float(config["aompnv_spatial_weight"]),
                    float(config["aompnv_bootstrap_weight"]),
                ],
                device=parts["final"].device,
                dtype=parts["final"].dtype,
            )
            weighted_probs = router_probs * branch_weights.view(1, 3)
            weighted_probs = weighted_probs / weighted_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
            loss_nodes = (
                weighted_probs[:, 0] * loss_semantic_nodes
                + weighted_probs[:, 1] * loss_spatial_nodes
                + weighted_probs[:, 2] * loss_bootstrap_nodes
            )
            loss_self = loss_nodes.mean()
            loss_cache = parts["final"].new_tensor(0.0)
        elif mpnv_gcl:
            semantic_mask, spatial_mask = mpnv_masks
            loss_semantic = multi_positive_info_nce(
                pred_ego,
                parts["high"],
                semantic_mask,
                float(config["tau"]),
            )
            loss_spatial = multi_positive_info_nce(
                pred_ego,
                parts["low"],
                spatial_mask,
                float(config["tau"]),
            )
            loss_bootstrap = 0.5 * (
                negative_cosine(pred_ego, parts["graph"])
                + negative_cosine(pred_high, parts["ego"])
            )
            loss_self = (
                float(config["mpnv_bootstrap_weight"]) * loss_bootstrap
                + float(config["mpnv_semantic_weight"]) * loss_semantic
                + float(config["mpnv_spatial_weight"]) * loss_spatial
            )
            loss_cache = parts["final"].new_tensor(0.0)
        elif energy_spgcl:
            pos_idx = _sample_positive_indices(cache_idx)
            neg_idx = _sample_negative_indices(
                data.num_nodes,
                int(config["num_negative_samples"]),
                data.x.device,
            )
            high_proj = model.pred_high(parts["high"])
            loss_self = sampled_info_nce(
                high_proj,
                high_proj[pos_idx],
                high_proj[neg_idx],
                float(config["tau"]),
            )
            loss_cache = parts["final"].new_tensor(0.0)
        else:
            loss_self = 0.5 * (
                negative_cosine(pred_ego, target)
                + negative_cosine(pred_high, parts["ego"])
            )
        if residual_only or energy_spgcl:
            loss_cache = parts["final"].new_tensor(0.0)
        else:
            pos_high = _cache_positive_mean(parts["high"].detach(), cache_idx)
            pos_ego = _cache_positive_mean(parts["ego"].detach(), cache_idx)
            confidence_result = _cache_confidence(cache_keys, cache_idx, config)
            cache_weight = None if confidence_result is None else confidence_result[1]
            if cache_weight is None:
                loss_cache = 0.5 * (
                    negative_cosine(pred_ego, pos_high)
                    + negative_cosine(pred_high, pos_ego)
                )
            else:
                loss_cache = 0.5 * (
                    weighted_negative_cosine(pred_ego, pos_high, cache_weight)
                    + weighted_negative_cosine(pred_high, pos_ego, cache_weight)
                )
        var_loss, cov_loss = vicreg_regularizer(parts["final"])
        loss = (
            float(config["self_loss_weight"]) * loss_self
            + (0.0 if residual_only else float(config["cache_loss_weight"])) * loss_cache
            + float(config["variance_loss_weight"]) * var_loss
            + float(config["covariance_loss_weight"]) * cov_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        row = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "self_loss": float(loss_self.item()),
            "cache_loss": float(loss_cache.item()),
            "variance_loss": float(var_loss.item()),
            "covariance_loss": float(cov_loss.item()),
        }
        if tns_stats is not None:
            row["tns_loss"] = float(tns_stats["tns_loss"].item())
            row["tns_weight_mean"] = float(tns_stats["tns_weight_mean"].item())
        history.append(row)
        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"epoch={epoch:03d} loss={row['loss']:.6f} "
                f"self={row['self_loss']:.6f} cache={row['cache_loss']:.6f}"
            )
    model.eval()
    with torch.no_grad():
        parts = model(data.x, data.edge_index, final_mode=config["final_repr"])
        if lcos_gcl or lcm_gcl:
            lcos_gate, _, _, _ = _lcos_conflict_gate(data, config)
            lcos_mix = _lcos_structural_mix(parts, lcos_gate)
            parts["final"] = _lcos_final(model, parts, lcos_mix)
        if dsrrnv_gcl or dirrnv_gcl or dprrnv_gcl or nprrnv_gcl or rwirrnv_gcl:
            high_gate, _ = _dsrrnv_density_high_gate(data, config)
            parts["final"] = _density_mixed_final(model, parts, high_gate)
        final = parts["final"].detach()
        confidence_result = _cache_confidence(cache_keys, cache_idx, config)
        cache_weight = None if confidence_result is None else confidence_result[1]
        diagnostics = _cache_diagnostics(parts, cache_idx, cache_keys, cache_weight)
        if danv_gcl:
            gate = _danv_alignment_gate(data, parts, config)
            diagnostics["danv_gate_mean"] = float(gate.mean().item())
            diagnostics["danv_gate_std"] = float(gate.std(unbiased=False).item())
        if danv_degree_gcl:
            degree_gate = _degree_disagreement_gate(data, config)
            diagnostics["danv_degree_gate_mean"] = float(degree_gate.mean().item())
            diagnostics["danv_degree_gate_std"] = float(degree_gate.std(unbiased=False).item())
        if fdnv_gcl:
            high_gate, low_gate = _fdnv_filter_gate(data, config)
            diagnostics["fdnv_high_gate_mean"] = float(high_gate.mean().item())
            diagnostics["fdnv_high_gate_std"] = float(high_gate.std(unbiased=False).item())
            diagnostics["fdnv_low_gate_mean"] = float(low_gate.mean().item())
            diagnostics["fdnv_low_gate_std"] = float(low_gate.std(unbiased=False).item())
        if sspnv_gcl:
            sem_first = semantic_idx[:, 0]
            semantic_sim = (
                F.normalize(cache_keys, dim=1)
                * F.normalize(cache_keys[sem_first], dim=1)
            ).sum(dim=1)
            spatial_is_self = spatial_idx == torch.arange(data.num_nodes, device=data.x.device)
            diagnostics["sspnv_semantic_sim_mean"] = float(semantic_sim.mean().item())
            diagnostics["sspnv_semantic_sim_std"] = float(semantic_sim.std(unbiased=False).item())
            diagnostics["sspnv_spatial_self_fraction"] = float(spatial_is_self.float().mean().item())
            diagnostics["sspnv_semantic_topk"] = int(config["sspnv_semantic_topk"])
            diagnostics["sspnv_random_semantic"] = bool(config.get("sspnv_random_semantic", False))
            diagnostics["sspnv_random_spatial"] = bool(config.get("sspnv_random_spatial", False))
        if afpnv_gcl:
            semantic_weight, spatial_weight, semantic_conf, spatial_conf = afpnv_branch_stats
            diagnostics["afpnv_semantic_weight_mean"] = float(semantic_weight.mean().item())
            diagnostics["afpnv_semantic_weight_std"] = float(
                semantic_weight.std(unbiased=False).item()
            )
            diagnostics["afpnv_spatial_weight_mean"] = float(spatial_weight.mean().item())
            diagnostics["afpnv_spatial_weight_std"] = float(
                spatial_weight.std(unbiased=False).item()
            )
            diagnostics["afpnv_semantic_conf_mean"] = float(semantic_conf.mean().item())
            diagnostics["afpnv_spatial_conf_mean"] = float(spatial_conf.mean().item())
        if bspnv_gcl:
            semantic_prob, spatial_prob, bootstrap_prob, semantic_conf, spatial_conf = bspnv_branch_stats
            winners = torch.stack([semantic_prob, spatial_prob, bootstrap_prob], dim=1).argmax(dim=1)
            diagnostics["bspnv_semantic_prob_mean"] = float(semantic_prob.mean().item())
            diagnostics["bspnv_spatial_prob_mean"] = float(spatial_prob.mean().item())
            diagnostics["bspnv_bootstrap_prob_mean"] = float(bootstrap_prob.mean().item())
            diagnostics["bspnv_semantic_win_fraction"] = float((winners == 0).float().mean().item())
            diagnostics["bspnv_spatial_win_fraction"] = float((winners == 1).float().mean().item())
            diagnostics["bspnv_bootstrap_win_fraction"] = float((winners == 2).float().mean().item())
            diagnostics["bspnv_semantic_conf_mean"] = float(semantic_conf.mean().item())
            diagnostics["bspnv_spatial_conf_mean"] = float(spatial_conf.mean().item())
        if aompnv_gcl:
            semantic_mask, spatial_mask = aompnv_masks
            semantic_conf, spatial_conf = aompnv_confidence
            pred_ego = model.pred_ego(parts["ego"])
            pred_high = model.pred_high(parts["graph"])
            loss_semantic_nodes = multi_positive_info_nce_per_node(
                pred_ego,
                parts["high"],
                semantic_mask,
                float(config["tau"]),
            )
            loss_spatial_nodes = multi_positive_info_nce_per_node(
                pred_ego,
                parts["low"],
                spatial_mask,
                float(config["tau"]),
            )
            loss_bootstrap_nodes = 0.5 * (
                negative_cosine_per_node(pred_ego, parts["graph"])
                + negative_cosine_per_node(pred_high, parts["ego"])
            )
            router_probs = _aompnv_router_weights(
                loss_semantic_nodes,
                loss_spatial_nodes,
                loss_bootstrap_nodes,
                semantic_conf,
                spatial_conf,
                config,
            )
            winners = router_probs.argmax(dim=1)
            diagnostics["aompnv_semantic_prob_mean"] = float(router_probs[:, 0].mean().item())
            diagnostics["aompnv_spatial_prob_mean"] = float(router_probs[:, 1].mean().item())
            diagnostics["aompnv_bootstrap_prob_mean"] = float(router_probs[:, 2].mean().item())
            diagnostics["aompnv_semantic_win_fraction"] = float((winners == 0).float().mean().item())
            diagnostics["aompnv_spatial_win_fraction"] = float((winners == 1).float().mean().item())
            diagnostics["aompnv_bootstrap_win_fraction"] = float((winners == 2).float().mean().item())
            diagnostics["aompnv_semantic_conf_mean"] = float(semantic_conf.mean().item())
            diagnostics["aompnv_spatial_conf_mean"] = float(spatial_conf.mean().item())
            diagnostics["aompnv_semantic_loss_mean"] = float(loss_semantic_nodes.mean().item())
            diagnostics["aompnv_spatial_loss_mean"] = float(loss_spatial_nodes.mean().item())
            diagnostics["aompnv_bootstrap_loss_mean"] = float(loss_bootstrap_nodes.mean().item())
            diagnostics["aompnv_semantic_pos_mean"] = float(semantic_mask.float().sum(dim=1).mean().item())
            diagnostics["aompnv_spatial_pos_mean"] = float(spatial_mask.float().sum(dim=1).mean().item())
            diagnostics["aompnv_shuffle_positives"] = bool(config.get("aompnv_shuffle_positives", False))
        if srgnv_gcl:
            residual_target, residual_score, residual_gate = _srgnv_residual_target(data, parts, config)
            residual_pred = model.pred_high(parts["ego"])
            residual_cos = (
                F.normalize(residual_pred, dim=1)
                * F.normalize(residual_target, dim=1)
            ).sum(dim=1)
            diagnostics["srgnv_raw_residual_score_mean"] = float(residual_score.mean().item())
            diagnostics["srgnv_raw_residual_score_std"] = float(residual_score.std(unbiased=False).item())
            diagnostics["srgnv_residual_gate_mean"] = float(residual_gate.mean().item())
            diagnostics["srgnv_residual_gate_std"] = float(residual_gate.std(unbiased=False).item())
            diagnostics["srgnv_residual_cos_mean"] = float(residual_cos.mean().item())
            diagnostics["srgnv_shuffle_residual"] = bool(config.get("srgnv_shuffle_residual", False))
        if pcnv_gcl:
            _, _, proto_stats = _pcnv_loss(
                parts["ego"],
                parts["graph"],
                pcnv_prototypes,
                config,
            )
            proto = F.normalize(pcnv_prototypes.detach(), dim=1)
            proto_sim = proto @ proto.t()
            off_diag = proto_sim[~torch.eye(proto_sim.size(0), dtype=torch.bool, device=proto_sim.device)]
            diagnostics["pcnv_consistency_loss"] = float(proto_stats["pcnv_consistency_loss"].item())
            diagnostics["pcnv_guarded_consistency_loss"] = float(
                proto_stats["pcnv_guarded_consistency_loss"].item()
            )
            diagnostics["pcnv_balance_loss"] = float(proto_stats["pcnv_balance_loss"].item())
            diagnostics["pcnv_assignment_entropy_mean"] = float(
                proto_stats["pcnv_assignment_entropy_mean"].item()
            )
            diagnostics["pcnv_assignment_max_prob_mean"] = float(
                proto_stats["pcnv_assignment_max_prob_mean"].item()
            )
            diagnostics["pcnv_usage_entropy_mean"] = float(proto_stats["pcnv_usage_entropy_mean"].item())
            diagnostics["pcnv_target_confidence_mean"] = float(
                proto_stats["pcnv_target_confidence_mean"].item()
            )
            diagnostics["pcnv_view_agreement_mean"] = float(
                proto_stats["pcnv_view_agreement_mean"].item()
            )
            diagnostics["pcnv_view_weight_mean"] = float(
                proto_stats["pcnv_view_weight_mean"].item()
            )
            diagnostics["pcnv_target_weight_mean"] = float(
                proto_stats["pcnv_target_weight_mean"].item()
            )
            diagnostics["pcnv_entropy_guard_mean"] = float(
                proto_stats["pcnv_entropy_guard_mean"].item()
            )
            diagnostics["pcnv_num_prototypes"] = int(config["pcnv_num_prototypes"])
            diagnostics["pcnv_entropy_guard"] = bool(config.get("pcnv_entropy_guard", False))
            diagnostics["pcnv_shuffle_assignments"] = bool(config.get("pcnv_shuffle_assignments", False))
            diagnostics["pcnv_prototype_cosine_offdiag_mean"] = float(off_diag.mean().item())
        if lcos_gcl or lcm_gcl:
            gate, score, raw_agreement, raw_residual = _lcos_conflict_gate(data, config)
            diagnostics["lcos_high_gate_mean"] = float(gate.mean().item())
            diagnostics["lcos_high_gate_std"] = float(gate.std(unbiased=False).item())
            diagnostics["lcos_score_mean"] = float(score.mean().item())
            diagnostics["lcos_score_std"] = float(score.std(unbiased=False).item())
            diagnostics["lcos_raw_agreement_mean"] = float(raw_agreement.mean().item())
            diagnostics["lcos_raw_residual_mean"] = float(raw_residual.mean().item())
            diagnostics["lcos_shuffle_gate"] = bool(config.get("lcos_shuffle_gate", False))
        if dsp_gcl:
            weight, margin, view_consistency, score = _dsp_separability_weight(parts, config)
            diagnostics["dsp_weight_mean"] = float(weight.mean().item())
            diagnostics["dsp_weight_std"] = float(weight.std(unbiased=False).item())
            diagnostics["dsp_margin_mean"] = float(margin.mean().item())
            diagnostics["dsp_margin_std"] = float(margin.std(unbiased=False).item())
            diagnostics["dsp_view_consistency_mean"] = float(view_consistency.mean().item())
            diagnostics["dsp_view_consistency_std"] = float(
                view_consistency.std(unbiased=False).item()
            )
            diagnostics["dsp_score_mean"] = float(score.mean().item())
            diagnostics["dsp_score_std"] = float(score.std(unbiased=False).item())
            diagnostics["dsp_shuffle_weight"] = bool(config.get("dsp_shuffle_weight", False))
        if rrnv_gcl:
            pred_ego = model.pred_ego(parts["ego"])
            pred_high = model.pred_high(parts["graph"])
            _, rrnv_stats = _rrnv_loss(pred_ego, pred_high, config)
            cosine = (
                F.normalize(pred_ego, dim=1)
                * F.normalize(pred_high, dim=1)
            ).sum(dim=1)
            diagnostics["rrnv_invariance_loss"] = float(rrnv_stats["rrnv_invariance_loss"].item())
            diagnostics["rrnv_variance_loss"] = float(rrnv_stats["rrnv_variance_loss"].item())
            diagnostics["rrnv_covariance_loss"] = float(rrnv_stats["rrnv_covariance_loss"].item())
            diagnostics["rrnv_pair_cosine_mean"] = float(cosine.mean().item())
            diagnostics["rrnv_pair_cosine_std"] = float(cosine.std(unbiased=False).item())
            diagnostics["rrnv_shuffle_pairs"] = bool(config.get("rrnv_shuffle_pairs", False))
        if dsrrnv_gcl:
            pred_ego = model.pred_ego(parts["ego"])
            pred_high = model.pred_high(parts["graph"])
            _, rrnv_stats = _rrnv_loss(pred_ego, pred_high, config)
            high_gate, avg_degree = _dsrrnv_density_high_gate(data, config)
            cosine = (
                F.normalize(pred_ego, dim=1)
                * F.normalize(pred_high, dim=1)
            ).sum(dim=1)
            diagnostics["rrnv_invariance_loss"] = float(rrnv_stats["rrnv_invariance_loss"].item())
            diagnostics["rrnv_variance_loss"] = float(rrnv_stats["rrnv_variance_loss"].item())
            diagnostics["rrnv_covariance_loss"] = float(rrnv_stats["rrnv_covariance_loss"].item())
            diagnostics["rrnv_pair_cosine_mean"] = float(cosine.mean().item())
            diagnostics["rrnv_pair_cosine_std"] = float(cosine.std(unbiased=False).item())
            diagnostics["rrnv_shuffle_pairs"] = bool(config.get("rrnv_shuffle_pairs", False))
            diagnostics["dsrrnv_high_gate"] = float(high_gate.item())
            diagnostics["dsrrnv_avg_degree"] = float(avg_degree)
        if dirrnv_gcl:
            pred_ego = model.pred_ego(parts["ego"])
            pred_high = model.pred_high(parts["graph"])
            high_gate, avg_degree = _dsrrnv_density_high_gate(data, config)
            invariance_scale = _dirrnv_invariance_scale(high_gate, config)
            _, rrnv_stats = _rrnv_loss(
                pred_ego,
                pred_high,
                config,
                invariance_scale=invariance_scale,
            )
            cosine = (
                F.normalize(pred_ego, dim=1)
                * F.normalize(pred_high, dim=1)
            ).sum(dim=1)
            diagnostics["rrnv_invariance_loss"] = float(rrnv_stats["rrnv_invariance_loss"].item())
            diagnostics["rrnv_variance_loss"] = float(rrnv_stats["rrnv_variance_loss"].item())
            diagnostics["rrnv_covariance_loss"] = float(rrnv_stats["rrnv_covariance_loss"].item())
            diagnostics["rrnv_pair_cosine_mean"] = float(cosine.mean().item())
            diagnostics["rrnv_pair_cosine_std"] = float(cosine.std(unbiased=False).item())
            diagnostics["rrnv_shuffle_pairs"] = bool(config.get("rrnv_shuffle_pairs", False))
            diagnostics["dsrrnv_high_gate"] = float(high_gate.item())
            diagnostics["dsrrnv_avg_degree"] = float(avg_degree)
            diagnostics["dirrnv_invariance_scale"] = float(invariance_scale.item())
        if dprrnv_gcl:
            pred_ego = model.pred_ego(parts["ego"])
            pred_high = model.pred_high(parts["graph"])
            high_gate, avg_degree = _dsrrnv_density_high_gate(data, config)
            shuffle_prob = _dprrnv_shuffle_prob(high_gate, config)
            perturbed_high = _dprrnv_target(
                pred_high,
                shuffle_prob,
                force_shuffle=bool(config.get("rrnv_shuffle_pairs", False)),
            )
            _, rrnv_stats = _rrnv_loss(
                pred_ego,
                perturbed_high,
                config,
                shuffle_pairs=False,
            )
            cosine = (
                F.normalize(pred_ego, dim=1)
                * F.normalize(perturbed_high, dim=1)
            ).sum(dim=1)
            diagnostics["rrnv_invariance_loss"] = float(rrnv_stats["rrnv_invariance_loss"].item())
            diagnostics["rrnv_variance_loss"] = float(rrnv_stats["rrnv_variance_loss"].item())
            diagnostics["rrnv_covariance_loss"] = float(rrnv_stats["rrnv_covariance_loss"].item())
            diagnostics["rrnv_pair_cosine_mean"] = float(cosine.mean().item())
            diagnostics["rrnv_pair_cosine_std"] = float(cosine.std(unbiased=False).item())
            diagnostics["rrnv_shuffle_pairs"] = bool(config.get("rrnv_shuffle_pairs", False))
            diagnostics["dsrrnv_high_gate"] = float(high_gate.item())
            diagnostics["dsrrnv_avg_degree"] = float(avg_degree)
            diagnostics["dprrnv_shuffle_prob"] = float(
                1.0 if bool(config.get("rrnv_shuffle_pairs", False)) else shuffle_prob.item()
            )
        if nprrnv_gcl:
            pred_ego = model.pred_ego(parts["ego"])
            pred_high = model.pred_high(parts["graph"])
            gate_stats = _nprrnv_pair_gate(data, parts, config)
            shuffle_prob = gate_stats["shuffle_prob"]
            perturbed_high = _dprrnv_target(
                pred_high,
                shuffle_prob,
                force_shuffle=bool(config.get("rrnv_shuffle_pairs", False)),
            )
            _, rrnv_stats = _rrnv_loss(
                pred_ego,
                perturbed_high,
                config,
                shuffle_pairs=False,
            )
            cosine = (
                F.normalize(pred_ego, dim=1)
                * F.normalize(perturbed_high, dim=1)
            ).sum(dim=1)
            effective_prob = (
                torch.ones_like(shuffle_prob)
                if bool(config.get("rrnv_shuffle_pairs", False))
                else shuffle_prob
            )
            diagnostics["rrnv_invariance_loss"] = float(rrnv_stats["rrnv_invariance_loss"].item())
            diagnostics["rrnv_variance_loss"] = float(rrnv_stats["rrnv_variance_loss"].item())
            diagnostics["rrnv_covariance_loss"] = float(rrnv_stats["rrnv_covariance_loss"].item())
            diagnostics["rrnv_pair_cosine_mean"] = float(cosine.mean().item())
            diagnostics["rrnv_pair_cosine_std"] = float(cosine.std(unbiased=False).item())
            diagnostics["rrnv_shuffle_pairs"] = bool(config.get("rrnv_shuffle_pairs", False))
            diagnostics["nprrnv_shuffle_gate"] = bool(config.get("nprrnv_shuffle_gate", False))
            diagnostics["dsrrnv_high_gate"] = float(gate_stats["graph_high_gate"].item())
            diagnostics["dsrrnv_avg_degree"] = float(gate_stats["avg_degree"])
            diagnostics["nprrnv_shuffle_prob_mean"] = float(effective_prob.mean().item())
            diagnostics["nprrnv_shuffle_prob_std"] = float(effective_prob.std(unbiased=False).item())
            diagnostics["nprrnv_shuffle_prob_min"] = float(effective_prob.min().item())
            diagnostics["nprrnv_shuffle_prob_max"] = float(effective_prob.max().item())
            diagnostics["nprrnv_node_gate_mean"] = float(gate_stats["node_gate"].mean().item())
            diagnostics["nprrnv_node_gate_std"] = float(gate_stats["node_gate"].std(unbiased=False).item())
            diagnostics["nprrnv_score_mean"] = float(gate_stats["score"].mean().item())
            diagnostics["nprrnv_score_std"] = float(gate_stats["score"].std(unbiased=False).item())
            diagnostics["nprrnv_raw_agreement_mean"] = float(gate_stats["raw_agreement"].mean().item())
            diagnostics["nprrnv_raw_residual_mean"] = float(gate_stats["raw_residual"].mean().item())
            diagnostics["nprrnv_view_cosine_mean"] = float(gate_stats["view_cosine"].mean().item())
            diagnostics["nprrnv_log_degree_mean"] = float(gate_stats["log_degree"].mean().item())
        if rwirrnv_gcl:
            pred_ego = model.pred_ego(parts["ego"])
            pred_high = model.pred_high(parts["graph"])
            reliability, gate_stats = _rwirrnv_reliability(data, parts, config)
            _, rrnv_stats = _rrnv_weighted_invariance_loss(
                pred_ego,
                pred_high,
                reliability,
                config,
            )
            effective_reliability = rrnv_stats["rwirrnv_reliability"]
            cosine = (
                F.normalize(pred_ego, dim=1)
                * F.normalize(pred_high, dim=1)
            ).sum(dim=1)
            diagnostics["rrnv_invariance_loss"] = float(rrnv_stats["rrnv_invariance_loss"].item())
            diagnostics["rrnv_unweighted_invariance_loss"] = float(
                rrnv_stats["rrnv_unweighted_invariance_loss"].item()
            )
            diagnostics["rrnv_variance_loss"] = float(rrnv_stats["rrnv_variance_loss"].item())
            diagnostics["rrnv_covariance_loss"] = float(rrnv_stats["rrnv_covariance_loss"].item())
            diagnostics["rrnv_pair_cosine_mean"] = float(cosine.mean().item())
            diagnostics["rrnv_pair_cosine_std"] = float(cosine.std(unbiased=False).item())
            diagnostics["rrnv_shuffle_pairs"] = bool(config.get("rrnv_shuffle_pairs", False))
            diagnostics["dsrrnv_high_gate"] = float(gate_stats["graph_high_gate"].item())
            diagnostics["dsrrnv_avg_degree"] = float(gate_stats["avg_degree"])
            diagnostics["rwirrnv_shuffle_weight"] = bool(config.get("rwirrnv_shuffle_weight", False))
            diagnostics["rwirrnv_constant_weight"] = bool(config.get("rwirrnv_constant_weight", False))
            diagnostics["rwirrnv_reliability_mean"] = float(effective_reliability.mean().item())
            diagnostics["rwirrnv_reliability_std"] = float(
                effective_reliability.std(unbiased=False).item()
            )
            diagnostics["rwirrnv_reliability_min"] = float(effective_reliability.min().item())
            diagnostics["rwirrnv_reliability_max"] = float(effective_reliability.max().item())
            diagnostics["nprrnv_node_gate_mean"] = float(gate_stats["node_gate"].mean().item())
            diagnostics["nprrnv_node_gate_std"] = float(gate_stats["node_gate"].std(unbiased=False).item())
            diagnostics["nprrnv_score_mean"] = float(gate_stats["score"].mean().item())
            diagnostics["nprrnv_score_std"] = float(gate_stats["score"].std(unbiased=False).item())
            diagnostics["nprrnv_raw_agreement_mean"] = float(gate_stats["raw_agreement"].mean().item())
            diagnostics["nprrnv_raw_residual_mean"] = float(gate_stats["raw_residual"].mean().item())
            diagnostics["nprrnv_view_cosine_mean"] = float(gate_stats["view_cosine"].mean().item())
            diagnostics["nprrnv_log_degree_mean"] = float(gate_stats["log_degree"].mean().item())
        if eairrnv_gcl:
            pred_ego = model.pred_ego(parts["ego"])
            pred_high = model.pred_high(parts["graph"])
            invariance_scale, energy_stats = _eairrnv_invariance_scale(parts, config)
            _, rrnv_stats = _rrnv_loss(
                pred_ego,
                pred_high,
                config,
                invariance_scale=invariance_scale,
            )
            cosine = (
                F.normalize(pred_ego, dim=1)
                * F.normalize(pred_high, dim=1)
            ).sum(dim=1)
            diagnostics["rrnv_invariance_loss"] = float(rrnv_stats["rrnv_invariance_loss"].item())
            diagnostics["rrnv_variance_loss"] = float(rrnv_stats["rrnv_variance_loss"].item())
            diagnostics["rrnv_covariance_loss"] = float(rrnv_stats["rrnv_covariance_loss"].item())
            diagnostics["rrnv_pair_cosine_mean"] = float(cosine.mean().item())
            diagnostics["rrnv_pair_cosine_std"] = float(cosine.std(unbiased=False).item())
            diagnostics["rrnv_shuffle_pairs"] = bool(config.get("rrnv_shuffle_pairs", False))
            diagnostics["eairrnv_energy_ratio_mean"] = float(
                energy_stats["energy_ratio_mean"].item()
            )
            diagnostics["eairrnv_energy_ratio_std"] = float(
                energy_stats["energy_ratio_std"].item()
            )
            diagnostics["eairrnv_conflict"] = float(energy_stats["conflict"].item())
            diagnostics["eairrnv_invariance_scale"] = float(invariance_scale.item())
            diagnostics["eairrnv_energy_threshold"] = float(config["eairrnv_energy_threshold"])
            diagnostics["eairrnv_strength"] = float(config["eairrnv_strength"])
            diagnostics["eairrnv_power"] = float(config["eairrnv_power"])
        if bprrnv_gcl:
            pred_ego = model.pred_ego(parts["ego"])
            pred_high = model.pred_high(parts["graph"])
            gate_stats = _bprrnv_aux_gate(data, parts, config)
            bootstrap_loss = 0.5 * (
                negative_cosine(pred_ego, parts["graph"])
                + negative_cosine(pred_high, parts["ego"])
            )
            _, rrnv_stats = _bprrnv_regularizer_loss(
                pred_ego,
                pred_high,
                config,
                gate_stats["aux_gate"],
            )
            cosine = rrnv_stats["rrnv_pair_cosine"]
            diagnostics["bprrnv_bootstrap_loss"] = float(bootstrap_loss.item())
            diagnostics["bprrnv_regularizer_loss"] = float(
                rrnv_stats["bprrnv_regularizer_loss"].item()
            )
            diagnostics["bprrnv_core_loss"] = float(rrnv_stats["bprrnv_core_loss"].item())
            diagnostics["rrnv_invariance_loss"] = float(rrnv_stats["rrnv_invariance_loss"].item())
            diagnostics["rrnv_variance_loss"] = float(rrnv_stats["rrnv_variance_loss"].item())
            diagnostics["rrnv_covariance_loss"] = float(rrnv_stats["rrnv_covariance_loss"].item())
            diagnostics["rrnv_pair_cosine_mean"] = float(cosine.mean().item())
            diagnostics["rrnv_pair_cosine_std"] = float(cosine.std(unbiased=False).item())
            diagnostics["rrnv_shuffle_pairs"] = bool(config.get("rrnv_shuffle_pairs", False))
            diagnostics["bprrnv_aux_gate"] = float(gate_stats["aux_gate"].item())
            diagnostics["bprrnv_density_factor"] = float(gate_stats["density_factor"].item())
            diagnostics["bprrnv_energy_factor"] = float(gate_stats["energy_factor"].item())
            diagnostics["bprrnv_energy_conflict"] = float(
                gate_stats["energy_conflict"].item()
            )
            diagnostics["bprrnv_energy_ratio_mean"] = float(
                gate_stats["energy_ratio_mean"].item()
            )
            diagnostics["bprrnv_energy_ratio_std"] = float(
                gate_stats["energy_ratio_std"].item()
            )
            diagnostics["bprrnv_avg_degree"] = float(gate_stats["avg_degree"])
            diagnostics["bprrnv_rr_weight"] = float(config["bprrnv_rr_weight"])
            diagnostics["bprrnv_uniform_gate"] = bool(config.get("bprrnv_uniform_gate", False))
            diagnostics["bprrnv_no_density_gate"] = bool(
                config.get("bprrnv_no_density_gate", False)
            )
            diagnostics["bprrnv_no_energy_gate"] = bool(
                config.get("bprrnv_no_energy_gate", False)
            )
        if tns_gcl:
            pred_ego = model.pred_ego(parts["ego"])
            pred_high = model.pred_high(parts["graph"])
            bootstrap_loss = 0.5 * (
                negative_cosine(pred_ego, parts["graph"])
                + negative_cosine(pred_high, parts["ego"])
            )
            _, tns_stats = _tns_loss(parts["final"], cache_keys, config)
            diagnostics["tns_bootstrap_loss"] = float(bootstrap_loss.item())
            diagnostics["tns_loss"] = float(tns_stats["tns_loss"].item())
            diagnostics["tns_weight_mean"] = float(tns_stats["tns_weight_mean"].item())
            diagnostics["tns_weight_std"] = float(tns_stats["tns_weight_std"].item())
            diagnostics["tns_key_sim_mean"] = float(tns_stats["tns_key_sim_mean"].item())
            diagnostics["tns_key_sim_std"] = float(tns_stats["tns_key_sim_std"].item())
            diagnostics["tns_pair_cosine_mean"] = float(
                tns_stats["tns_pair_cosine_mean"].item()
            )
            diagnostics["tns_pair_cosine_std"] = float(
                tns_stats["tns_pair_cosine_std"].item()
            )
            diagnostics["tns_repulsion_active_fraction"] = float(
                tns_stats["tns_repulsion_active_fraction"].item()
            )
            diagnostics["tns_weight"] = float(config["tns_weight"])
            diagnostics["tns_num_negatives"] = int(config["tns_num_negatives"])
            diagnostics["tns_margin"] = float(config["tns_margin"])
            diagnostics["tns_key_threshold"] = float(config["tns_key_threshold"])
            diagnostics["tns_shuffle_weight"] = bool(config.get("tns_shuffle_weight", False))
            diagnostics["tns_uniform_weight"] = bool(config.get("tns_uniform_weight", False))
        if darrnv_gcl:
            pred_ego = model.pred_ego(parts["ego"])
            pred_high = model.pred_high(parts["graph"])
            _, rrnv_stats = _rrnv_loss(pred_ego, pred_high, config)
            density_gate, avg_degree = _darrnv_density_gate(data, config)
            cosine = (
                F.normalize(pred_ego, dim=1)
                * F.normalize(pred_high, dim=1)
            ).sum(dim=1)
            diagnostics["rrnv_invariance_loss"] = float(rrnv_stats["rrnv_invariance_loss"].item())
            diagnostics["rrnv_variance_loss"] = float(rrnv_stats["rrnv_variance_loss"].item())
            diagnostics["rrnv_covariance_loss"] = float(rrnv_stats["rrnv_covariance_loss"].item())
            diagnostics["rrnv_pair_cosine_mean"] = float(cosine.mean().item())
            diagnostics["rrnv_pair_cosine_std"] = float(cosine.std(unbiased=False).item())
            diagnostics["rrnv_shuffle_pairs"] = bool(config.get("rrnv_shuffle_pairs", False))
            diagnostics["darrnv_density_gate"] = float(density_gate.item())
            diagnostics["darrnv_avg_degree"] = float(avg_degree)
            diagnostics["darrnv_rr_weight"] = float(config["darrnv_rr_weight"])
        if mpnv_gcl:
            semantic_mask, spatial_mask = mpnv_masks
            diagnostics["mpnv_semantic_pos_mean"] = float(semantic_mask.float().sum(dim=1).mean().item())
            diagnostics["mpnv_spatial_pos_mean"] = float(spatial_mask.float().sum(dim=1).mean().item())
            diagnostics["mpnv_semantic_density"] = float(semantic_mask.float().mean().item())
            diagnostics["mpnv_spatial_density"] = float(spatial_mask.float().mean().item())
            diagnostics["mpnv_shuffle_positives"] = bool(config.get("mpnv_shuffle_positives", False))
    diagnostics["cache_control"] = (
        ("danv_degree" if danv_degree_gcl else "danv") if danv_gcl else
        "fdnv" if fdnv_gcl else
        ("srgnv_shuffled" if bool(config.get("srgnv_shuffle_residual", False)) else "srgnv") if srgnv_gcl else
        _pcnv_control_name(config) if pcnv_gcl else
        ("lcos_shuffled" if bool(config.get("lcos_shuffle_gate", False)) else "lcos") if lcos_gcl else
        ("lcm_shuffled" if bool(config.get("lcos_shuffle_gate", False)) else "lcm") if lcm_gcl else
        ("dsp_shuffled" if bool(config.get("dsp_shuffle_weight", False)) else "dsp") if dsp_gcl else
        ("rrnv_shuffled" if bool(config.get("rrnv_shuffle_pairs", False)) else "rrnv") if rrnv_gcl else
        ("darrnv_shuffled" if bool(config.get("rrnv_shuffle_pairs", False)) else "darrnv") if darrnv_gcl else
        ("dsrrnv_shuffled" if bool(config.get("rrnv_shuffle_pairs", False)) else "dsrrnv") if dsrrnv_gcl else
        ("dirrnv_shuffled" if bool(config.get("rrnv_shuffle_pairs", False)) else "dirrnv") if dirrnv_gcl else
        ("dprrnv_shuffled" if bool(config.get("rrnv_shuffle_pairs", False)) else "dprrnv") if dprrnv_gcl else
        ("nprrnv_full_shuffled" if bool(config.get("rrnv_shuffle_pairs", False)) else "nprrnv_gate_shuffled" if bool(config.get("nprrnv_shuffle_gate", False)) else "nprrnv") if nprrnv_gcl else
        ("rwirrnv_constant_weight" if bool(config.get("rwirrnv_constant_weight", False)) else "rwirrnv_weight_shuffled" if bool(config.get("rwirrnv_shuffle_weight", False)) else "rwirrnv") if rwirrnv_gcl else
        ("eairrnv_shuffled" if bool(config.get("rrnv_shuffle_pairs", False)) else "eairrnv") if eairrnv_gcl else
        ("bprrnv_uniform" if bool(config.get("bprrnv_uniform_gate", False)) else "bprrnv_no_density" if bool(config.get("bprrnv_no_density_gate", False)) else "bprrnv_no_energy" if bool(config.get("bprrnv_no_energy_gate", False)) else "bprrnv_shuffled" if bool(config.get("rrnv_shuffle_pairs", False)) else "bprrnv") if bprrnv_gcl else
        ("tns_uniform" if bool(config.get("tns_uniform_weight", False)) else "tns_shuffled" if bool(config.get("tns_shuffle_weight", False)) else "tns") if tns_gcl else
        "ragc_train" if ragc_gcl else
        ("aompnv_shuffled" if bool(config.get("aompnv_shuffle_positives", False)) else "aompnv") if aompnv_gcl else
        ("mpnv_shuffled" if bool(config.get("mpnv_shuffle_positives", False)) else "mpnv") if mpnv_gcl else
        "bspnv" if bspnv_gcl else
        "afpnv" if afpnv_gcl else
        _sspnv_control_name(config) if sspnv_gcl else
        "energy_spgcl" if energy_spgcl else
        "gcn_mlp_only" if graph_target else
        "residual_only" if residual_only else
        "disabled_self_only" if args.disable_cache else
        "shuffled" if args.shuffle_cache else
        "normal"
    )
    diagnostics["cache_topk"] = int(topk)
    diagnostics["cache_key_mode"] = config.get("cache_key_mode", "raw_signature")
    return final, history, diagnostics


def evaluate_embeddings(embeddings, data, dataset_name, split_index, config, args):
    if args.skip_eval:
        return {}
    if should_use_mask_eval(dataset_name, data, split_index, config["eval_mode"]):
        train_mask, val_mask, test_mask = split_masks(data, split_index)
        return linear_probe_with_masks(
            embeddings,
            data.y,
            train_mask,
            val_mask,
            test_mask,
        )
    return linear_probe_random(
        embeddings,
        data.y,
        ratio=float(args.eval_ratio),
        seed=int(config["seed"]),
    )


def main():
    args = parse_args()
    config = override_config(load_yaml(args.config), args)
    if args.method == "gcn_mlp_gcl" and args.final_repr is None:
        config["final_repr"] = "ego_graph"
    if args.method in {"danv_gcl", "danv_degree_gcl", "fdnv_gcl", "sspnv_gcl", "afpnv_gcl", "bspnv_gcl", "mpnv_gcl", "aompnv_gcl", "srgnv_gcl", "pcnv_gcl", "lcos_gcl", "lcm_gcl", "dsp_gcl", "rrnv_gcl", "darrnv_gcl", "dsrrnv_gcl", "dirrnv_gcl", "dprrnv_gcl", "nprrnv_gcl", "rwirrnv_gcl", "eairrnv_gcl", "bprrnv_gcl", "tns_gcl", "ragc_gcl"} and args.final_repr is None:
        config["final_repr"] = "ego_graph"
    if args.method == "energy_spgcl" and args.final_repr is None:
        config["final_repr"] = "ego_high"
    set_seed(int(config["seed"]))
    device = get_device(args.gpu_id)
    project_root = Path(__file__).resolve().parents[2]
    data_root = args.data_root or str(project_root / "data")
    dataset = load_dataset(data_root, args.dataset)
    data = dataset[0].to(device)
    run_dir = make_run_dir(args, config)

    stats = graph_stats(dataset, data)
    print(json.dumps({"dataset": args.dataset, **stats}, indent=2, sort_keys=True))
    if args.method == "raw_features":
        embeddings = data.x.detach().float()
        history = []
        diagnostics = {
            "cache_control": "raw_features",
            "raw_feature_dim": int(data.x.size(1)),
        }
    elif args.method == "grace":
        model = GraceModel(
            dataset.num_features,
            int(config["hidden_dim"]),
            int(config["proj_dim"]),
            int(config["num_layers"]),
            float(config["dropout"]),
        ).to(device)
        embeddings, history, diagnostics = train_grace(model, data, config, args)
    else:
        model = EnergyRoutedCacheGCL(
            dataset.num_features,
            int(config["hidden_dim"]),
            int(config["proj_dim"]),
            int(config["num_layers"]),
            float(config["dropout"]),
        ).to(device)
        embeddings, history, diagnostics = train_er_cache_gcl(model, data, config, args)
        if args.method == "ragc_gcl":
            learned_dim = int(embeddings.size(1))
            embeddings = _ragc_embeddings(data.x, embeddings, config)
            ragc_control = config.get("ragc_control", "normal")
            diagnostics["cache_control"] = (
                "ragc_raw_anchor"
                if ragc_control in {"normal", None}
                else f"ragc_{ragc_control}"
            )
            diagnostics["ragc_raw_weight"] = float(config["ragc_raw_weight"])
            diagnostics["ragc_learned_weight"] = float(config["ragc_learned_weight"])
            diagnostics["ragc_control"] = ragc_control
            diagnostics["ragc_raw_dim"] = int(data.x.size(1))
            diagnostics["ragc_learned_dim"] = learned_dim
            diagnostics["ragc_output_dim"] = int(embeddings.size(1))

    metrics = evaluate_embeddings(
        embeddings,
        data,
        args.dataset,
        args.split_index,
        config,
        args,
    )
    payload = {
        "dataset": args.dataset,
        "method": args.method,
        "seed": int(config["seed"]),
        "split_index": int(args.split_index),
        "config": config,
        "graph_stats": stats,
        "metrics": metrics,
        "diagnostics": diagnostics,
    }
    write_json(run_dir / "run.json", payload)
    torch.save(
        {
            "embeddings": embeddings.cpu(),
            "labels": data.y.detach().cpu(),
            "payload": payload,
        },
        run_dir / "artifacts.pt",
    )
    for row in history:
        append_csv(run_dir / "train_log.csv", row)
    summary_row = {
        "run_dir": str(run_dir),
        "dataset": args.dataset,
        "method": args.method,
        "seed": int(config["seed"]),
        "split_index": int(args.split_index),
        "cache_control": diagnostics.get("cache_control", "none"),
        **{f"metric_{key}": value for key, value in metrics.items()},
        **{f"diag_{key}": value for key, value in diagnostics.items()},
    }
    append_csv(Path(args.runs_dir) / "summary.csv", summary_row)
    print(json.dumps(summary_row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
