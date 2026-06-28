import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs/split_study")
    parser.add_argument("--baseline-method", default="grace")
    parser.add_argument("--out", default=None)
    parser.add_argument("--aggregate-out", default=None)
    return parser.parse_args()


def read_run_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload.get("metrics", {})
    diagnostics = payload.get("diagnostics", {})
    config = payload.get("config", {})
    return {
        "run_dir": str(path.parent),
        "dataset": payload.get("dataset"),
        "method": payload.get("method"),
        "seed": payload.get("seed"),
        "split_index": payload.get("split_index"),
        "cache_control": diagnostics.get("cache_control", "none"),
        "epochs": config.get("epochs"),
        "F1Mi": metrics.get("F1Mi", metrics.get("accuracy")),
        "F1Ma": metrics.get("F1Ma"),
        "accuracy": metrics.get("accuracy"),
        "best_c": metrics.get("best_c"),
        "energy_ratio_mean": diagnostics.get("energy_ratio_mean"),
        "cache_low_sim_mean": diagnostics.get("cache_low_sim_mean"),
        "sspnv_semantic_sim_mean": diagnostics.get("sspnv_semantic_sim_mean"),
        "sspnv_spatial_self_fraction": diagnostics.get("sspnv_spatial_self_fraction"),
        "sspnv_random_semantic": diagnostics.get("sspnv_random_semantic"),
        "sspnv_random_spatial": diagnostics.get("sspnv_random_spatial"),
        "afpnv_semantic_weight_mean": diagnostics.get("afpnv_semantic_weight_mean"),
        "afpnv_spatial_weight_mean": diagnostics.get("afpnv_spatial_weight_mean"),
        "afpnv_semantic_conf_mean": diagnostics.get("afpnv_semantic_conf_mean"),
        "afpnv_spatial_conf_mean": diagnostics.get("afpnv_spatial_conf_mean"),
        "bspnv_semantic_prob_mean": diagnostics.get("bspnv_semantic_prob_mean"),
        "bspnv_spatial_prob_mean": diagnostics.get("bspnv_spatial_prob_mean"),
        "bspnv_bootstrap_prob_mean": diagnostics.get("bspnv_bootstrap_prob_mean"),
        "bspnv_semantic_win_fraction": diagnostics.get("bspnv_semantic_win_fraction"),
        "bspnv_spatial_win_fraction": diagnostics.get("bspnv_spatial_win_fraction"),
        "bspnv_bootstrap_win_fraction": diagnostics.get("bspnv_bootstrap_win_fraction"),
        "aompnv_semantic_prob_mean": diagnostics.get("aompnv_semantic_prob_mean"),
        "aompnv_spatial_prob_mean": diagnostics.get("aompnv_spatial_prob_mean"),
        "aompnv_bootstrap_prob_mean": diagnostics.get("aompnv_bootstrap_prob_mean"),
        "aompnv_semantic_win_fraction": diagnostics.get("aompnv_semantic_win_fraction"),
        "aompnv_spatial_win_fraction": diagnostics.get("aompnv_spatial_win_fraction"),
        "aompnv_bootstrap_win_fraction": diagnostics.get("aompnv_bootstrap_win_fraction"),
        "aompnv_semantic_conf_mean": diagnostics.get("aompnv_semantic_conf_mean"),
        "aompnv_spatial_conf_mean": diagnostics.get("aompnv_spatial_conf_mean"),
        "aompnv_semantic_loss_mean": diagnostics.get("aompnv_semantic_loss_mean"),
        "aompnv_spatial_loss_mean": diagnostics.get("aompnv_spatial_loss_mean"),
        "aompnv_bootstrap_loss_mean": diagnostics.get("aompnv_bootstrap_loss_mean"),
        "aompnv_semantic_pos_mean": diagnostics.get("aompnv_semantic_pos_mean"),
        "aompnv_spatial_pos_mean": diagnostics.get("aompnv_spatial_pos_mean"),
        "aompnv_shuffle_positives": diagnostics.get("aompnv_shuffle_positives"),
        "srgnv_raw_residual_score_mean": diagnostics.get("srgnv_raw_residual_score_mean"),
        "srgnv_residual_gate_mean": diagnostics.get("srgnv_residual_gate_mean"),
        "srgnv_residual_cos_mean": diagnostics.get("srgnv_residual_cos_mean"),
        "srgnv_shuffle_residual": diagnostics.get("srgnv_shuffle_residual"),
        "pcnv_consistency_loss": diagnostics.get("pcnv_consistency_loss"),
        "pcnv_guarded_consistency_loss": diagnostics.get("pcnv_guarded_consistency_loss"),
        "pcnv_balance_loss": diagnostics.get("pcnv_balance_loss"),
        "pcnv_assignment_entropy_mean": diagnostics.get("pcnv_assignment_entropy_mean"),
        "pcnv_assignment_max_prob_mean": diagnostics.get("pcnv_assignment_max_prob_mean"),
        "pcnv_usage_entropy_mean": diagnostics.get("pcnv_usage_entropy_mean"),
        "pcnv_target_confidence_mean": diagnostics.get("pcnv_target_confidence_mean"),
        "pcnv_view_agreement_mean": diagnostics.get("pcnv_view_agreement_mean"),
        "pcnv_view_weight_mean": diagnostics.get("pcnv_view_weight_mean"),
        "pcnv_target_weight_mean": diagnostics.get("pcnv_target_weight_mean"),
        "pcnv_entropy_guard_mean": diagnostics.get("pcnv_entropy_guard_mean"),
        "pcnv_num_prototypes": diagnostics.get("pcnv_num_prototypes"),
        "pcnv_entropy_guard": diagnostics.get("pcnv_entropy_guard"),
        "pcnv_shuffle_assignments": diagnostics.get("pcnv_shuffle_assignments"),
        "pcnv_prototype_cosine_offdiag_mean": diagnostics.get("pcnv_prototype_cosine_offdiag_mean"),
        "lcos_high_gate_mean": diagnostics.get("lcos_high_gate_mean"),
        "lcos_high_gate_std": diagnostics.get("lcos_high_gate_std"),
        "lcos_score_mean": diagnostics.get("lcos_score_mean"),
        "lcos_score_std": diagnostics.get("lcos_score_std"),
        "lcos_raw_agreement_mean": diagnostics.get("lcos_raw_agreement_mean"),
        "lcos_raw_residual_mean": diagnostics.get("lcos_raw_residual_mean"),
        "lcos_shuffle_gate": diagnostics.get("lcos_shuffle_gate"),
        "dsp_weight_mean": diagnostics.get("dsp_weight_mean"),
        "dsp_weight_std": diagnostics.get("dsp_weight_std"),
        "dsp_margin_mean": diagnostics.get("dsp_margin_mean"),
        "dsp_margin_std": diagnostics.get("dsp_margin_std"),
        "dsp_view_consistency_mean": diagnostics.get("dsp_view_consistency_mean"),
        "dsp_view_consistency_std": diagnostics.get("dsp_view_consistency_std"),
        "dsp_score_mean": diagnostics.get("dsp_score_mean"),
        "dsp_score_std": diagnostics.get("dsp_score_std"),
        "dsp_shuffle_weight": diagnostics.get("dsp_shuffle_weight"),
        "rrnv_invariance_loss": diagnostics.get("rrnv_invariance_loss"),
        "rrnv_unweighted_invariance_loss": diagnostics.get("rrnv_unweighted_invariance_loss"),
        "rrnv_variance_loss": diagnostics.get("rrnv_variance_loss"),
        "rrnv_covariance_loss": diagnostics.get("rrnv_covariance_loss"),
        "rrnv_pair_cosine_mean": diagnostics.get("rrnv_pair_cosine_mean"),
        "rrnv_pair_cosine_std": diagnostics.get("rrnv_pair_cosine_std"),
        "rrnv_shuffle_pairs": diagnostics.get("rrnv_shuffle_pairs"),
        "darrnv_density_gate": diagnostics.get("darrnv_density_gate"),
        "darrnv_avg_degree": diagnostics.get("darrnv_avg_degree"),
        "darrnv_rr_weight": diagnostics.get("darrnv_rr_weight"),
        "dsrrnv_high_gate": diagnostics.get("dsrrnv_high_gate"),
        "dsrrnv_avg_degree": diagnostics.get("dsrrnv_avg_degree"),
        "dirrnv_invariance_scale": diagnostics.get("dirrnv_invariance_scale"),
        "dprrnv_shuffle_prob": diagnostics.get("dprrnv_shuffle_prob"),
        "nprrnv_shuffle_prob_mean": diagnostics.get("nprrnv_shuffle_prob_mean"),
        "nprrnv_shuffle_prob_std": diagnostics.get("nprrnv_shuffle_prob_std"),
        "nprrnv_shuffle_prob_min": diagnostics.get("nprrnv_shuffle_prob_min"),
        "nprrnv_shuffle_prob_max": diagnostics.get("nprrnv_shuffle_prob_max"),
        "nprrnv_node_gate_mean": diagnostics.get("nprrnv_node_gate_mean"),
        "nprrnv_node_gate_std": diagnostics.get("nprrnv_node_gate_std"),
        "nprrnv_score_mean": diagnostics.get("nprrnv_score_mean"),
        "nprrnv_score_std": diagnostics.get("nprrnv_score_std"),
        "nprrnv_raw_agreement_mean": diagnostics.get("nprrnv_raw_agreement_mean"),
        "nprrnv_raw_residual_mean": diagnostics.get("nprrnv_raw_residual_mean"),
        "nprrnv_view_cosine_mean": diagnostics.get("nprrnv_view_cosine_mean"),
        "nprrnv_log_degree_mean": diagnostics.get("nprrnv_log_degree_mean"),
        "nprrnv_shuffle_gate": diagnostics.get("nprrnv_shuffle_gate"),
        "rwirrnv_reliability_mean": diagnostics.get("rwirrnv_reliability_mean"),
        "rwirrnv_reliability_std": diagnostics.get("rwirrnv_reliability_std"),
        "rwirrnv_reliability_min": diagnostics.get("rwirrnv_reliability_min"),
        "rwirrnv_reliability_max": diagnostics.get("rwirrnv_reliability_max"),
        "rwirrnv_shuffle_weight": diagnostics.get("rwirrnv_shuffle_weight"),
        "eairrnv_energy_ratio_mean": diagnostics.get("eairrnv_energy_ratio_mean"),
        "eairrnv_energy_ratio_std": diagnostics.get("eairrnv_energy_ratio_std"),
        "eairrnv_conflict": diagnostics.get("eairrnv_conflict"),
        "eairrnv_invariance_scale": diagnostics.get("eairrnv_invariance_scale"),
        "eairrnv_energy_threshold": diagnostics.get("eairrnv_energy_threshold"),
        "eairrnv_strength": diagnostics.get("eairrnv_strength"),
        "eairrnv_power": diagnostics.get("eairrnv_power"),
        "bprrnv_bootstrap_loss": diagnostics.get("bprrnv_bootstrap_loss"),
        "bprrnv_regularizer_loss": diagnostics.get("bprrnv_regularizer_loss"),
        "bprrnv_core_loss": diagnostics.get("bprrnv_core_loss"),
        "bprrnv_aux_gate": diagnostics.get("bprrnv_aux_gate"),
        "bprrnv_density_factor": diagnostics.get("bprrnv_density_factor"),
        "bprrnv_energy_factor": diagnostics.get("bprrnv_energy_factor"),
        "bprrnv_energy_conflict": diagnostics.get("bprrnv_energy_conflict"),
        "bprrnv_energy_ratio_mean": diagnostics.get("bprrnv_energy_ratio_mean"),
        "bprrnv_energy_ratio_std": diagnostics.get("bprrnv_energy_ratio_std"),
        "bprrnv_avg_degree": diagnostics.get("bprrnv_avg_degree"),
        "bprrnv_rr_weight": diagnostics.get("bprrnv_rr_weight"),
        "bprrnv_uniform_gate": diagnostics.get("bprrnv_uniform_gate"),
        "bprrnv_no_density_gate": diagnostics.get("bprrnv_no_density_gate"),
        "bprrnv_no_energy_gate": diagnostics.get("bprrnv_no_energy_gate"),
        "tns_bootstrap_loss": diagnostics.get("tns_bootstrap_loss"),
        "tns_loss": diagnostics.get("tns_loss"),
        "tns_weight_mean": diagnostics.get("tns_weight_mean"),
        "tns_weight_std": diagnostics.get("tns_weight_std"),
        "tns_key_sim_mean": diagnostics.get("tns_key_sim_mean"),
        "tns_key_sim_std": diagnostics.get("tns_key_sim_std"),
        "tns_pair_cosine_mean": diagnostics.get("tns_pair_cosine_mean"),
        "tns_pair_cosine_std": diagnostics.get("tns_pair_cosine_std"),
        "tns_repulsion_active_fraction": diagnostics.get("tns_repulsion_active_fraction"),
        "tns_weight": diagnostics.get("tns_weight"),
        "tns_num_negatives": diagnostics.get("tns_num_negatives"),
        "tns_margin": diagnostics.get("tns_margin"),
        "tns_key_threshold": diagnostics.get("tns_key_threshold"),
        "tns_shuffle_weight": diagnostics.get("tns_shuffle_weight"),
        "tns_uniform_weight": diagnostics.get("tns_uniform_weight"),
        "raw_feature_dim": diagnostics.get("raw_feature_dim"),
        "ragc_raw_weight": diagnostics.get("ragc_raw_weight"),
        "ragc_learned_weight": diagnostics.get("ragc_learned_weight"),
        "ragc_raw_dim": diagnostics.get("ragc_raw_dim"),
        "ragc_learned_dim": diagnostics.get("ragc_learned_dim"),
        "ragc_output_dim": diagnostics.get("ragc_output_dim"),
        "mpnv_semantic_pos_mean": diagnostics.get("mpnv_semantic_pos_mean"),
        "mpnv_spatial_pos_mean": diagnostics.get("mpnv_spatial_pos_mean"),
        "mpnv_semantic_density": diagnostics.get("mpnv_semantic_density"),
        "mpnv_spatial_density": diagnostics.get("mpnv_spatial_density"),
        "mpnv_shuffle_positives": diagnostics.get("mpnv_shuffle_positives"),
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = [float(v) for v in values if v is not None and v != ""]
    if not values:
        return ""
    return sum(values) / len(values)


def aggregate(rows, baseline_method):
    key_to_baseline = {}
    for row in rows:
        key = (row["dataset"], row["split_index"], row["seed"])
        if row["method"] == baseline_method:
            key_to_baseline[key] = row

    enriched = []
    for row in rows:
        item = dict(row)
        key = (row["dataset"], row["split_index"], row["seed"])
        baseline = key_to_baseline.get(key)
        if baseline and row["method"] != baseline_method:
            for metric in ["F1Mi", "F1Ma", "accuracy"]:
                base_value = baseline.get(metric)
                value = row.get(metric)
                item[f"delta_vs_{baseline_method}_{metric}"] = (
                    float(value) - float(base_value)
                    if value is not None and base_value is not None
                    else ""
                )
        else:
            for metric in ["F1Mi", "F1Ma", "accuracy"]:
                item[f"delta_vs_{baseline_method}_{metric}"] = ""
        enriched.append(item)

    groups = defaultdict(list)
    for row in enriched:
        groups[(row["dataset"], row["method"], row["cache_control"])].append(row)

    aggregate_rows = []
    for (dataset, method, cache_control), items in sorted(groups.items()):
        agg = {
            "dataset": dataset,
            "method": method,
            "cache_control": cache_control,
            "num_runs": len(items),
            "F1Mi_mean": mean([item.get("F1Mi") for item in items]),
            "F1Ma_mean": mean([item.get("F1Ma") for item in items]),
            "accuracy_mean": mean([item.get("accuracy") for item in items]),
            f"delta_vs_{baseline_method}_F1Mi_mean": mean([
                item.get(f"delta_vs_{baseline_method}_F1Mi") for item in items
            ]),
            f"delta_vs_{baseline_method}_F1Ma_mean": mean([
                item.get(f"delta_vs_{baseline_method}_F1Ma") for item in items
            ]),
            f"delta_vs_{baseline_method}_accuracy_mean": mean([
                item.get(f"delta_vs_{baseline_method}_accuracy") for item in items
            ]),
            "positive_delta_F1Mi_count": sum(
                1 for item in items
                if item.get(f"delta_vs_{baseline_method}_F1Mi") not in ["", None]
                and float(item[f"delta_vs_{baseline_method}_F1Mi"]) > 0
            ),
            "negative_delta_F1Mi_count": sum(
                1 for item in items
                if item.get(f"delta_vs_{baseline_method}_F1Mi") not in ["", None]
                and float(item[f"delta_vs_{baseline_method}_F1Mi"]) < 0
            ),
        }
        aggregate_rows.append(agg)
    return enriched, aggregate_rows


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    rows = [
        read_run_json(path)
        for path in sorted(runs_dir.glob("*/run.json"))
    ]
    rows, aggregate_rows = aggregate(rows, args.baseline_method)
    out = args.out or str(runs_dir / "split_study_runs.csv")
    aggregate_out = args.aggregate_out or str(runs_dir / "split_study_aggregate.csv")
    write_csv(out, rows)
    write_csv(aggregate_out, aggregate_rows)
    print(f"Wrote {len(rows)} run rows to {out}")
    print(f"Wrote {len(aggregate_rows)} aggregate rows to {aggregate_out}")


if __name__ == "__main__":
    main()
