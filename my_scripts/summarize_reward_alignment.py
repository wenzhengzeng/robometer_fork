#!/usr/bin/env python3
"""
用法:
    python my_scripts/summarize_reward_alignment.py \
        exp_name_1 path/to/all_metrics_1.json \
        exp_name_2 path/to/all_metrics_2.json \
        [exp_name_3 path/to/all_metrics_3.json ...]

输出: 与第一个 all_metrics.json 同目录下的 reward_alignment_summary.md
（含 VOC Pearson r、MSE loss，以及 policy ranking tau）
"""
import json
import sys
from pathlib import Path


DATASET_ROWS = [
    ("RBM-EVAL-ID", "RACER (Val)", "racer_val"),
    ("RBM-EVAL-ID", "OXE (BC-Z Eval)", "oxe_bc_z_eval"),
    ("RBM-EVAL-ID", "OXE (Berkeley Cable Routing Eval)", "oxe_berkeley_cable_eval"),
    ("RBM-EVAL-ID", "OXE (Bridge V2 Eval)", "oxe_bridge_v2_eval"),
    ("RBM-EVAL-ID", "OXE (Jaco Play Eval)", "oxe_jaco_eval"),
    ("RBM-EVAL-ID", "OXE (Toto Eval)", "oxe_toto_eval"),
    ("RBM-EVAL-ID", "OXE (Viola Eval)", "oxe_viola_eval"),
    ("RBM-EVAL-ID", "Metaworld (Eval)", "mw_eval"),
    ("RBM-EVAL-ID", "Libero (90)", "libero_90"),
    ("RBM-EVAL-OOD", "USC Franka", "usc_franka"),
    ("RBM-EVAL-OOD", "USC Koch", "jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all"),
    ("RBM-EVAL-OOD", "USC Trossen", "usc_trossen"),
    ("RBM-EVAL-OOD", "USC xArm", "usc_xarm"),
    ("RBM-EVAL-OOD", "MIT Franka", "rfm_new_mit_franka"),
    ("RBM-EVAL-OOD", "UTD; SO101", "utd_so101_clean_top"),
]

PAPER_COLUMNS = {
    "ROBOMETER": {
        "racer_val": 0.937,
        "oxe_bc_z_eval": 0.683,
        "oxe_berkeley_cable_eval": 0.900,
        "oxe_bridge_v2_eval": 0.633,
        "oxe_jaco_eval": 0.861,
        "oxe_toto_eval": 0.947,
        "oxe_viola_eval": 0.967,
        "mw_eval": 0.737,
        "libero_90": 0.912,
        "usc_franka": 0.959,
        "jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all": 0.969,
        "usc_trossen": 0.925,
        "usc_xarm": 0.951,
        "rfm_new_mit_franka": 0.868,
        "utd_so101_clean_top": 0.888,
    },
    "ROBOMETER (w/ RBM-1M)": {
        "racer_val": 0.943,
        "oxe_bc_z_eval": 0.922,
        "oxe_berkeley_cable_eval": 0.887,
        "oxe_bridge_v2_eval": 0.920,
        "oxe_jaco_eval": 0.872,
        "oxe_toto_eval": 0.930,
        "oxe_viola_eval": 0.947,
        "mw_eval": 0.900,
        "libero_90": 0.967,
        "usc_franka": 0.959,
        "jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all": 0.950,
        "usc_trossen": 0.911,
        "usc_xarm": 0.961,
        "rfm_new_mit_franka": 0.954,
        "utd_so101_clean_top": 0.952,
    },
}

POLICY_METRIC_PRIORITY = [
    "kendall_avg",
    "kendall_last",
    "kendall_sum",
    "kendall",
    "spearman_avg",
    "spearman_last",
    "spearman_sum",
    "ranking_acc_avg",
    "ranking_acc_last",
    "ranking_acc_sum",
]


def load_all_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_reward_alignment_metrics(all_metrics: dict) -> tuple[dict, dict]:
    """从 all_metrics 读取 reward_alignment 的 pearson 和 loss。"""
    metrics = all_metrics.get("reward_alignment", {})
    pearson = {}
    loss = {}
    for k, v in metrics.items():
        if not isinstance(v, (float, int)):
            continue
        if k.endswith("/pearson"):
            pearson[k[: -len("/pearson")]] = float(v)
        elif k.endswith("/loss"):
            loss[k[: -len("/loss")]] = float(v)
    return pearson, loss


def load_policy_ranking_metrics(all_metrics: dict) -> dict:
    """从 all_metrics 读取 policy_ranking 的 tau/correlation（按优先级选一个）。"""
    metrics = all_metrics.get("policy_ranking", {})
    grouped = {}
    for k, v in metrics.items():
        if not isinstance(v, (float, int)):
            continue
        if "/" not in k:
            continue
        ds, metric_name = k.split("/", 1)
        grouped.setdefault(ds, {})[metric_name] = float(v)

    selected = {}
    for ds, ds_metrics in grouped.items():
        for metric_name in POLICY_METRIC_PRIORITY:
            if metric_name in ds_metrics:
                selected[ds] = ds_metrics[metric_name]
                break
    return selected


def avg_for_split(values: dict, split: str) -> float:
    keys = [k for s, _, k in DATASET_ROWS if s == split]
    vals = [values[k] for k in keys if k in values]
    return sum(vals) / len(vals) if vals else float("nan")


def avg_values(values: dict) -> float:
    vals = [v for v in values.values() if isinstance(v, (float, int))]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(x):
    if x != x:
        return "-"
    return f"{x:.3f}"


def fmt_mse(x):
    if x != x:
        return "-"
    return f"{x:.4f}"


def main():
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2 != 0:
        print("用法: python summarize_reward_alignment.py <exp_name> <metrics.json> [<exp_name> <metrics.json> ...]")
        sys.exit(1)

    experiments = []
    for i in range(0, len(args), 2):
        name, path = args[i], Path(args[i + 1])
        raw = load_all_metrics(path)
        pearson, loss = load_reward_alignment_metrics(raw)
        policy = load_policy_ranking_metrics(raw)
        experiments.append((name, pearson, loss, policy))

    # reward alignment section: 只包含真正有 reward_alignment 的实验列
    ra_experiments = [(n, p, l) for n, p, l, _ in experiments if p or l]

    all_columns = list(PAPER_COLUMNS.keys()) + [name for name, _, _ in ra_experiments]
    all_values = list(PAPER_COLUMNS.values()) + [vals for _, vals, _ in ra_experiments]

    nan_col = {key: float("nan") for _, _, key in DATASET_ROWS}
    paper_loss_cols = [nan_col.copy() for _ in PAPER_COLUMNS]
    all_loss_values = paper_loss_cols + [loss for _, _, loss in ra_experiments]

    header = "| Split | Dataset | " + " | ".join(all_columns) + " |"
    sep = "|---|---| " + " | ".join(["---:"] * len(all_columns)) + " |"

    lines = ["# Reward Alignment Summary", ""]
    for name, _, _, _ in experiments:
        lines.append(f"- **{name}**: metrics from eval run")

    if ra_experiments:
        lines.extend(["", "## VOC Pearson r", "", header, sep])

        for split in ("RBM-EVAL-ID", "RBM-EVAL-OOD"):
            for s, display_name, key in DATASET_ROWS:
                if s != split:
                    continue
                cells = [fmt(col.get(key, float("nan"))) for col in all_values]
                lines.append(f"| {split} | {display_name} | " + " | ".join(cells) + " |")

            avg_cells = [fmt(avg_for_split(col, split)) for col in all_values]
            lines.append(f"| **{split}** | **Average** | " + " | ".join(f"**{c}**" for c in avg_cells) + " |")

        lines.extend(["", "## MSE loss (continuous) / eval loss metric", "", header, sep])
        for split in ("RBM-EVAL-ID", "RBM-EVAL-OOD"):
            for s, display_name, key in DATASET_ROWS:
                if s != split:
                    continue
                cells = [fmt_mse(col.get(key, float("nan"))) for col in all_loss_values]
                lines.append(f"| {split} | {display_name} | " + " | ".join(cells) + " |")

            avg_cells = [fmt_mse(avg_for_split(col, split)) for col in all_loss_values]
            lines.append(f"| **{split}** | **Average** | " + " | ".join(f"**{c}**" for c in avg_cells) + " |")

    # policy ranking section
    policy_experiments = [(n, pol) for n, _, _, pol in experiments if pol]
    if policy_experiments:
        lines.extend(["", "## Policy ranking (tau / correlation)", ""])
        p_header = "| Dataset | " + " | ".join(name for name, _ in policy_experiments) + " |"
        p_sep = "|---| " + " | ".join(["---:"] * len(policy_experiments)) + " |"
        lines.extend([p_header, p_sep])

        # 收集所有出现过的数据集键
        all_policy_datasets = sorted({ds for _, pol in policy_experiments for ds in pol.keys()})
        for ds in all_policy_datasets:
            cells = [fmt(pol.get(ds, float("nan"))) for _, pol in policy_experiments]
            lines.append(f"| {ds} | " + " | ".join(cells) + " |")

        avg_cells = [fmt(avg_values(pol)) for _, pol in policy_experiments]
        lines.append(f"| **Average** | " + " | ".join(f"**{c}**" for c in avg_cells) + " |")

    first_metrics = Path(args[1]).resolve()
    out_path = first_metrics.parent / "reward_alignment_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summary] wrote: {out_path}")

    print()
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
