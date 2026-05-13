#!/usr/bin/env python3
"""
RoboReward-4B：论文报告值 vs 本地复现。
汇总 reward_alignment 的两类指标：
1) VOC Pearson r（ID/OOD）
2) MSE loss（ID/OOD，仅用于复现实验间对比）

用法（支持多组实验）:
    python my_scripts/roboreward/summarize_roboreward_reported_vs_repro.py \
        exp_name_1 path/to/all_metrics_1.json \
        [exp_name_2 path/to/all_metrics_2.json ...]

可选:
    --reported-id 0.77 --reported-ood 0.88
    --out-dir baseline_eval_output/rbm_4b_my_abla
    --no-plot

输出:
    <out-dir>/roboreward_4b_reported_vs_repro.md
    <out-dir>/roboreward_4b_reported_vs_repro.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ID_DATASETS = [
    "racer_val",
    "oxe_bc_z_eval",
    "oxe_berkeley_cable_eval",
    "oxe_bridge_v2_eval",
    "oxe_jaco_eval",
    "oxe_toto_eval",
    "oxe_viola_eval",
    "mw_eval",
    "libero_90",
]

OOD_DATASETS = [
    "usc_franka",
    "jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all",
    "usc_trossen",
    "usc_xarm",
    "rfm_new_mit_franka",
    "utd_so101_clean_top",
]

ROBOREWARD_4B_REPORTED = {
    "ra_id": 0.77,
    "ra_ood": 0.88,
}

PEARSON_ROW_DEFS = [
    ("Reward alignment (VOC Pearson r) ↑", "RBM-EVAL-ID", "ra_id"),
    ("Reward alignment (VOC Pearson r) ↑", "RBM-EVAL-OOD", "ra_ood"),
]

LOSS_ROW_DEFS = [
    ("Reward alignment (MSE loss) ↓", "RBM-EVAL-ID", "loss_id"),
    ("Reward alignment (MSE loss) ↓", "RBM-EVAL-OOD", "loss_ood"),
]


def fmt(v: float) -> str:
    if v != v:
        return "-"
    return f"{v:.2f}"


def avg(values: dict[str, float], keys: list[str]) -> float:
    found = [values[k] for k in keys if k in values]
    return sum(found) / len(found) if found else float("nan")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_reward_alignment_metric(data: dict, suffix: str) -> dict[str, float]:
    ra = data.get("reward_alignment", data)
    out: dict[str, float] = {}
    for k, v in ra.items():
        full_suffix = f"/{suffix}"
        if k.endswith(full_suffix) and isinstance(v, (int, float)):
            out[k[: -len(full_suffix)]] = float(v)
    return out


def build_repro_column(path: Path) -> dict[str, float]:
    data = load_json(path)
    pearson = extract_reward_alignment_metric(data, "pearson")
    loss = extract_reward_alignment_metric(data, "loss")
    return {
        "ra_id": avg(pearson, ID_DATASETS),
        "ra_ood": avg(pearson, OOD_DATASETS),
        "loss_id": avg(loss, ID_DATASETS),
        "loss_ood": avg(loss, OOD_DATASETS),
    }


def write_plot(
    out_png: Path,
    reported: dict[str, float],
    repro_columns: list[tuple[str, dict[str, float]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = ["RBM-EVAL-ID", "RBM-EVAL-OOD"]
    keys = ["ra_id", "ra_ood"]
    x = [0, 1]

    series = [("Reported (paper)", reported)] + [
        (f"Reproduced ({name})", vals) for name, vals in repro_columns
    ]
    n_series = len(series)
    total_width = 0.8
    bar_width = total_width / n_series
    start = -total_width / 2 + bar_width / 2

    fig, ax = plt.subplots(figsize=(max(7.0, 2.2 + 1.4 * n_series), 4.6))
    for idx, (label, vals) in enumerate(series):
        offset = start + idx * bar_width
        ys = [vals.get(k, float("nan")) for k in keys]
        ax.bar([v + offset for v in x], ys, width=bar_width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("VOC Pearson r (↑)")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")
    ax.set_title("RoboReward-4B: reported vs reproduced")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare RoboReward-4B reported metrics vs reproduced all_metrics.json."
    )
    parser.add_argument(
        "exp_and_metrics",
        nargs="+",
        help="实验参数对: <exp_name> <all_metrics.json> [<exp_name> <all_metrics.json> ...]",
    )
    parser.add_argument("--reported-id", type=float, default=ROBOREWARD_4B_REPORTED["ra_id"])
    parser.add_argument("--reported-ood", type=float, default=ROBOREWARD_4B_REPORTED["ra_ood"])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("baseline_eval_output/rbm_4b_my_abla"),
        help="写入 md/png 的目录",
    )
    parser.add_argument("--no-plot", action="store_true", help="只写表格，不生成 png")
    args = parser.parse_args()

    if len(args.exp_and_metrics) < 2 or len(args.exp_and_metrics) % 2 != 0:
        print(
            "用法: python summarize_roboreward_reported_vs_repro.py "
            "<exp_name> <all_metrics.json> [<exp_name> <all_metrics.json> ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    repro_columns: list[tuple[str, dict[str, float]]] = []
    for i in range(0, len(args.exp_and_metrics), 2):
        exp_name = args.exp_and_metrics[i]
        metrics_json = Path(args.exp_and_metrics[i + 1])
        if not metrics_json.is_file():
            print(f"找不到 metrics 文件: {metrics_json}", file=sys.stderr)
            sys.exit(1)
        repro_columns.append((exp_name, build_repro_column(metrics_json)))

    reported = {"ra_id": args.reported_id, "ra_ood": args.reported_ood}

    col_names = ["RoboReward-4B (reported)"] + [f"{n} (reproduced)" for n, _ in repro_columns]
    header = "| Metric | Dataset | " + " | ".join(col_names) + " |"
    sep = "|---|---| " + " | ".join(["---:"] * len(col_names)) + " |"

    lines = [
        "# RoboReward-4B: Reward Alignment 指标对比",
        "",
        f"- **报告基线**: ID={reported['ra_id']:.2f}, OOD={reported['ra_ood']:.2f}（可用 `--reported-id` / `--reported-ood` 覆盖）",
    ]
    for name, _ in repro_columns:
        lines.append(f"- **复现实验**: `{name}`")
    lines.extend(["", header, sep])

    value_columns = [reported] + [vals for _, vals in repro_columns]
    for metric_name, dataset_name, metric_key in PEARSON_ROW_DEFS:
        row_cells = [fmt(col.get(metric_key, float("nan"))) for col in value_columns]
        lines.append(f"| {metric_name} | {dataset_name} | " + " | ".join(row_cells) + " |")

    # Loss 无论文基线，这里仅做复现实验间横向对比（reported 列置为 -）
    for metric_name, dataset_name, metric_key in LOSS_ROW_DEFS:
        row_cells = ["-"] + [fmt(vals.get(metric_key, float("nan"))) for _, vals in repro_columns]
        lines.append(f"| {metric_name} | {dataset_name} | " + " | ".join(row_cells) + " |")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "roboreward_4b_reported_vs_repro.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_png = out_dir / "roboreward_4b_reported_vs_repro.png"
    if not args.no_plot:
        try:
            write_plot(out_png, reported, repro_columns)
        except ModuleNotFoundError as e:
            print(f"[plot] skip: {e}")

    print(f"[summary] wrote: {out_md}")
    if out_png.exists():
        print(f"[plot]    wrote: {out_png}")
    print()
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
