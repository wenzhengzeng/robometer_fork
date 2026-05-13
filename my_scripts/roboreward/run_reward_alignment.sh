#!/usr/bin/env bash
set -euo pipefail

source /users/wenzheng/anaconda3/etc/profile.d/conda.sh
conda activate robometer

export ROBOMETER_PROCESSED_DATASETS_PATH=/storage/wenzheng/dataset/robotics/robometer_processed_datasets
export CUDA_VISIBLE_DEVICES=3

cd /users/wenzheng/code/robotics/reward_modeling/robometer_fork

# ============================================================
# 实验定义（只改这里就行）
# ============================================================
EXP1_NAME="roboreward_bug_fix"
RUN_EXP1=true

OUT1="baseline_eval_output/rbm_4b_my_abla/${EXP1_NAME}"

# ============================================================
# 跑实验
# ============================================================
echo "====== [1/5] ${EXP1_NAME} ======"
if [[ "${RUN_EXP1}" == "true" ]]; then
    uv run python robometer/evals/run_baseline_eval.py \
        reward_model=roboreward \
        model_path=teetone/RoboReward-4B \
        custom_eval.eval_types=[reward_alignment] \
        custom_eval.reward_alignment=[rbm-1m-id,rbm-1m-ood] \
        custom_eval.use_frame_steps=true \
        custom_eval.subsample_n_frames=5 \
        custom_eval.reward_alignment_max_trajectories=30 \
        max_frames=64 \
        output_dir="${OUT1}"
else
    echo "[skip] use existing metrics: ${OUT1}/all_metrics.json"
fi

# ============================================================
# 汇总表格
# ============================================================

# echo "====== [5/5] RoboReward-4B reported vs reproduced ======"
python my_scripts/roboreward/summarize_roboreward_reported_vs_repro.py \
    "${EXP1_NAME}" "${OUT1}/all_metrics.json" \

#     --out-dir baseline_eval_output/rbm_4b_my_abla

echo "[done] see: baseline_eval_output/rbm_4b_my_abla/roboreward_4b_reported_vs_repro.md"
echo "[done] see: baseline_eval_output/rbm_4b_my_abla/roboreward_4b_reported_vs_repro.png"
