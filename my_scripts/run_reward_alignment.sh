#!/usr/bin/env bash
set -euo pipefail

source /users/wenzheng/anaconda3/etc/profile.d/conda.sh
conda activate robometer

export ROBOMETER_PROCESSED_DATASETS_PATH=/storage/wenzheng/dataset/robotics/robometer_processed_datasets
export CUDA_VISIBLE_DEVICES=1

# 让 torchcodec 能找到 .venv 中 av 包内置的 FFmpeg 8 共享库
# export LD_LIBRARY_PATH="/users/wenzheng/code/robotics/reward_modeling/robometer_fork/.venv/lib/python3.10/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

cd /users/wenzheng/code/robotics/reward_modeling/robometer_fork

# ============================================================
# 实验定义（只改这里就行）
# ============================================================
EXP_NAME="after_fix"
OUT_BASE="baseline_eval_output/rbm_4b_my_abla"
OUT1="${OUT_BASE}/${EXP_NAME}"
OUT_POLICY="${OUT_BASE}/${EXP_NAME}_policy_ranking"

# ============================================================
# 跑实验
# ============================================================
# echo "====== [1/3] ${EXP1_NAME} ======"
# uv run python robometer/evals/run_baseline_eval.py \
#     reward_model=rbm \
#     model_path=robometer/Robometer-4B \
#     custom_eval.eval_types=[reward_alignment] \
#     custom_eval.reward_alignment=[rbm-1m-id,rbm-1m-ood] \
#     custom_eval.use_frame_steps=true \
#     custom_eval.last_frame_only=false \
#     custom_eval.subsample_n_frames=5 \
#     custom_eval.reward_alignment_max_trajectories=30 \
#     max_frames=8 \
#     model_config.batch_size=16 \
#     custom_eval.use_true_endpoint_index="${EXP1_USE_TRUE_ENDPOINT_INDEX}" \
#     output_dir="${OUT1}"

# echo "====== [3/3] build summary table ======"
# python my_scripts/summarize_reward_alignment.py \
#     "${EXP1_NAME}" "${OUT1}/all_metrics.json"

# echo "====== [2/3] ${EXP2_NAME} ======"
# uv run python robometer/evals/run_baseline_eval.py \
#     reward_model=rbm \
#     model_path=robometer/Robometer-4B \
#     custom_eval.eval_types=[reward_alignment] \
#     custom_eval.reward_alignment=[rbm-1m-id,rbm-1m-ood] \
#     custom_eval.use_frame_steps=true \
#     custom_eval.last_frame_only=false \
#     custom_eval.subsample_n_frames=5 \
#     custom_eval.reward_alignment_max_trajectories=30 \
#     max_frames=8 \
#     model_config.batch_size=16 \
#     custom_eval.use_true_endpoint_index="${EXP2_USE_TRUE_ENDPOINT_INDEX}" \
#     output_dir="${OUT2}"



# uv run python robometer/evals/run_baseline_eval.py \
#     reward_model=rbm \
#     model_path=robometer/Robometer-4B \
#     custom_eval.eval_types=[reward_alignment] \
#     custom_eval.reward_alignment=[rbm-1m-ood] \
#     custom_eval.use_frame_steps=true \
#     custom_eval.last_frame_only=false \
#     custom_eval.subsample_n_frames=5 \
#     custom_eval.reward_alignment_max_trajectories=1 \
#     max_frames=8 \
#     model_config.batch_size=16 \
#     output_dir="${OUT1}"

# uv run python robometer/evals/run_baseline_eval.py \
#     reward_model=rbm \
#     model_path=robometer/Robometer-4B \
#     custom_eval.eval_types=[reward_alignment] \
#     custom_eval.reward_alignment=[rbm-1m-id,rbm-1m-ood] \
#     custom_eval.use_frame_steps=true \
#     custom_eval.subsample_n_frames=5 \
#     custom_eval.reward_alignment_max_trajectories=30 \
#     max_frames=8 \
#     model_config.batch_size=16 \
#     output_dir="${OUT1}"

# ============================================================
# policy ranking（用于复验）
# ============================================================
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rbm \
    model_path=robometer/Robometer-4B \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rbm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.num_examples_per_quality_pr=1000 \
    max_frames=8 \
    model_config.batch_size=32 \
    output_dir="${OUT_POLICY}"

# ============================================================
# 汇总表格
# ============================================================
echo "====== build summary table ======"
uv run python my_scripts/summarize_reward_alignment.py \
    "${EXP_NAME}" "${OUT1}/all_metrics.json" \
    "${EXP_NAME}_policy_ranking" "${OUT_POLICY}/all_metrics.json"

echo "[done] see: ${OUT1}/reward_alignment_summary.md"
echo "[done] policy ranking: ${OUT_POLICY}/all_metrics.json"
