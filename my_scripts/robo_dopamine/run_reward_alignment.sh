#!/usr/bin/env bash
set -euo pipefail

source /users/wenzheng/anaconda3/etc/profile.d/conda.sh
conda activate robometer

export ROBOMETER_PROCESSED_DATASETS_PATH=/storage/wenzheng/dataset/robotics/robometer_processed_datasets
export CUDA_VISIBLE_DEVICES=1

cd /users/wenzheng/code/robotics/reward_modeling/robometer_fork

# ============================================================
# 实验定义（只改这里就行）
# ============================================================
EXP1_NAME="20_trajectories_per_dataset"
EXP1_USE_TRUE_ENDPOINT_INDEX=false

EXP2_NAME="30_trajectories_per_dataset"
EXP2_USE_TRUE_ENDPOINT_INDEX=false

OUT1="baseline_eval_output/rbm_4b_my_abla/dopamine2_4B/${EXP1_NAME}"
OUT2="baseline_eval_output/rbm_4b_my_abla/dopamine2_4B/${EXP2_NAME}"
# ============================================================
# 跑实验
# ============================================================

export VLLM_ENABLE_V1_MULTIPROCESSING=0   # for vllm debug使用
echo "====== [1/2] ${EXP1_NAME} ======"
# Robo-Dopamine 4B
.venv-robodopamine/bin/python robometer/evals/run_baseline_eval.py \
    reward_model=robodopamine \
    model_path=tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview \
    model_config.eval_mode=forward \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rbm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.reward_alignment_max_trajectories=1 \
    max_frames=64 \
    model_config.batch_size=1 \
    output_dir="${OUT1}"




# echo "====== [2/2] ${EXP2_NAME} ======"
# .venv-robodopamine/bin/python robometer/evals/run_baseline_eval.py \
#     reward_model=robodopamine \
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



# ============================================================
# 汇总表格
# ============================================================
echo "====== [2/2] build dopamine-only summary table ======"
python my_scripts/robo_dopamine/summarize_dopamine_only.py \
    "${EXP1_NAME}" "${OUT1}/all_metrics.json"

echo "[done] see: baseline_eval_output/rbm_4b_my_abla/dopamine_only_summary.md"
