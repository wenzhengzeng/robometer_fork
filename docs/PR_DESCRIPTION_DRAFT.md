# Pull Request Description

I would like to thank all the authors for their excellent work and for open-sourcing this project to the community!

This pull request fixes several bugs in the current implementation (see below).

---

## 1. Sample index in reward alignment metric computation

### What went wrong

When **`use_frame_steps=true`** (default setting of RoboMeter and RoboReward), multiple positions are sampled evenly from the full video to compute metrics. The previous implementation used the `enumerate` index as the retrieval index when aggregating predictions and targets (see the loop starting at [this line](https://github.com/robometer/robometer/blob/main/robometer/evals/compile_results.py#L363C13-L363C50), where **`timestep`** is the loop position, not the **actual subsampled frame index** in the original video). That results in the sampled indices being fixed to **`0, 1, 2, …`** (i.e. `range(0, custom_eval.subsample_n_frames)`, e.g. **`0 … 4`** when `subsample_n_frames=5`) for every video, instead of the **actual subsampled frame indices** in the original video (e.g. evenly spaced timesteps along the clip).

### What this PR does

I added **`original_sampled_index`** to record the sampled index in the original video. It is populated consistently (including after subsampling / padding), used in **`compile_results`** when aggregating metrics, and the `enumerate`-based path is kept only as a fallback when no valid index is available—so alignment follows the real sampling, not the loop position.

### Performance comparison (VOC average Pearson · policy ranking τ) based on my reproduction


| Metric (split)                      | ROBOMETER w/ RoboReward Training Data (from paper) | ROBOMETER (w/ RBM-1M) (from paper) | reimplementation before this fix | new implementation |
| ----------------------------------- | -------------------------------------------------- | ---------------------------------- | -------------------------------- | ------------------ |
| **VOC r** (RBM-EVAL-ID)             | **0.842**                                          | **0.921**                          | **0.841**                        | **0.860**          |
| **VOC r** (RBM-EVAL-OOD)            | **0.927**                                          | **0.948**                          | **0.927**                        | **0.886**          |
| **policy ranking τ** (RBM-EVAL-OOD) | **0.552**                                          | **0.655**                          | **0.645**                        | **0.645**          |


While reproducing the released code, I noticed that **before this fix**, VOC numbers line up more closely with the paper’s **“w/ RoboReward Training Data”** column than with the other reported columns. Could the authors help clarify whether this is expected, or whether some eval/config details on my side may still be misaligned with the paper?

---

## 2. Robo Dopamine implementation

### What went wrong

For **reward alignment** evaluation, the current behavior **includes padded frames** in the metric computation.

### What this PR does

To exclude padding, set `custom_eval.pad_frames=false` in the run script (If I understand correctly, the original Robo Dopamine setup did not use padding, and whether use padding will not influence the inference result, but it can change calculated metrics when padded repetitive frames are included in the metric aggregation).

---

## 3. Minor environment/requirements updates

I found that some dependency versions released after March 21 can introduce minor incompatibilities. This PR changes the requirements slightly so that packages do not conflict with each other.

---

## 4. vLLM GPU selection at startup (for Robo-Dopamine)

The previous startup path could overwrite externally provided GPU-related environment variables (e.g., `CUDA_VISIBLE_DEVICES` / `LOCAL_RANK`) with hardcoded defaults.

This PR changes the initialization logic to respect pre-set environment variables and only apply defaults when they are missing. This makes vLLM startup GPU assignment configurable from the launch environment (for example, shell scripts or scheduler settings), instead of being forced to a fixed device.

---

Thanks again for the solid work—looking forward to your feedback on this PR. :)