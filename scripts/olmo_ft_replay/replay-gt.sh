#!/bin/bash

replay_every=8
temp=0.2

train_split_task="65 18 8 60 6 13 37 22 30 19 64 50 25 31 32 61 16 5 53 49"

for task_id in ${train_split_task}
do
  echo "Task id ${task_id}"
  config="temp_${temp}_re${replay_every}"
  python -m src.run_ocl_train --config_files configs/defaults.yaml configs/llm/olmo_defaults.yaml \
 configs/llm/ocl/7b_peft_flan_5kstep.yaml configs/llm/ocl/mir_pred_dolma_flan.yaml configs/llm/ocl/mir_pred_temp.yaml \
 configs/llm/ocl/greedy.yaml configs/llm/ocl/replay_every.yaml \
 configs/llm/ocl/lr1e-4.yaml \
 --ocl_task flan --n_gpu 1 --skip_before_eval --templates TASK_ID=${task_id} \
 CONFIG=${config} weight_temp=${temp} replay_freq=${replay_every}

  python vllm_inference.py --config_files configs/defaults.yaml configs/llm/olmo_defaults.yaml \
  configs/llm/stats/7b_flan_dolma_tokenize_fix_cl.yaml --templates ocl_task_id=${task_id} \
  task_model_dir="runs_olmo_ocl/flan-5k-mirpred-gt-${config}/task_${task_id}/model_save" \
  cl_method="mirpred-gt-${config}" --stat_ppl --skip_eval_ocl_ds

done