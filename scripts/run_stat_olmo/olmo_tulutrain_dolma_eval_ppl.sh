#!/bin/bash

start_task_id=${1}
stop_task_id=${2}

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "Current task id ${task_id}"
python -m src.vllm_inference --config_files configs/defaults.yaml configs/llm/olmo_defaults.yaml \
configs/llm/stats/7b_tulutrain_dolma_tokenize_fix.yaml --templates ocl_task_id=${task_id} --stat_ppl --skip_eval_ocl_ds
done
