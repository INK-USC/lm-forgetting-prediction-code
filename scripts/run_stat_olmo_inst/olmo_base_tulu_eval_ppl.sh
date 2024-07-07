#!/bin/bash

start_task_id=${1}
stop_task_id=${2}

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "Tulu eval: Current task id ${task_id}"

if [[ ${task_id} == 0 ]]; then

python -m src.vllm_inference --config_files configs/defaults.yaml configs/llm/olmo_defaults.yaml \
configs/llm/stats/7b_mmlu_tulu.yaml --templates ocl_task_id=${task_id} --stat_ppl --eval_base

else

python -m src.vllm_inference --config_files configs/defaults.yaml configs/llm/olmo_defaults.yaml \
configs/llm/stats/7b_mmlu_tulu.yaml --templates ocl_task_id=${task_id} --stat_ppl --skip_eval_pt_ds --eval_base

fi
done