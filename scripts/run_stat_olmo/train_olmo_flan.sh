#!/bin/bash

start_task_id=${1}
stop_task_id=${2}

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "Current task id ${task_id}"
python -m src.run_ocl_train --config_files configs/defaults.yaml configs/llm/olmo_defaults.yaml configs/llm/ocl/7b_peft_flan_5kstep.yaml \
configs/llm/ocl/greedy.yaml configs/llm/ocl/lr1e-4.yaml \
--ocl_task flan --n_gpu 1 --skip_before_eval \
--templates TASK_ID=${task_id}
done