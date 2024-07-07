#!/bin/bash


task_id=0
python -m src.vllm_inference --config_files configs/defaults.yaml configs/llm/olmo_defaults.yaml \
configs/llm/stats/7b_flan_dolma_tokenize_fix.yaml --templates ocl_task_id=${task_id} --stat_ppl --skip_eval_ocl_ds --eval_base
