#!/bin/bash

python -m src.train_rep_dot_fpd --config_files configs/defaults.yaml configs/llm/olmo_defaults.yaml \
configs/fpd/fpd_defaults.yaml configs/fpd/fpd_rep_task.yaml configs/data/fpd_olmo_instruct_presplit.yaml \
configs/fpd/lr1e-5.yaml configs/fpd/step100k.yaml configs/fpd/lr_scale10.yaml \
--templates postfix=_fpd_olmo_inst_tulu/rep-based-1e-5 task=mmlu_bbh --ocl_task mmlu_bbh --do_train --eval_step 1000 --skip_first_eval

python -m src.train_rep_dot_fpd --config_files configs/defaults.yaml configs/llm/olmo_defaults.yaml \
configs/fpd/fpd_defaults.yaml configs/fpd/fpd_rep_task_bin.yaml configs/data/fpd_olmo_instruct_presplit.yaml \
configs/fpd/lr1e-5.yaml configs/fpd/step100k.yaml configs/fpd/lr_scale10.yaml \
--templates postfix=_fpd_olmo_inst_tulu/rep-based-1e-5-bin task=mmlu_bbh --ocl_task mmlu_bbh --do_train --eval_step 1000 --skip_first_eval