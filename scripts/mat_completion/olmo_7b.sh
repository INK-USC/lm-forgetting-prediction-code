#!/bin/bash

for ((seed = 0 ; seed < 20 ; seed++ ))
do
for method in additive knn knn_baseline svd
do

python -m src.run_matrix_completion ${method} olmo-7b-dolma --known_k 30 --seed ${seed}


done
done