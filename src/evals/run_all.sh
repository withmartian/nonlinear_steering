#!/bin/bash

for model in olmo-2-7b llama-3.2-3b mistral-7b
do
  for combo in 1 2 3
  do
    echo "Running for model_name=$model and num_combo=$combo"
    python steered_mmlu_eval.py --num_combo $combo --model_name $model
  done
done
