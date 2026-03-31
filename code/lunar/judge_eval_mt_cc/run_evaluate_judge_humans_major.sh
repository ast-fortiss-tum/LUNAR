#!/bin/bash

# Ensure PYTHONPATH sees the root module
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the evaluation script
python judge_eval_mt_cc/evaluate_judge_humans_major.py \
    --data_folder ./judge_eval_mt_cc/data \
    --majority_csv ./judge_eval_mt_cc/labels/majority_labels_cc.csv \
    --output_folder ./judge_eval_mt_cc/out \
    --num_samples 1 \
    --aggregator mean \
    --llm_types GPT_4O GPT_4O_MINI GPT_5_MINI DEEPSEEK_V3
