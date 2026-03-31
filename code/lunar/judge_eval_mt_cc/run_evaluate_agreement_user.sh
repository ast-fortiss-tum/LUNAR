#!/bin/bash

# Specify paths
INPUT_DIR="./judge_eval_mt_cc/raters"
OUTPUT_DIR="./judge_eval_mt_cc/labels"
MERGED_CSV="${OUTPUT_DIR}/majority_labels_cc.csv"
DEV_CSV="${OUTPUT_DIR}/deviations_cc.csv"
OUT_JSON="${OUTPUT_DIR}/agreement_metrics.json"
CL_PLOT="${OUTPUT_DIR}/clarity_deviations.png"
REQ_PLOT="${OUTPUT_DIR}/request_deviations.png"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Ensure PYTHONPATH sees the root modules
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Execute the agreement script
python judge_eval_mt_cc/evaluate_agreement_users.py \
    --input_dir "$INPUT_DIR" \
    --output_csv "$MERGED_CSV" \
    --deviation_csv "$DEV_CSV" \
    --output_json "$OUT_JSON" \
    --clarity_plot "$CL_PLOT" \
    --request_plot "$REQ_PLOT" \
    --items_mode union_error_on_missing \
    --tie_break max
