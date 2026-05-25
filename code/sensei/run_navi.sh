#!/bin/bash

# Ensure unbuffered Python output
export PYTHONUNBUFFERED=1

# Timestamp for this batch
TIMESTAMP=$(date +"%d-%m-%Y")

BASE_DIR="results/poi-search/${TIMESTAMP}"
mkdir -p "${BASE_DIR}"

echo "Storing all runs in: ${BASE_DIR}"

for SEED in 1
do
  RUN_DIR="${BASE_DIR}/seed_${SEED}"
  mkdir -p "${RUN_DIR}"

  echo "Running seed ${SEED}"

  PYTHONPATH=src stdbuf -oL -eL python src/sensei-chat.py \
      --technology convnavi \
      --chatbot http://127.0.0.1:8000/query \
      --user examples/profiles/poi-search/user_sim_poi_search.yml \
      --personality ./personalities_car/ \
      --save_folder "${RUN_DIR}" \
      --generator_llm "DeepSeek-V3-0324" \
      --judge_llm "gpt-5-mini" \
      --sut_llm "gpt-4o" \
      --population_size 10000 \
      --max_time "00:01:00" \
      --weight_request_orientedness 0.65 \
      --weight_clarity 0.35 \
      --critical_threshold 0.65 \
      --seed ${SEED} \
      --wandb_project "NaviYelp" \
      --shuffle_personalities \
      --no_wandb

      2>&1 | tee "${RUN_DIR}/run.log"
done