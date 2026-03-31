#!/bin/bash

# Create timestamp for this batch
for SEED in 1 2 3 4 5 6
do
  TIMESTAMP=$(date +"%d-%m-%Y_%H-%M-%S")
  RESULTS_DIR="results/${TIMESTAMP}"
  RUN_DIR="${RESULTS_DIR}/seed_${SEED}_${TIMESTAMP}/"

  mkdir -p "${RUN_DIR}"

  echo "Running seed ${SEED}, output -> ${RUN_DIR}"

  PYTHONUNBUFFERED=1 LLM_IPA="gpt-5-chat" python run_mt_car_control_discrete.py \
      --algorithm nsga2 \
      --n 15 \
      --sut ipa_yelp \
      --llm_ipa gpt-4o \
      --llm_intent_classifier DeepSeek-V3-0324  \
      --llm_judge gpt-5-mini \
      --llm_generator DeepSeek-V3-0324  \
      --features_config configs/features_simple_judge_cc.json \
      --max_time "03:00:00" \
      --wandb_project "CarControlYELP" \
      --store_turns_details \
      --seed ${SEED} \
      --weight_clarity 0.35 \
      --weight_request_orientedness 0.65 \
      --th_dims 0.65 \
      --th_efficiency 0.65 \
      --th_effectiveness 0.75 \
      --save_folder "${RUN_DIR}" \
       2>&1 | tee "${RUN_DIR}/run.log"
done