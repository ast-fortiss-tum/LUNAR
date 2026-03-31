python -m judge_eval_mt.generator.generate_convs_carcontrol_openai \
    --llm_generator  "GPT-4O" \
    --features_config "./configs/features_simple_judge_cc.json" \
    --num_conversations 3 \
    --seed 9 \
    --scores 0 1 2 \
    --min_turns 3 \
    --max_turns 6