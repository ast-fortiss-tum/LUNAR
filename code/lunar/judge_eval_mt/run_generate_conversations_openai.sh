python -m judge_eval_mt.generator.generate_convs_openai \
    --llm_generator  "DEEPSEEK_V3_0324" \
    --features_config "./configs/features_simple_judge_industry.json" \
    --num_conversations 15 \
    --seed 9 \
    --scores 0 1 2