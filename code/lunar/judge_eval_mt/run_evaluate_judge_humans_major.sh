python -m judge_eval_mt.evaluate_judge_humans_major \
    --max_files 15 \
    --data_folder "./judge_eval_mt/generator/new_data/" \
    --majority_csv "./judge_eval_mt/out/agreement/merged_with_majority.csv" \
    --output_folder "./judge_eval_mt/out_tuned/judge_vs_humans_major/" \
    --llm_types "DEEPSEEK_V3_0324" "GPT_5_MINI" "GPT_4O_MINI" "GPT_4O"