python -m judge_eval_mt.calculate_weights \
  --input_csv ./judge_eval_mt/out/agreement/merged_with_majority.csv \
  --output_json ./judge_eval_mt/out/agreement/critical_rule_full.json \
  --output_rule_json ./judge_eval_mt/out/agreement/critical_rule.json \
  --output_csv ./judge_eval_mt/out/agreement/critical_rule_per_item.csv \
  --threshold_metric f1