python -m judge_eval_mt.evaluate_agreement_users \
  --input_dir /home/lev/Documents/testing/MT-Survey/results \
  --output_csv ./judge_eval_mt/out/agreement/merged_with_majority.csv \
  --output_json ./judge_eval_mt/out/agreement/kappa_results.json \
  --deviation_csv ./judge_eval_mt/out/agreement/vote_deviation.csv \
  --clarity_plot ./judge_eval_mt/out/agreement/clarity_deviation.png \
  --request_plot ./judge_eval_mt/out/agreement/request_orientedness_deviation.png \
  --critical_plot ./judge_eval_mt/out/agreement/is_critical_deviation.png \
  --items_mode intersection