python -m judge_eval_mt.evaluate_agreement_users \
  --input_dir /home/lev/Documents/testing/MT-Survey-CC/results_old \
  --output_csv ./judge_eval_mt_cc/out/agreement/merged_with_majority.csv \
  --output_json ./judge_eval_mt_cc/out/agreement/kappa_results.json \
  --deviation_csv ./judge_eval_mt_cc/out/agreement/vote_deviation.csv \
  --clarity_plot ./judge_eval_mt_cc/out/agreement/clarity_deviation.png \
  --request_plot ./judge_eval_mt_cc/out/agreement/request_orientedness_deviation.png \
  --items_mode intersection