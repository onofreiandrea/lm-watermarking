python run_watermarking.py --model_name facebook/opt-1.3b --dataset_name c4 --dataset_config_name realnewslike --max_new_tokens 200 --min_prompt_tokens 50 --limit_indices 50 --bl_proportion 0.75 --bl_logit_bias 0.5 --num_beams 1 --use_sampling True --sampling_temp 0.7 --output_dir ./all_runs_delta_0.5

python run_watermarking.py --model_name facebook/opt-1.3b --dataset_name c4 --dataset_config_name realnewslike --max_new_tokens 200 --min_prompt_tokens 50 --limit_indices 50 --bl_proportion 0.75 --bl_logit_bias 1.0 --num_beams 1 --use_sampling True --sampling_temp 0.7 --output_dir ./all_runs_delta_1.0

python run_watermarking.py --model_name facebook/opt-1.3b --dataset_name c4 --dataset_config_name realnewslike --max_new_tokens 200 --min_prompt_tokens 50 --limit_indices 50 --bl_proportion 0.75 --bl_logit_bias 2.0 --num_beams 1 --use_sampling True --sampling_temp 0.7 --output_dir ./all_runs_delta_2.0

python run_watermarking.py --model_name facebook/opt-1.3b --dataset_name c4 --dataset_config_name realnewslike --max_new_tokens 200 --min_prompt_tokens 50 --limit_indices 50 --bl_proportion 0.75 --bl_logit_bias 5.0 --num_beams 1 --use_sampling True --sampling_temp 0.7 --output_dir ./all_runs_delta_5.0

python run_watermarking.py --model_name facebook/opt-1.3b --dataset_name c4 --dataset_config_name realnewslike --max_new_tokens 200 --min_prompt_tokens 50 --limit_indices 50 --bl_proportion 0.75 --bl_logit_bias 10.0 --num_beams 1 --use_sampling True --sampling_temp 0.7 --output_dir ./all_runs_delta_10.0