model_name=TimeMixer
seq_len=336
d_model=16
d_ff=32
e_layers=2
batch_size=16

down_sampling_layers=3
down_sampling_window=2

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len\_96 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data custom \
  --features M \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --itr 1 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len\_192 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data custom \
  --features M \
  --label_len 0 \
  --pred_len 192 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --itr 1 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len\_336 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data custom \
  --features M \
  --label_len 0 \
  --pred_len 336 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --itr 1 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len\_720 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data custom \
  --features M \
  --label_len 0 \
  --pred_len 720 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --itr 1 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window
