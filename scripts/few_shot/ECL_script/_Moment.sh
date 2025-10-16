model_name=Moment
seq_len=512
batch_size=4

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len\_96 \
  --model $model_name \
  --seq_len $seq_len \
  --batch_size $batch_size \
  --data custom \
  --features M \
  --label_len 0 \
  --pred_len 96 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len\_192 \
  --model $model_name \
  --seq_len $seq_len \
  --batch_size $batch_size \
  --data custom \
  --features M \
  --label_len 0 \
  --pred_len 192 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len\_336 \
  --model $model_name \
  --seq_len $seq_len \
  --batch_size $batch_size \
  --data custom \
  --features M \
  --label_len 0 \
  --pred_len 336 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len\_720 \
  --model $model_name \
  --seq_len $seq_len \
  --batch_size $batch_size \
  --data custom \
  --features M \
  --label_len 0 \
  --pred_len 720 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1