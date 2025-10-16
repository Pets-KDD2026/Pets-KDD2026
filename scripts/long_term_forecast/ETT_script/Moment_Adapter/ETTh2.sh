model_name=Moment_Adapter
seq_len=512
batch_size=16

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len\_96 \
  --model $model_name \
  --seq_len $seq_len \
  --batch_size $batch_size \
  --data ETTh2 \
  --features M \
  --label_len 0 \
  --pred_len 96 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
   --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len\_192 \
  --model $model_name \
  --seq_len $seq_len \
  --batch_size $batch_size \
  --data ETTh2 \
  --features M \
  --label_len 0 \
  --pred_len 192 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len\_336 \
  --model $model_name \
  --seq_len $seq_len \
  --batch_size $batch_size \
  --data ETTh2 \
  --features M \
  --label_len 0 \
  --pred_len 336 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len\_720 \
  --model $model_name \
  --seq_len $seq_len \
  --batch_size $batch_size \
  --data ETTh2 \
  --features M \
  --label_len 0 \
  --pred_len 720 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1