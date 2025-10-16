model_name=Moment
seq_len=512
batch_size=2

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
 --root_path ./dataset/PEMS/ \
 --data_path PEMS03.npz \
 --model_id PEMS03 \
  --model $model_name \
  --seq_len $seq_len \
  --batch_size $batch_size \
 --data PEMS \
  --features M \
  --label_len 0 \
  --pred_len 12 \
  --d_layers 0 \
  --factor 3 \
 --enc_in 358 \
 --dec_in 358 \
 --c_out 358 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS04.npz \
  --model_id PEMS04 \
  --model $model_name \
  --seq_len $seq_len \
  --batch_size $batch_size \
  --data PEMS \
  --features M \
  --label_len 0 \
  --pred_len 12 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
 --root_path ./dataset/PEMS/ \
 --data_path PEMS07.npz \
 --model_id PEMS07 \
  --model $model_name \
  --seq_len $seq_len \
  --batch_size $batch_size \
  --data PEMS \
  --features M \
  --label_len 0 \
  --pred_len 12 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 883 \
  --dec_in 883 \
  --c_out 883 \
   --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
 --root_path ./dataset/PEMS/ \
 --data_path PEMS08.npz \
 --model_id PEMS08 \
  --model $model_name \
  --seq_len $seq_len \
  --batch_size $batch_size \
  --data PEMS \
  --features M \
  --label_len 0 \
  --pred_len 12 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 170 \
 --dec_in 170 \
 --c_out 170 \
   --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1
