model_name=PatchTST_Adapter
d_model=32		# 64
d_ff=64			# 128
n_heads=4		# 8
e_layers=4
batch_size=16

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model $model_name \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data UEA \
  --des 'Exp' \
  --train_epochs 100 \
  --patience 10 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model $model_name \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data UEA \
  --des 'Exp' \
  --train_epochs 100 \
  --patience 10 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Handwriting/ \
  --model_id Handwriting \
  --model $model_name \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data UEA \
  --des 'Exp' \
  --train_epochs 100 \
  --patience 10 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Heartbeat/ \
  --model_id Heartbeat \
  --model $model_name \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data UEA \
  --des 'Exp' \
  --train_epochs 100 \
  --patience 10 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/JapaneseVowels/ \
  --model_id JapaneseVowels \
  --model $model_name \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data UEA \
  --des 'Exp' \
  --train_epochs 100 \
  --patience 10 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PEMS-SF/ \
  --model_id PEMS-SF \
  --model $model_name \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 4 \
  --data UEA \
  --des 'Exp' \
  --train_epochs 100 \
  --patience 10 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP1/ \
  --model_id SelfRegulationSCP1 \
  --model $model_name \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data UEA \
  --des 'Exp' \
  --train_epochs 100 \
  --patience 10 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP2/ \
  --model_id SelfRegulationSCP2 \
  --model $model_name \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data UEA \
  --des 'Exp' \
  --train_epochs 100 \
  --patience 10 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SpokenArabicDigits/ \
  --model_id SpokenArabicDigits \
  --model $model_name \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data UEA \
  --des 'Exp' \
  --train_epochs 100 \
  --patience 10 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UWaveGestureLibrary/ \
  --model_id UWaveGestureLibrary \
  --model $model_name \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data UEA \
  --des 'Exp' \
  --train_epochs 100 \
  --patience 10 \
  --learning_rate 0.001 \
  --itr 1
