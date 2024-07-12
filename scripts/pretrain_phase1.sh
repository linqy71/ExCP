#! /bin/bash

export LOGLEVEL=INFO

export OMP_NUM_THREADS=1

GPUS_PER_NODE=1

DIST_ARGS="--nproc_per_node $GPUS_PER_NODE --master_port=16000 --nnodes 1 --node_rank 0 --rdzv_conf timeout=5400"
echo $DIST_ARGS

WORKSPACE='./LMtrainer'
cd $WORKSPACE

lr=3e-4
adam_beta1=0.9
adam_beta2=0.95
weight_decay=0.1
pretrained_model=./model/
data_dir=$1
seed=2032
OUTPUT_DIR='/cache/models'
gradient_accumulation_steps=1
MP_SIZE=1
set +x
torchrun $DIST_ARGS \
                pretrain.py \
                --output_dir $OUTPUT_DIR \
                --model_name_or_path ${pretrained_model} \
                --overwrite_output_dir \
                --validation_split_percentage 0.00004 \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 1 \
                --do_train \
                --seed $seed \
                --logging_strategy steps \
                --save_strategy steps \
                --save_steps 1000 \
                --save_total_limit 1000 \
                --gradient_accumulation_steps ${gradient_accumulation_steps} \
                --preprocessing_num_workers 4 \
                --model_max_length 1024 \
                --output_dir $OUTPUT_DIR \
                --logging_first_step True \
                --num_train_epochs 1 \
                --fp16 True \
                --report_to tensorboard \
                --logging_dir $OUTPUT_DIR/tensorboard \
                --logging_steps 1 \
                --evaluation_strategy steps \
                --eval_steps 100000000 \
                --fp16_full_eval \
                --gradient_checkpointing True \
                --flash_attention True \
                --model_parallel_size $MP_SIZE \
                --lr_scheduler_type cosine \
                --learning_rate ${lr} \
                --adam_beta1 ${adam_beta1} \
                --adam_beta2 ${adam_beta2} \
                --weight_decay ${weight_decay} \
                --warmup_steps 1000 \
                --ddp_timeout 5400 \
                --dataset_dir $data_dir \
                # --data_cache_dir $data_dir \
                # --read_cached \

python ../compress_pythia.py $OUTPUT_DIR/checkpoint-2000 $OUTPUT_DIR/checkpoint-1000 --output $OUTPUT_DIR/checkpoint-2000 --recon
python ../compress_pythia.py $OUTPUT_DIR/checkpoint-3000 $OUTPUT_DIR/checkpoint-2000 --output $OUTPUT_DIR/checkpoint-3000 --recon
python ../compress_pythia.py $OUTPUT_DIR/checkpoint-4000 $OUTPUT_DIR/checkpoint-3000 --output $OUTPUT_DIR/checkpoint-4000 --recon
python ../compress_pythia.py $OUTPUT_DIR/checkpoint-5000 $OUTPUT_DIR/checkpoint-4000 --output $OUTPUT_DIR/checkpoint-5000 --recon

for iter in {5000..19000..5000}
do
	torchrun $DIST_ARGS \
                pretrain.py \
                --output_dir $OUTPUT_DIR \
                --model_name_or_path ${pretrained_model} \
                --overwrite_output_dir \
                --validation_split_percentage 0.00004 \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 1 \
                --do_train \
                --resume_from_checkpoint $OUTPUT_DIR/checkpoint-$iter \
                --seed $seed \
                --logging_strategy steps \
                --save_strategy steps \
                --save_steps 1000 \
                --save_total_limit 1000 \
                --gradient_accumulation_steps ${gradient_accumulation_steps} \
                --preprocessing_num_workers 4 \
                --model_max_length 1024 \
                --output_dir $OUTPUT_DIR \
                --logging_first_step True \
                --num_train_epochs 1 \
                --fp16 True \
                --report_to tensorboard \
                --logging_dir $OUTPUT_DIR/tensorboard \
                --logging_steps 1 \
                --evaluation_strategy steps \
                --eval_steps 100000000 \
                --fp16_full_eval \
                --gradient_checkpointing True \
                --flash_attention True \
                --model_parallel_size $MP_SIZE \
                --lr_scheduler_type cosine \
                --learning_rate ${lr} \
                --adam_beta1 ${adam_beta1} \
                --adam_beta2 ${adam_beta2} \
                --weight_decay ${weight_decay} \
                --warmup_steps 1000 \
                --ddp_timeout 5400 \
                --dataset_dir $data_dir \
                # --data_cache_dir $data_dir \
                # --read_cached \

        next_iter=$[iter+5000]
        last_iter=$[iter+4000]
        last_iter2=$[iter+3000]
        last_iter3=$[iter+2000]
        last_iter4=$[iter+1000]
        python ../compress_pythia.py $OUTPUT_DIR/checkpoint-$last_iter4 $OUTPUT_DIR/checkpoint-$iter --output $OUTPUT_DIR/checkpoint-$last_iter4 --recon
        python ../compress_pythia.py $OUTPUT_DIR/checkpoint-$last_iter3 $OUTPUT_DIR/checkpoint-$last_iter4 --output $OUTPUT_DIR/checkpoint-$last_iter3 --recon
        python ../compress_pythia.py $OUTPUT_DIR/checkpoint-$last_iter2 $OUTPUT_DIR/checkpoint-$last_iter3 --output $OUTPUT_DIR/checkpoint-$last_iter2 --recon
        python ../compress_pythia.py $OUTPUT_DIR/checkpoint-$last_iter $OUTPUT_DIR/checkpoint-$last_iter2 --output $OUTPUT_DIR/checkpoint-$last_iter --recon
        python ../compress_pythia.py $OUTPUT_DIR/checkpoint-$next_iter $OUTPUT_DIR/checkpoint-$last_iter --output $OUTPUT_DIR/checkpoint-$next_iter --recon
done

# tensorboard --logdir_spec=excp:~/Excp/LMtrainer/cache/models5/tensorboard,ours-beta2:~/Excp/LMtrainer/cache/models7/tensorboard,ours-beta1:~/Excp/LMtrainer/cache/models6/tensorboard,origin:~/Excp/LMtrainer/cache/models8/tensorboard --bind_all

