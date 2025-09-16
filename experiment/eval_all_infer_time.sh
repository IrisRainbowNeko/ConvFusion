# model_names=("sd1_5" "convdiff_sd1_5" "linfusion_sd1_5" "sdxl" "convdiff_sdxl" "linfusion_sdxl")
model_names=("convdiff_sd1_5" "convdiff_sdxl")

cd /mnt/data1/cxzhou/Cnn_as_Attn/conv-diff_triton

for model_name in "${model_names[@]}"; do
    echo "******************************** infer speed of ${model_name} ********************************"
    CUDA_VISIBLE_DEVICES=3 python experiment/run_eval.py --model_name $model_name \
        --eval_n_images 10 --eval_batchsize 1 --calc_infer_cost
    echo "******************************** infer speed of ${model_name} ********************************"
done