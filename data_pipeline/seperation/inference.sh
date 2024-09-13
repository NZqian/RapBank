input_dir=$1
output_dir=$2
ckpt_dir=$3

python3 inference.py \
    --model_type bs_roformer \
    --config_path ${ckpt_dir}/model_bs_roformer_ep_317_sdr_12.9755.yaml \
    --start_check_point ${ckpt_dir}/model_bs_roformer_ep_317_sdr_12.9755.ckpt \
    --input_folder ${input_dir} \
    --store_dir ${output_dir} \
    --extract_instrumental