export CUDA_VISIBLE_DEVICES=0
cd ..

# Variables
time=$(date "+%Y%m%d")
exp_name=${time}_"dstdgcn_3dpw"
save_dir="runs/"$exp_name

# Check save directory
if [ ! -d ${save_dir} ]; then
    mkdir -p ${save_dir}
fi

# Scripts
# Training
python main.py --exp_name ${exp_name} --run_dir ${save_dir} --config configs/dstdgcn/dstdgcn_3dpw.yaml

# Testing
# By default, it will load the pretrained checkpoint
# For your own checkpoint, please modify the `ckpt` in the config file
python main.py --exp_name ${exp_name} --run_dir ${save_dir} --config configs/dstdgcn/dstdgcn_3dpw_test.yaml