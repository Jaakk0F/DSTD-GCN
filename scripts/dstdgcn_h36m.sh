export CUDA_VISIBLE_DEVICES=0
cd ..

# Variables
time=$(date "+%Y%m%d")
exp_name=${time}_"dstdgcn_h36m"
save_dir="runs/"$exp_name

# Check save directory
if [ ! -d ${save_dir} ]; then
    mkdir -p ${save_dir}
fi

# Scripts
# Training
python main.py --exp_name ${exp_name} --run_dir ${save_dir} --config configs/dstdgcn/dstdgcn_h36m.yaml

# Testing
python main.py --exp_name ${exp_name} --run_dir ${save_dir} --config configs/dstdgcn/dstdgcn_h36m_test.yaml