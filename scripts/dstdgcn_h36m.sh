export CUDA_VISIBLE_DEVICES=3
cd ..

# variables
time=$(date "+%Y%m%d")
exp_name=${time}_"dstdgcn_h36m_train_motion"
exp_name="20211202_DSTDGCN_default_test"
save_dir="runs/"$exp_name

# check save directory
if [ ! -d ${save_dir} ]; then
    mkdir -p ${save_dir}
fi

# scripts
# training
# python main.py --exp_name ${exp_name} --run_dir ${save_dir} --config configs/dstdgcn/dstdgcn_h36m.yaml
# nohup python main.py --exp_name ${exp_name} --run_dir ${save_dir} --config configs/dstdgcn/dstdgcn_h36m.yaml > ${save_dir}/out.log 2>&1 &

# testing
python main.py --exp_name ${exp_name} --run_dir ${save_dir} --config configs/dstdgcn/dstdgcn_h36m_test.yaml