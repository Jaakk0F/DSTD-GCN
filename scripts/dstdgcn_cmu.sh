export CUDA_VISIBLE_DEVICES=0
cd ..

# variables
time=$(date "+%Y%m%d")
exp_name=${time}_"dstdgcn_cmu_test"
save_dir="runs/"$exp_name

# check save directory
if [ ! -d ${save_dir} ]; then
    mkdir -p ${save_dir}
fi

# scripts
# training
# python main.py --exp_name ${exp_name} --run_dir ${save_dir} --config configs/dstdgcn/dstdgcn_cmu.yaml
# nohup python main.py --exp_name ${exp_name} --run_dir ${save_dir} --config configs/dstdgcn/dstdgcn_cmu.yaml > ${save_dir}/out.log 2>&1 &

# testing
# exp_name="20211202_DSTDGCN_default_train_scene1"
python main.py --exp_name ${exp_name} --run_dir ${save_dir} --config configs/dstdgcn/dstdgcn_cmu_test.yaml